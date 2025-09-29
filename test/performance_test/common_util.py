import os
import sys

os.chdir("/workspaces/dev")
paths = [
    "/workspaces/dev/test/modules/whisper_streaming",
    "/workspaces/dev/test/modules/simul_whisper",
]
for path in paths:
    sys.path.append(os.path.abspath(path))

import torch
import numpy as np

from pathlib import Path
from typing import Callable
from whisper.normalizers import EnglishTextNormalizer

from sj_utils.file.json import JsonSaver
from sj_utils.evaluator import TimeChecker
from sj_utils.evaluator.asr import TimeEvaluator, TimeEvaluatorSummary
from sj_utils.audio import segment_audio
from sj_utils.collection import SafetyDict
from sj_ai_utils.datasets import Dataset
from sj_ai_utils.asr.whisper_utils import segments_to_text
from sj_ai_utils.evaluator.sclite_utils import (
    TRNFormat,
    sclite_trn,
    parse_sclite_summary,
)

normalizer = EnglishTextNormalizer()


def normalize_text(text: str):
    return normalizer(text)


def test_process(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, TimeEvaluator], str],
    all: bool,
) -> dict:
    result_ref = []
    result_hyp = []
    processed_time = TimeChecker()
    transcribe_time = TimeEvaluatorSummary()

    with processed_time.timeit():
        for _id, audio, text in dataset:
            te = TimeEvaluator(L=len(audio))

            txt = normalize_text(text)
            ref = TRNFormat(id=_id, text=txt)

            pred = transcriber(audio, te)
            pred = normalize_text(pred)
            hyp = TRNFormat(id=_id, text=pred)

            result_ref.append(ref)
            result_hyp.append(hyp)
            transcribe_time.add(te)

    if all:
        output = sclite_trn(result_ref, result_hyp)
        result = parse_sclite_summary(output)
    else:
        result = {}
        for r, h in zip(result_ref, result_hyp):
            output = sclite_trn([r], [h])
            result[r.id] = parse_sclite_summary(output)

    result["processed_time"] = processed_time.metric()
    result["transcribe_time"] = transcribe_time.metric()

    return result


def test_process_for_rt(
    dataset: Dataset,
    transcriber: Callable[[np.ndarray, TimeEvaluator, TimeEvaluator, str], str],
    all: bool,
) -> dict:
    result_ref = []
    result_hyp = []
    processed_time = TimeChecker()
    completed = TimeEvaluatorSummary()
    candidate = TimeEvaluatorSummary()

    with processed_time.timeit():
        for _id, audio, text in dataset:
            te = TimeEvaluator(L=len(audio))
            te2 = TimeEvaluator(L=len(audio))

            txt = normalize_text(text)
            ref = TRNFormat(id=_id, text=txt)

            pred = transcriber(audio, te, te2, _id)
            pred = normalize_text(pred)
            hyp = TRNFormat(id=_id, text=pred)

            result_ref.append(ref)
            result_hyp.append(hyp)
            completed.add(te)
            candidate.add(te2)

    if all:
        output = sclite_trn(result_ref, result_hyp)
        result = parse_sclite_summary(output)
    else:
        result = {}
        for r, h in zip(result_ref, result_hyp):
            output = sclite_trn([r], [h])
            result[r.id] = parse_sclite_summary(output)

    result["processed_time"] = processed_time.metric()
    result["completed_time"] = completed.metric()
    result["candidate_time"] = candidate.metric()

    return result


def get_simul_whisper_transcriber(model_size: str, language: str, chunk_size: int):
    from simul_whisper.transcriber.config import AlignAttConfig
    from simul_whisper.transcriber.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
    from simul_whisper.whisper.audio import N_FFT, HOP_LENGTH, SAMPLE_RATE

    class Segment:
        def __init__(self, audio: np.ndarray, samples_to_read, samples_in_chunk):
            self.audio = torch.from_numpy(audio).float()
            self.audio_len_s = self.audio.shape[0] / SAMPLE_RATE
            self.samples_to_read = samples_to_read
            self.samples_in_chunk = samples_in_chunk
            self.buffer_len = samples_in_chunk - samples_to_read

        def __iter__(self):
            frames_in_chunk = self.audio[: self.samples_in_chunk]
            read_pointer = frames_in_chunk.shape[0]
            yield frames_in_chunk, (read_pointer >= self.audio.shape[0]), min(read_pointer, self.audio.shape[0])
            while read_pointer < self.audio.shape[0]:
                frames_in_chunk = torch.cat(
                    (
                        frames_in_chunk[-self.buffer_len :],
                        self.audio[read_pointer : read_pointer + self.samples_to_read],
                    ),
                    dim=0,
                )
                read_pointer += self.samples_to_read
                yield frames_in_chunk, (read_pointer >= self.audio.shape[0]), min(read_pointer, self.audio.shape[0])

    class SegmentWrapper(Segment):
        def __init__(self, audio: np.ndarray, segment_length):
            frames_to_read = int((segment_length * SAMPLE_RATE) / HOP_LENGTH)
            samples_to_read = frames_to_read * HOP_LENGTH
            samples_in_chunk = samples_to_read + N_FFT - HOP_LENGTH
            super().__init__(
                audio,
                samples_to_read=samples_to_read,
                samples_in_chunk=samples_in_chunk,
            )

    cfg = AlignAttConfig(
        model_path=model_size,
        segment_length=chunk_size / SAMPLE_RATE,
        frame_threshold=12,
        language=language,
        buffer_len=20,
        min_seg_len=0.0,
        if_ckpt_path=f"/workspaces/dev/test/modules/simul_whisper/cif_models/{model_size}.pt",
    )
    model = PaddedAlignAttWhisper(cfg)

    def transcriber(
        audio: np.ndarray,
        transcribe_time: TimeEvaluator,
    ) -> str:
        hyp_list = []
        for seg, is_last, read_pointer in SegmentWrapper(audio, cfg.segment_length):
            with transcribe_time.timeit():
                new_toks = model.infer(seg, is_last)
                hyp_list.append(new_toks)
                hyp = torch.cat(hyp_list, dim=0)
                hyp = hyp[hyp < DEC_PAD]
                hyp = model.tokenizer.decode(hyp)
            transcribe_time.add_coverage(
                np.full(len(hyp.split()), read_pointer, dtype=np.float32)
            )
        model.refresh_segment(complete=True)
        return hyp

    return transcriber


def get_whisper_streaming_transcriber(model_size: str, language: str, chunk_size: int):
    from whisper_online import FasterWhisperASR, OnlineASRProcessor

    asr = FasterWhisperASR(language, model_size)
    asr.use_vad()
    online = OnlineASRProcessor(asr)

    def transcriber(
        audio: np.ndarray,
        transcribe_time: TimeEvaluator,
    ) -> str:
        online.init()
        full_text = ""
        input_length = 0
        for segment in segment_audio(audio, mean=chunk_size):
            input_length += len(segment)
            with transcribe_time.timeit():
                online.insert_audio_chunk(segment)
                _, _, text = online.process_iter()
            full_text += text
            transcribe_time.add_coverage(
                np.full(len(text.split()), input_length, dtype=np.float32)
            )
        _, _, text = online.finish()
        full_text += text
        transcribe_time.add_coverage(
            np.full(len(text.split()), input_length, dtype=np.float32)
        )
        return full_text

    return transcriber


def get_faster_whisper_transcriber(model_size: str, language: str):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcriber(audio: np.ndarray, transcribe_time: TimeEvaluator) -> str:
        with transcribe_time.timeit():
            segments, _ = model.transcribe(
                audio,
                language=language,
                word_timestamps=True,
            )
            text = segments_to_text(segments)
        transcribe_time.add_coverage(
            np.full(len(text.split()), len(audio), dtype=np.float32)
        )

        return text

    return transcriber


def get_rt_whisper_transcriber(
    model_size: str,
    language: str,
    chunk_size: int,
    hyperparameter: SafetyDict | Path,
    use_prompt: bool,
):
    from rt_whisper.data import Param, Result
    from rt_whisper.streamers import get_token_streamer_with_vad_v2_min_filter

    token_streamer = get_token_streamer_with_vad_v2_min_filter(
        hyperparameter=hyperparameter, model_size_or_path=model_size
    )

    def transcriber(
        audio: np.ndarray,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
    ) -> str:

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for segment in segment_audio(audio, mean=chunk_size):
            input_length += len(segment)
            param.chunk = segment
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=use_prompt)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    return transcriber


def get_token_saver_loader_transcriber(
    model_size: str,
    language: str,
    chunk_size: int,
    hyperparameter: SafetyDict | Path,
    storage: Path,
):
    from rt_whisper import saveloaders
    from rt_whisper.data import Param, Result
    from rt_whisper.utils import init_hyperparameter

    hyperparameter = init_hyperparameter(hyperparameter, model_size_or_path=model_size)
    overlap_duration = hyperparameter["asr"]["max_overlap_duration"]

    def token_saver(
        audio: np.ndarray,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        save_path: Path,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_saver(
            save_path=save_path, hyperparameter=hyperparameter
        )

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for segment in segment_audio(audio, mean=chunk_size):
            input_length += len(segment)
            param.chunk = segment
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def token_loader(
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        save_path: Path,
    ) -> str:
        token_streamer = saveloaders.get_token_streamer_loader(
            saved_path=save_path, hyperparameter=hyperparameter
        )

        completed = []
        param = Param()
        input_length = 0
        end = 0
        for _ in range(len(list(save_path.iterdir()))):
            input_length += 1
            param.language = language
            with completed_time.timeit():
                result: Result = token_streamer.process(param)
            completed.extend(result.completed)
            param.update(result, update_prompt=False)

            completed_tokens = [t for t in result.completed_tokens if t.is_word]
            candidate_tokens = [
                t for t in result.candidate_tokens if t.is_word and t.start >= end
            ]
            end = candidate_tokens[-1].end if candidate_tokens else end
            completed_time.add_coverage(
                np.full(len(completed_tokens), input_length, dtype=np.float32)
            )
            candidate_time.add_coverage(
                np.full(len(candidate_tokens), input_length, dtype=np.float32)
            )

        completed.extend(result.candidate)
        text = " ".join([s.text for s in completed])
        return text

    def transcriber(
        audio: np.ndarray,
        completed_time: TimeEvaluator,
        candidate_time: TimeEvaluator,
        audio_key: Path | str,
    ) -> str:
        if hyperparameter is None:
            raise ValueError("hyperparameter must be provided")

        saved_path = storage / f"{overlap_duration}" / audio_key

        if saved_path.exists():
            return token_loader(
                completed_time,
                candidate_time,
                saved_path,
            )
        return token_saver(
            audio,
            completed_time,
            candidate_time,
            saved_path,
        )

    return transcriber


def simul_whisper(
    dataset: Dataset,
    model_size: str,
    language: str,
    chunk_size: int,
    test_all: bool,
):
    print("Running Simul Whisper...")

    transcriber = get_simul_whisper_transcriber(model_size, language, chunk_size)
    result = test_process(dataset=dataset, transcriber=transcriber, all=test_all)

    del transcriber
    return result


def whisper_streaming(
    dataset: Dataset,
    model_size: str,
    language: str,
    chunk_size: int,
    test_all: bool,
):
    print("Running Whisper Streaming...")

    transcriber = get_whisper_streaming_transcriber(model_size, language, chunk_size)
    result = test_process(dataset=dataset, transcriber=transcriber, all=test_all)

    del transcriber
    return result


def rt_whisper(
    dataset: Dataset,
    model_size: str,
    language: str,
    chunk_size: int,
    test_all: bool,
    storage: Path,
    use_save_loader: bool,
    use_prompt: bool,
    hyperparameter: Path | SafetyDict,
):
    print("Running RT Whisper...")

    if use_save_loader:
        transcriber = get_token_saver_loader_transcriber(
            model_size=model_size,
            language=language,
            chunk_size=chunk_size,
            hyperparameter=hyperparameter,
            storage=storage,
        )

    else:
        t = get_rt_whisper_transcriber(
            model_size=model_size,
            language=language,
            chunk_size=chunk_size,
            hyperparameter=hyperparameter,
            use_prompt=use_prompt,
        )

        def transcriber(
            audio: np.ndarray,
            completed_time: TimeChecker,
            candidate_time: TimeChecker,
            _: str,
        ) -> str:
            return t(audio, completed_time, candidate_time)

    result = test_process_for_rt(dataset=dataset, transcriber=transcriber, all=test_all)

    del transcriber
    if t:
        del t
    return result


def whisper(
    dataset: Dataset,
    model_size: str,
    language: str,
    test_all: bool,
):
    print("Running Whisper...")

    transcriber = get_faster_whisper_transcriber(model_size, language)
    result = test_process(dataset=dataset, transcriber=transcriber, all=test_all)

    del transcriber
    return result


def evaluate(
    output_path: Path,
    description: str,
    dataset: Dataset,
    models: list[str],
    model_size: str,
    chunk_size: int,
    language: str = "en",
    test_all: bool = True,
    storage: Path = None,
    use_save_loader: bool = True,
    use_prompt: bool = False,
    hyperparameter: Path | dict = None,
):
    json_saver = JsonSaver(description)

    results = {}
    for key in models:
        if key == "whisper":
            results[key] = whisper(dataset, model_size, language, test_all)
        elif key == "rt_whisper":
            results[key] = rt_whisper(
                dataset,
                model_size,
                language,
                chunk_size,
                test_all,
                storage,
                use_save_loader,
                use_prompt,
                hyperparameter,
            )
        elif key == "whisper_streaming":
            results[key] = whisper_streaming(
                dataset, model_size, language, chunk_size, test_all
            )
        elif key == "simul_whisper":
            results[key] = simul_whisper(
                dataset, model_size, language, chunk_size, test_all
            )

    json_saver.save(results, output_path)
    print(f"Results saved to {output_path}")


__all__ = [
    "whisper_streaming",
    "rt_whisper",
    "whisper",
    "evaluate",
]
