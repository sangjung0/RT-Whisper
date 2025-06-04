from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class VadPostParam:
    candidate_tokens: list[Token]
    vad_timestamps: list[dict[str, int]]
    prev_vad_timestamps_mapping: list[dict[str, int]]

    @staticmethod
    def validate(context: Context) -> bool:
        return context.merged_vad_chunk_size > 0

    @staticmethod
    def from_context(context: Context):
        if VadPostParam.validate(context):
            return VadPostParam(
                candidate_tokens=context.candidate_tokens,
                vad_timestamps=context.vad_timestamps,
                prev_vad_timestamps_mapping=context.prev_vad_timestamps_mapping,
            )
        return None
