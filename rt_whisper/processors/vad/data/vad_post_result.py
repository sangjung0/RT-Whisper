from dataclasses import dataclass

from rt_whisper.data import Context  # , Token


@dataclass(slots=True)
class VadPostResult:
    vad_timestamps_mapping: list[dict[str, int]]
    merged_vad_timestamps_mapping: list[dict[str, int]]
    # tokens: list[Token]

    def update_context(self, context: Context):
        context.vad_timestamps_mapping = self.vad_timestamps_mapping
        context.merged_timestamps_mapping = self.merged_vad_timestamps_mapping
        # context.merged_candidate_tokens = tokens
