from dataclasses import dataclass

from rt_whisper.data import Context, Token


@dataclass(slots=True)
class PositionWeightedFilterParam:
    prev_chunk_offset: int
    chunk_size: int
    prev_chunk_size: int
    candidate_tokens: list[Token]

    @staticmethod
    def validate(context: Context) -> "PositionWeightedFilterParam":
        return context.merged_vad_chunk_size > 0

    @staticmethod
    def from_context(context: Context) -> "PositionWeightedFilterParam":
        if PositionWeightedFilterParam.validate(context):
            return PositionWeightedFilterParam(
                prev_chunk_offset=context.prev_chunk_offset,
                chunk_size=context.chunk_size,
                prev_chunk_size=context.prev_chunk_size,
                candidate_tokens=context.candidate_tokens,
            )
        return None
