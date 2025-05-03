from RTWhisper import Pipeline
from RTWhisper.data import Context


class AdjustWeightAndOffset(Pipeline):
    def __init__(
        self,
        BOUNDARY: float,
    ):
        super().__init__()
        self.__BOUNDARY = BOUNDARY

    def __get_weighted_probability(
        self,
        probabilities: float,
        start_sc: int,
        end_sc: int,
        duration_sc: int,
        boundary: int,
    ) -> float:
        center = (start_sc + end_sc) / 2
        if center < duration_sc - boundary:
            return probabilities
        return probabilities * ((duration_sc - center) / boundary)

    def can_process(self, context: Context) -> bool:
        if context.merged_processed_audio_sc == 0:
            return False
        return (
            context.sc_offset,
            context.audio_sc,
            context.prev_audio_sc,
            context.merged_candidate_tokens,
        )

    def compute_process(self, param: tuple):
        sc_offset, audio_sc, prev_audio_sc, tokens = param
        audio_sc = audio_sc + prev_audio_sc

        for token in tokens:
            start = token.start
            end = token.end
            token.start = start + sc_offset
            token.end = end + sc_offset
            if not token.is_word:
                continue

            token.probability = self.__get_weighted_probability(
                token.probability, start, end, audio_sc, self.__BOUNDARY
            )

        return tokens

    def apply_process(self, context: Context, result):
        pass
        # context.merged_candidate_tokens = result
