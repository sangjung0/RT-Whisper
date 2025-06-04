import kss


class KSSTokenizer:
    # KSSTokenizer likes Segmenter
    def segment(self, text) -> list[str] | list[list[str]]:
        # Use kss to segment the text
        return kss.split_sentences(text)
