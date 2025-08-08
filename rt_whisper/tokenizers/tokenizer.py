import pysbd

from tokenizers.kss_tokenizer import KSSTokenizer


class Tokenizer:
    tokenizers: dict[pysbd.Segmenter] = {}

    @staticmethod
    def get_tokenizer(lang: str) -> pysbd.Segmenter:
        # whisper와 지원하는 언어 차이를 확인해야 한다.
        lang = lang.strip()
        if lang not in Tokenizer.tokenizers:
            if lang == "ko":
                Tokenizer.tokenizers[lang] = KSSTokenizer()
            else:
                try:
                    Tokenizer.tokenizers[lang] = pysbd.Segmenter(language=lang)
                except Exception as e:
                    print(
                        f"Tokenizer not found for {lang}. Error: {e}. 이거 whisper하고 비교해서 제대로 되게 만들어야 함"
                    )
                    return None
        return Tokenizer.tokenizers[lang]


__all__ = ["Tokenizer"]
