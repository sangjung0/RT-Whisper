import importlib, kss, os
print("kss:", kss.__version__)
print("has mecab module:", importlib.util.find_spec("mecab") is not None)
try:
    from konlpy.tag import Mecab
    m = Mecab(dicpath="/usr/local/lib/mecab/dic/mecab-ko-dic")
    print("konlpy.Mecab OK:", m.pos("형태소 테스트")[:3])
except Exception as e:
    print("konlpy.Mecab ERR ->", repr(e))
