"""1단계: 형태소 분석기(Kiwi)로 장단음 판별 대상(동음이의어 후보)을 찾는다.

형태소 분석기는 '눈이'를 '눈(명사) + 이(조사)'로 나눠 주기 때문에
조사·어미가 붙어 있어도 명사만 정확히 찾아낼 수 있다.
하지만 '눈'이 目인지 雪인지는 형태소 분석기도 구별하지 못한다(둘 다 NNG).
그 판단은 2단계(disambiguate.py)에서 문맥을 보고 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from kiwipiepy import Kiwi

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon.json"
NOUN_TAGS = {"NNG", "NNB", "NNP"}

_kiwi: Kiwi | None = None


def get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def load_lexicon(path: Path = LEXICON_PATH) -> dict:
    lex = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in lex.items() if not k.startswith("_")}


@dataclass
class Candidate:
    form: str            # 형태소 표기 (예: '눈')
    tag: str             # 품사 (예: 'NNG')
    start: int           # 문장 안에서의 글자 위치
    word_initial: bool   # 어절(단어)의 첫머리인가? (표준 발음법 제6항)
    senses: list = field(default_factory=list)
    sense_id: str | None = None   # 판별 결과
    reason: str = ""

    @property
    def end(self) -> int:
        return self.start + len(self.form)

    def long_char_offsets(self) -> list[int]:
        """판별된 의미에 따라 '실제로 길게 발음해야 하는' 음절의 문장 내 위치."""
        if self.sense_id is None:
            return []
        sense = next(s for s in self.senses if s["id"] == self.sense_id)
        offsets = []
        for i in sense["long_syllables"]:
            # 표준 발음법 제6항: 긴소리는 단어의 첫음절에서만 나타난다.
            # 예) 눈[눈ː] 이지만 첫눈[천눈] 에서는 짧아진다.
            if i == 0 and not self.word_initial:
                continue
            offsets.append(self.start + i)
        return offsets


def find_candidates(text: str, lexicon: dict | None = None) -> list[Candidate]:
    lexicon = lexicon if lexicon is not None else load_lexicon()
    cands = []
    for tok in get_kiwi().tokenize(text):
        if tok.tag not in NOUN_TAGS or tok.form not in lexicon:
            continue
        word_start = text.rfind(" ", 0, tok.start) + 1   # 이 형태소가 속한 어절의 시작 위치
        cands.append(Candidate(
            form=tok.form, tag=tok.tag, start=tok.start,
            word_initial=(tok.start == word_start),
            senses=lexicon[tok.form],
        ))
    return cands


if __name__ == "__main__":
    for c in find_candidates("올해 첫눈은 눈이 부시게 하얗게 내렸다."):
        print(c.form, c.tag, c.start, "어절 첫머리" if c.word_initial else "어절 중간(장음 불가)")
