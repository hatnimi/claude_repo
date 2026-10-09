"""방법 1: 판별한 장단음을 상용 TTS의 '입력'에 반영한다. (모델 내부는 건드리지 않음)

상용 TTS는 모델 내부를 열어 볼 수 없으므로, 우리가 바꿀 수 있는 것은 입력뿐이다.
  (a) SSML  : 장음 음절만 <prosody rate>로 느리게 읽게 한다.  → Google Cloud TTS
  (b) 철자 변형 : '눈' → '누운'처럼 모음을 한 번 더 적는다.      → SSML을 못 쓰는 TTS(gTTS 등)
"""
from __future__ import annotations

import base64
import io
import os
from xml.sax.saxutils import escape

import numpy as np
import requests

from morph import Candidate

# ---------- 한글 음절 분해/조합 ----------
_BASE, _N_JUNG, _N_JONG = 0xAC00, 21, 28
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
# 이중모음을 늘이면 뒤쪽 단모음이 길어진다 (ㅘ = ㅗ+ㅏ → ㅏ가 늘어남)
_TAIL_VOWEL = {"ㅑ": "ㅏ", "ㅒ": "ㅐ", "ㅕ": "ㅓ", "ㅖ": "ㅔ", "ㅘ": "ㅏ", "ㅙ": "ㅐ",
               "ㅛ": "ㅗ", "ㅝ": "ㅓ", "ㅞ": "ㅔ", "ㅠ": "ㅜ", "ㅢ": "ㅣ", "ㅟ": "ㅣ"}
_IEUNG = 11  # 초성 ㅇ의 번호


def _split(ch: str) -> tuple[int, int, int]:
    code = ord(ch) - _BASE
    return code // (_N_JUNG * _N_JONG), (code // _N_JONG) % _N_JUNG, code % _N_JONG


def _join(cho: int, jung: int, jong: int) -> str:
    return chr(_BASE + (cho * _N_JUNG + jung) * _N_JONG + jong)


def lengthen_syllable(ch: str) -> str:
    """'눈' → '누운', '사' → '사아', '성' → '서엉'"""
    cho, jung, jong = _split(ch)
    tail = _JUNG.index(_TAIL_VOWEL.get(_JUNG[jung], _JUNG[jung]))
    return _join(cho, jung, 0) + _join(_IEUNG, tail, jong)


def long_offsets(cands: list[Candidate]) -> set[int]:
    return {o for c in cands for o in c.long_char_offsets()}


def to_respelled(text: str, cands: list[Candidate]) -> str:
    longs = long_offsets(cands)
    return "".join(lengthen_syllable(ch) if i in longs else ch for i, ch in enumerate(text))


def to_ssml(text: str, cands: list[Candidate], rate: str = "55%") -> str:
    longs = long_offsets(cands)
    body = "".join(f'<prosody rate="{rate}">{escape(ch)}</prosody>' if i in longs else escape(ch)
                   for i, ch in enumerate(text))
    return f"<speak>{body}</speak>"


# ---------- 상용 TTS 호출 ----------
def synth_google_cloud(ssml: str, voice: str = "ko-KR-Standard-A", sr: int = 22050) -> tuple[np.ndarray, int]:
    """Google Cloud Text-to-Speech (REST, API 키 사용). 환경변수 GOOGLE_TTS_API_KEY 필요."""
    import soundfile as sf

    resp = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={"X-Goog-Api-Key": os.environ["GOOGLE_TTS_API_KEY"]},   # 주소에 넣으면 오류 메시지에 키가 찍힌다
        json={"input": {"ssml": ssml},
              "voice": {"languageCode": "ko-KR", "name": voice},
              "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": sr}},
        timeout=30,
    )
    if not resp.ok:
        reason = resp.json().get("error", {}).get("message", resp.text[:200]) if resp.content else ""
        raise RuntimeError(f"Google Cloud TTS {resp.status_code}: {reason}")
    wav, sr = sf.read(io.BytesIO(base64.b64decode(resp.json()["audioContent"])), dtype="float32")
    return wav, sr


def synth_gtts(text: str) -> tuple[np.ndarray, int]:
    """gTTS(구글 번역 음성). 무료지만 SSML을 지원하지 않아 철자 변형만 쓸 수 있다."""
    import librosa
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text, lang="ko").write_to_fp(buf)
    buf.seek(0)
    wav, sr = librosa.load(buf, sr=None)
    return wav, sr


def synthesize_pair(text: str, cands: list[Candidate]) -> dict:
    """원래 문장과 장단음을 반영한 문장을 같은 엔진으로 합성해서 돌려준다."""
    if os.environ.get("GOOGLE_TTS_API_KEY"):
        try:
            base = synth_google_cloud(f"<speak>{escape(text)}</speak>")
            mod_input = to_ssml(text, cands)
            mod = synth_google_cloud(mod_input)
            return {"engine": "Google Cloud TTS + SSML", "input": mod_input, "base": base, "mod": mod}
        except RuntimeError as e:
            # 403 = 이 키의 프로젝트에서 Cloud Text-to-Speech API가 켜져 있지 않거나, 키가 다른 API 전용으로 제한됨
            print(f"  ⚠️ {e}\n  → Google Cloud TTS를 쓸 수 없어 gTTS + 철자 변형으로 대신합니다.")
            del os.environ["GOOGLE_TTS_API_KEY"]   # 다음 문장부터는 바로 gTTS 사용
    base = synth_gtts(text)
    mod_input = to_respelled(text, cands)
    mod = synth_gtts(mod_input)
    return {"engine": "gTTS + 철자 변형", "input": mod_input, "base": base, "mod": mod}
