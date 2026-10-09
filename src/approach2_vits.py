"""방법 2: 사전학습된 한국어 TTS(VITS) 모델의 텐서에 직접 개입해서 장음을 만든다.

VITS는 글자(토큰)마다 '몇 프레임 동안 소리를 낼지'를 duration predictor가 예측한다.
    log_duration = duration_predictor(텍스트 인코딩)          # 모양: (1, 1, 토큰 수)
    duration     = ceil(exp(log_duration) * length_scale)     # 토큰별 프레임 수
여기에 PyTorch forward hook을 걸어 장음 모음 토큰의 log_duration에 log(w)를 더한다.
그러면 exp를 거쳐 그 토큰의 프레임 수만 w배가 되고, 나머지 토큰은 그대로다.
모델 전체 속도를 바꾸는 speaking_rate와 달리 '그 모음 하나'만 길어진다.

사용 모델: facebook/mms-tts-kor (Meta MMS 프로젝트, VITS 구조, 로마자 입력)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from morph import Candidate

VOWELS = set("aeiou")


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


class VowelLengthVITS:
    def __init__(self, model_id: str = "facebook/mms-tts-kor", model=None, tokenizer=None):
        from transformers import AutoTokenizer, VitsModel
        import uroman

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = (model or VitsModel.from_pretrained(model_id)).eval()
        self.uroman = uroman.Uroman()
        self.hop_length = int(np.prod(self.model.config.upsample_rates))   # 프레임 1개 = 256 샘플
        self.sr = self.model.config.sampling_rate
        self._log_weights: torch.Tensor | None = None
        self.last_log_duration: torch.Tensor | None = None
        # ★ 핵심: duration predictor의 출력 텐서를 가로채는 hook
        self.model.duration_predictor.register_forward_hook(self._duration_hook)

    def _duration_hook(self, module, inputs, output):
        if self._log_weights is not None:
            output = output + self._log_weights.to(output)
        self.last_log_duration = output.detach().clone()
        return output

    # ---------- 텍스트 → 토큰, 그리고 '어느 토큰이 어느 한글 음절에서 왔는지' ----------
    def romanize(self, text: str) -> tuple[str, list[int]]:
        """한 음절씩 로마자로 바꾸고, 로마자 글자마다 원래 한글 위치를 기록한다."""
        vocab = self.tokenizer.get_vocab()
        chars, owner = [], []
        for i, ch in enumerate(text):
            rom = self.uroman.romanize_string(ch) if _is_hangul(ch) else ch
            for r in rom.lower():
                if r in vocab:                       # 모델이 모르는 문장부호 등은 버린다
                    chars.append(r)
                    owner.append(i)
        return "".join(chars), owner

    def token_layout(self, text: str):
        roman, owner = self.romanize(text)
        ids = self.tokenizer(roman, return_tensors="pt").input_ids
        blank = bool(getattr(self.tokenizer, "add_blank", False))
        pos = [2 * k + 1 if blank else k for k in range(len(roman))]   # add_blank: [_, n, _, u, _, n, _]
        assert ids.shape[1] == (2 * len(roman) + 1 if blank else len(roman)), "토큰 위치 계산이 어긋났다"
        return roman, owner, ids, pos, blank

    def target_tokens(self, text: str, long_offsets: set[int]) -> list[int]:
        """장음 음절의 모음 토큰과 (blank가 있으면) 그 바로 뒤 blank 토큰."""
        roman, owner, ids, pos, blank = self.token_layout(text)
        targets = []
        for k, (r, o) in enumerate(zip(roman, owner)):
            if o in long_offsets and r in VOWELS:
                targets.append(pos[k])
                if blank:
                    targets.append(pos[k] + 1)
        return targets

    def syllable_token_span(self, text: str, offset: int) -> tuple[int, int]:
        roman, owner, ids, pos, blank = self.token_layout(text)
        ks = [k for k, o in enumerate(owner) if o == offset]
        return pos[ks[0]], pos[ks[-1]] + (2 if blank else 1)

    # ---------- 합성 ----------
    @torch.no_grad()
    def synthesize(self, text: str, cands: list[Candidate] | None = None, weight: float = 1.0,
                   seed: int = 0) -> dict:
        long_offsets = {o for c in (cands or []) for o in c.long_char_offsets()}
        roman, owner, ids, pos, blank = self.token_layout(text)
        targets = self.target_tokens(text, long_offsets) if weight != 1.0 else []

        log_w = torch.zeros(1, 1, ids.shape[1])
        log_w[0, 0, targets] = math.log(weight) if targets else 0.0
        self._log_weights = log_w

        torch.manual_seed(seed)            # 기본/수정 음성이 같은 난수를 쓰게 해서 공정하게 비교
        wav = self.model(input_ids=ids).waveform[0].cpu().numpy()
        self._log_weights = None

        length_scale = 1.0 / self.model.speaking_rate
        frames = torch.ceil(torch.exp(self.last_log_duration) * length_scale)[0, 0].long().numpy()
        return {"wav": wav, "sr": self.sr, "roman": roman, "frames": frames,
                "targets": targets, "long_offsets": sorted(long_offsets)}

    def span_seconds(self, result: dict, text: str, offset: int) -> tuple[float, float]:
        """음절 하나가 음성의 몇 초~몇 초에 해당하는지 (duration으로 계산한 정확한 정렬)."""
        a, b = self.syllable_token_span(text, offset)
        cum = np.concatenate([[0], np.cumsum(result["frames"])])
        to_sec = self.hop_length / self.sr
        return cum[a] * to_sec, cum[b] * to_sec
