"""사전학습 가중치 없이(무작위 초기화 VITS) hook 동작을 검증하는 테스트.

실제 음질은 의미가 없고, '지정한 토큰의 프레임 수만 w배가 되는가'만 확인한다.
실행: python tests/test_vits_hook.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transformers import VitsConfig, VitsModel, VitsTokenizer  # noqa: E402

from approach2_vits import VowelLengthVITS  # noqa: E402
from morph import find_candidates  # noqa: E402


def make_dummy():
    chars = ["_"] + list(" '-abcdefghijklmnopqrstuvwxyz")      # MMS처럼 0번 = blank
    vocab = {c: i for i, c in enumerate(chars)}
    path = Path(tempfile.mkdtemp()) / "vocab.json"
    path.write_text(json.dumps(vocab))
    tok = VitsTokenizer(str(path), add_blank=True, normalize=True, phonemize=False, is_uroman=True)
    torch.manual_seed(0)
    model = VitsModel(VitsConfig(vocab_size=len(chars), use_stochastic_duration_prediction=True))
    return VowelLengthVITS(model=model, tokenizer=tok)


def main():
    tts = make_dummy()
    text = "눈에 눈이 들어가서 눈물이 났다."
    cands = find_candidates(text)
    cands[0].sense_id, cands[1].sense_id = "눈/目", "눈/雪"

    roman, owner, ids, pos, blank = tts.token_layout(text)
    print("로마자:", roman)
    targets = tts.target_tokens(text, {o for c in cands for o in c.long_char_offsets()})
    labels = tts.tokenizer.convert_ids_to_tokens(ids[0])
    print("개입 토큰:", [(k, labels[k]) for k in targets])
    assert [labels[k] for k in targets] == ["u", "_"], "두 번째 '눈'의 모음 u와 뒤 blank여야 한다"
    assert owner[pos.index(targets[0])] == 3, "두 번째 '눈'(문장 위치 3)에서 온 토큰이어야 한다"

    w = 2.0
    base = tts.synthesize(text, cands, weight=1.0)
    mod = tts.synthesize(text, cands, weight=w)

    for k in targets:
        print(f"토큰 {k} {labels[k]!r}: {base['frames'][k]} → {mod['frames'][k]} 프레임")
    others = [k for k in range(len(labels)) if k not in targets]
    assert (base["frames"][others] == mod["frames"][others]).all(), "개입하지 않은 토큰은 그대로여야 한다"
    assert mod["frames"][targets].sum() > base["frames"][targets].sum(), "개입한 토큰은 길어져야 한다"
    added = int(mod["frames"].sum() - base["frames"].sum())
    assert len(mod["wav"]) - len(base["wav"]) == added * tts.hop_length, "늘어난 프레임만큼 음성이 길어져야 한다"
    print(f"전체: {base['frames'].sum()} → {mod['frames'].sum()} 프레임, 음성 {len(base['wav'])} → {len(mod['wav'])} 샘플")

    s0, e0 = tts.span_seconds(base, text, 3)
    s1, e1 = tts.span_seconds(mod, text, 3)
    print(f"두 번째 '눈' 구간: {s0:.3f}~{e0:.3f}초 → {s1:.3f}~{e1:.3f}초")
    print("OK")


if __name__ == "__main__":
    main()
