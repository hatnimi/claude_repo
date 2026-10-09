"""전체 실험을 한 번에 돌린다.

  python run_experiment.py --step disambig   # 1·2단계: 형태소 분석 + Gemini 판별 정확도
  python run_experiment.py --step commercial # 방법 1: 상용 TTS
  python run_experiment.py --step vits       # 방법 2: VITS duration 개입
  python run_experiment.py                   # 전부

결과는 outputs/ 아래에 저장된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import plot_compare, trim  # noqa: E402
from disambiguate import disambiguate, evaluate  # noqa: E402

OUT = ROOT / "outputs"
SENTENCES = json.loads((ROOT / "data" / "sentences.json").read_text(encoding="utf-8"))
# 시연용 문장: 같은 단어가 짧게/길게 한 번씩 나오는 문장
DEMO = ["눈에 눈이 들어가서 눈물이 났다.",
        "그는 말을 타고 달리면서 계속 말을 걸었다.",
        "밤에 먹는 밤은 유난히 달다.",
        "사과를 깎다가 친구에게 사과했다."]


def md_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def error_hint(e: Exception) -> str:
    msg = str(e)
    if "404" in msg or "NOT_FOUND" in msg:
        return "모델 이름이 더 이상 제공되지 않습니다. 오류 메시지가 권하는 모델 이름을 os.environ['GEMINI_MODEL']에 넣고 다시 실행하세요."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "무료 사용량을 초과했습니다. 1~2분 뒤 다시 실행하세요. 이미 받은 답은 저장돼 있어 이어서 진행됩니다."
    if "API key" in msg or "PERMISSION_DENIED" in msg or "GEMINI_API_KEY" in msg:
        return "API 키 문제입니다. 🔑 보안 비밀 이름이 GEMINI_API_KEY인지, 값이 정확한지 확인하세요."
    return "위 오류 메시지를 확인하세요."


def step_disambig():
    report = ["# 동음이의어 장단음 판별 결과\n"]
    path = OUT / "disambiguation.md"
    for method in ["baseline", "gemini"]:
        try:
            r = evaluate(SENTENCES, method)
        except Exception as e:   # 실패해도 baseline 결과와 실패 원인을 파일에 남긴다
            err = f"{type(e).__name__}: {e}"
            report += [f"## {method}", f"**❌ 실행 실패**\n\n```\n{err}\n```\n", f"👉 {error_hint(e)}", ""]
            path.write_text("\n".join(report), encoding="utf-8")
            print(f"[{method}] ❌ 실패: {err}\n👉 {error_hint(e)}")
            sys.exit(1)
        report += [f"## {method}", f"- 의미 정확도: {r['sense_acc']:.1%}",
                   f"- 장단 정확도: {r['length_acc']:.1%}\n", md_table(r["rows"]), ""]
        path.write_text("\n".join(report), encoding="utf-8")
        print(f"[{method}] 의미 {r['sense_acc']:.1%} / 장단 {r['length_acc']:.1%}")


def step_commercial():
    from approach1_commercial import synthesize_pair

    d = OUT / "commercial"
    d.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(DEMO):
        cands = disambiguate(text)
        r = synthesize_pair(text, cands)
        base, mod = (trim(r["base"][0]), r["base"][1]), (trim(r["mod"][0]), r["mod"][1])
        sf.write(d / f"{i}_base.wav", *base)
        sf.write(d / f"{i}_mod.wav", *mod)
        plot_compare(base, mod, f"[{r['engine']}] {text}\n입력: {r['input']}", str(d / f"{i}_mel.png"))
        print(f"[상용 TTS] {text}  →  {r['input']}")


def step_vits(weight: float):
    from approach2_vits import VowelLengthVITS

    d = OUT / "vits"
    d.mkdir(parents=True, exist_ok=True)
    tts = VowelLengthVITS()
    rows = []
    for i, text in enumerate(DEMO):
        cands = disambiguate(text)
        base = tts.synthesize(text, cands, weight=1.0)
        mod = tts.synthesize(text, cands, weight=weight)
        offsets = mod["long_offsets"]
        b_spans = [tts.span_seconds(base, text, o) for o in offsets]
        m_spans = [tts.span_seconds(mod, text, o) for o in offsets]
        sf.write(d / f"{i}_base.wav", base["wav"], base["sr"])
        sf.write(d / f"{i}_mod.wav", mod["wav"], mod["sr"])
        plot_compare((base["wav"], base["sr"]), (mod["wav"], mod["sr"]),
                     f"[VITS duration ×{weight}] {text}  (하늘색 = 장음 음절)", str(d / f"{i}_mel.png"),
                     base_spans=b_spans, mod_spans=m_spans)
        for o, (bs, be), (ms, me) in zip(offsets, b_spans, m_spans):
            rows.append({"문장": text, "음절": text[o], "기본(ms)": round((be - bs) * 1000),
                         "개입 후(ms)": round((me - ms) * 1000), "배율": round((me - ms) / (be - bs), 2)})
        print(f"[VITS] {text}  로마자: {mod['roman']}  개입 토큰: {mod['targets']}")

    # 가중치를 바꿔 가며 '얼마나 늘려야 자연스러운 장음인가' 탐색
    sweep_text, sweep = DEMO[0], []
    cands = disambiguate(sweep_text)
    o = cands[1].long_char_offsets()[0]
    for w in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        r = tts.synthesize(sweep_text, cands, weight=w)
        s, e = tts.span_seconds(r, sweep_text, o)
        sf.write(d / f"sweep_w{w}.wav", r["wav"], r["sr"])
        sweep.append({"가중치": w, "장음 음절 길이(ms)": round((e - s) * 1000),
                      "전체 길이(초)": round(len(r["wav"]) / r["sr"], 2)})

    (OUT / "vits_result.md").write_text(
        "# VITS duration 개입 결과\n\n" + md_table(rows) + "\n\n## 가중치 탐색\n\n" + md_table(sweep) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["disambig", "commercial", "vits", "all"], default="all")
    ap.add_argument("--weight", type=float, default=1.8, help="장음 모음 duration 배율")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.step in ("disambig", "all"):
        step_disambig()
    if args.step in ("commercial", "all"):
        step_commercial()
    if args.step in ("vits", "all"):
        step_vits(args.weight)
