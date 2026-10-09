"""2단계: 문맥을 보고 동음이의어의 의미(→ 장단음)를 판별한다.

- baseline : 항상 첫 번째(짧은) 의미를 고른다. 장단 정보가 없는 일반 TTS의 동작과 같다.
- gemini   : 문장 전체를 Gemini에게 주고 후보 의미 중 하나를 고르게 한다.

Gemini 응답은 outputs/gemini_cache.json에 저장한다.
다시 실행해도 API를 또 부르지 않고, 어떤 근거로 골랐는지 나중에 확인할 수 있다.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from morph import Candidate, find_candidates

CACHE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "gemini_cache.json"
DEFAULT_MODEL = "gemini-3.8-flash"


def current_model() -> str:
    """노트북에서 os.environ["GEMINI_MODEL"]을 바꾸면 바로 반영되도록 호출할 때마다 읽는다."""
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL

PROMPT_TEMPLATE = """너는 한국어 표준 발음 전문가야.
아래 문장에서 【번호:단어】로 표시된 단어가 각각 어떤 의미로 쓰였는지 문맥을 보고 골라.

문장: {marked}

후보:
{options}

반드시 아래 JSON 형식으로만 답해.
{{"results": [{{"index": 번호, "sense_id": "후보 id 그대로", "reason": "한 문장 근거"}}]}}"""


def mark_sentence(text: str, cands: list[Candidate]) -> str:
    out, prev = [], 0
    for i, c in enumerate(cands, 1):
        out += [text[prev:c.start], f"【{i}:{c.form}】"]
        prev = c.end
    out.append(text[prev:])
    return "".join(out)


def build_prompt(text: str, cands: list[Candidate]) -> str:
    lines = []
    for i, c in enumerate(cands, 1):
        opts = " / ".join(f'"{s["id"]}"({s["gloss"]})' for s in c.senses)
        lines.append(f"{i}. {c.form}: {opts}")
    return PROMPT_TEMPLATE.format(marked=mark_sentence(text, cands), options="\n".join(lines))


def parse_response(raw: str, cands: list[Candidate]) -> None:
    raw = raw.strip()
    if raw.startswith("```"):                       # ```json ... ``` 으로 감싸서 오는 경우
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    for r in json.loads(raw)["results"]:
        c = cands[int(r["index"]) - 1]
        valid = {s["id"] for s in c.senses}
        if r["sense_id"] not in valid:
            raise ValueError(f"Gemini가 사전에 없는 의미를 골랐다: {r['sense_id']} (후보 {valid})")
        c.sense_id, c.reason = r["sense_id"], r.get("reason", "")


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def call_gemini(prompt: str, model: str | None = None) -> str:
    from google import genai
    from google.genai import errors, types

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다. Google AI Studio에서 키를 발급받아 등록하세요.")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = model or current_model()
    # 429: 무료 등급 분당 호출 한도 초과 / 500·503·504: 서버 과부하 → 둘 다 잠시 뒤 다시 시도하면 된다
    retry_reason = {429: "호출 한도 초과", 500: "서버 오류", 503: "서버 과부하", 504: "서버 응답 지연"}
    for wait in [15, 30, 60, 120, None]:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
            return resp.text
        except errors.APIError as e:
            if e.code not in retry_reason or wait is None:
                raise
            print(f"  ({e.code} {retry_reason[e.code]} → {wait}초 기다렸다 다시 시도)")
            time.sleep(wait)


def disambiguate(text: str, method: str = "gemini", llm=call_gemini) -> list[Candidate]:
    cands = find_candidates(text)
    ambiguous = []
    for c in cands:
        if len(c.senses) == 1:                    # 의미가 하나뿐이면 사전만으로 결정
            c.sense_id, c.reason = c.senses[0]["id"], "사전에 의미가 하나뿐"
        elif method == "baseline":
            c.sense_id, c.reason = c.senses[0]["id"], "항상 첫 번째 의미"
        else:
            ambiguous.append(c)

    if ambiguous:
        prompt = build_prompt(text, ambiguous)
        cache = _load_cache()
        key = f"[{current_model()}]\n{prompt}"       # 모델이 바뀌면 다시 물어본다 (모델끼리 결과가 섞이지 않게)
        if key not in cache:
            cache[key] = llm(prompt)
            _save_cache(cache)
        parse_response(cache[key], ambiguous)
    return cands


def is_long(c: Candidate, sense_id: str) -> bool:
    sense = next(s for s in c.senses if s["id"] == sense_id)
    return bool(sense["long_syllables"]) and c.word_initial


def evaluate(sentences: list[dict], method: str, llm=call_gemini) -> dict:
    """의미 정확도와 장단 정확도를 함께 잰다. (의미는 틀려도 장단은 맞을 수 있다: 배/舟 vs 배/梨)"""
    rows, sense_ok, length_ok, total = [], 0, 0, 0
    for item in sentences:
        cands = disambiguate(item["text"], method, llm)
        assert len(cands) == len(item["gold"]), f"후보 수와 정답 수가 다르다: {item['text']}"
        for c, gold in zip(cands, item["gold"]):
            s_ok = c.sense_id == gold
            l_ok = is_long(c, c.sense_id) == is_long(c, gold)
            sense_ok += s_ok
            length_ok += l_ok
            total += 1
            rows.append({"문장": item["text"], "단어": c.form, "정답": gold, "예측": c.sense_id,
                         "의미": "O" if s_ok else "X", "장단": "O" if l_ok else "X", "근거": c.reason})
    return {"method": method, "sense_acc": sense_ok / total, "length_acc": length_ok / total, "rows": rows}
