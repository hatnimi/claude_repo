"""2단계: 문맥을 보고 동음이의어의 의미(→ 장단음)를 판별한다.

- baseline : 항상 첫 번째(짧은) 의미를 고른다. 장단 정보가 없는 일반 TTS의 동작과 같다.
- gemini   : 문장 전체를 Gemini에게 주고 후보 의미 중 하나를 고르게 한다.
             평가 문장 12개를 한 번의 요청으로 묻는다 (요청 횟수를 줄여 서버 과부하에 덜 걸리게).

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
아래 문장들에서 【문장번호-번호:단어】로 표시된 단어가 각각 어떤 의미로 쓰였는지 문맥을 보고 골라.

{body}

반드시 아래 JSON 형식으로만 답하고 다른 말은 쓰지 마. 표시된 단어 모두에 답해야 한다.
{{"results": [{{"sentence": 문장번호, "index": 번호, "sense_id": "후보 id 그대로", "reason": "한 문장 근거"}}]}}"""


def mark_sentence(text: str, cands: list[Candidate], prefix: str = "") -> str:
    out, prev = [], 0
    for i, c in enumerate(cands, 1):
        out += [text[prev:c.start], f"【{prefix}{i}:{c.form}】"]
        prev = c.end
    out.append(text[prev:])
    return "".join(out)


def build_batch_prompt(items: list[tuple[str, list[Candidate]]]) -> str:
    """여러 문장을 한 번에 묻는 질문을 만든다. items = [(문장, 판별할 후보들), ...]"""
    blocks = []
    for n, (text, cands) in enumerate(items, 1):
        lines = [f"[문장 {n}] {mark_sentence(text, cands, prefix=f'{n}-')}"]
        for i, c in enumerate(cands, 1):
            opts = " / ".join(f'"{s["id"]}"({s["gloss"]})' for s in c.senses)
            lines.append(f"  {n}-{i}. {c.form}: {opts}")
        blocks.append("\n".join(lines))
    return PROMPT_TEMPLATE.format(body="\n\n".join(blocks))


def build_prompt(text: str, cands: list[Candidate]) -> str:
    """문장 하나에 대한 질문 (노트북에서 질문 모양을 보여 줄 때 사용)."""
    return build_batch_prompt([(text, [c for c in cands if len(c.senses) > 1])])


def parse_batch(raw: str, items: list[tuple[str, list[Candidate]]]) -> dict[str, list[dict]]:
    """Gemini 답을 문장별 [{"sense_id", "reason"}, ...]로 바꾼다. 빠진 답이나 사전에 없는 답은 오류."""
    body = raw[raw.find("{"): raw.rfind("}") + 1]     # 앞뒤에 ```json 같은 말이 붙어 와도 JSON 부분만
    answers = {(int(r["sentence"]), int(r["index"])): r for r in json.loads(body)["results"]}
    out = {}
    for n, (text, cands) in enumerate(items, 1):
        out[text] = []
        for i, c in enumerate(cands, 1):
            r = answers.get((n, i))
            if r is None:
                raise ValueError(f"Gemini가 [문장 {n}]의 {i}번째 '{c.form}'에 답하지 않았다")
            valid = {s["id"] for s in c.senses}
            if r["sense_id"] not in valid:
                raise ValueError(f"Gemini가 사전에 없는 의미를 골랐다: {r['sense_id']} (후보 {valid})")
            out[text].append({"sense_id": r["sense_id"], "reason": r.get("reason", "")})
    return out


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
                config=types.GenerateContentConfig(temperature=0),   # JSON 모드는 끄고 글로 받은 뒤 JSON만 골라낸다
            )
            return resp.text
        except errors.APIError as e:
            if e.code not in retry_reason or wait is None:
                raise
            print(f"  ({e.code} {retry_reason[e.code]} → {wait}초 기다렸다 다시 시도)")
            time.sleep(wait)


def disambiguate_many(texts: list[str], method: str = "gemini", llm=call_gemini) -> list[list[Candidate]]:
    """여러 문장을 판별한다. Gemini에게는 아직 답을 받지 못한 문장들만 모아서 '한 번에' 묻는다."""
    results, pending = [], []
    for text in texts:
        cands, ambiguous = find_candidates(text), []
        for c in cands:
            if len(c.senses) == 1:                # 의미가 하나뿐이면 사전만으로 결정
                c.sense_id, c.reason = c.senses[0]["id"], "사전에 의미가 하나뿐"
            elif method == "baseline":
                c.sense_id, c.reason = c.senses[0]["id"], "항상 첫 번째 의미"
            else:
                ambiguous.append(c)
        results.append(cands)
        if ambiguous:
            pending.append((text, ambiguous))

    if pending:
        model, cache = current_model(), _load_cache()
        key = lambda text: f"[{model}] {text}"      # 모델이 바뀌면 다시 물어본다 (모델끼리 결과가 섞이지 않게)
        todo = [(t, a) for t, a in pending if key(t) not in cache]
        if todo:
            print(f"  Gemini({model})에게 {len(todo)}개 문장을 한 번에 묻는 중...")
            prompt = build_batch_prompt(todo)
            raw = llm(prompt)
            for text, answer in parse_batch(raw, todo).items():
                cache[key(text)] = answer
            cache.setdefault("_원본_응답", []).append({"model": model, "prompt": prompt, "response": raw})
            _save_cache(cache)
        for text, ambiguous in pending:
            for c, a in zip(ambiguous, cache[key(text)]):
                c.sense_id, c.reason = a["sense_id"], a["reason"]
    return results


def disambiguate(text: str, method: str = "gemini", llm=call_gemini) -> list[Candidate]:
    return disambiguate_many([text], method, llm)[0]


def is_long(c: Candidate, sense_id: str) -> bool:
    sense = next(s for s in c.senses if s["id"] == sense_id)
    return bool(sense["long_syllables"]) and c.word_initial


def evaluate(sentences: list[dict], method: str, llm=call_gemini) -> dict:
    """의미 정확도와 장단 정확도를 함께 잰다. (의미는 틀려도 장단은 맞을 수 있다: 배/舟 vs 배/梨)"""
    rows, sense_ok, length_ok, total = [], 0, 0, 0
    all_cands = disambiguate_many([item["text"] for item in sentences], method, llm)
    for item, cands in zip(sentences, all_cands):
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
