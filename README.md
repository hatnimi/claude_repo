# TTS 동음이의어 장단음 개선 탐구

> 같은 글자, 다른 길이. '눈(目)'과 '눈ː(雪)', '말(馬)'과 '말ː(言)'을 TTS가 구별해서 읽게 만들 수 있을까?

## 1. 문제
한국어에는 표기는 같지만 **모음의 길이(장단)** 로 뜻이 갈리는 동음이의어가 있다.
표준국어대사전은 `눈[눈ː]`(雪)처럼 장음을 따로 표시하지만, TTS는 글자만 입력받는다.
그래서 문맥을 보고 의미를 정하는 단계가 없고, 결국 두 단어를 같은 길이로 읽는다.

이 문제는 두 단계로 나뉜다.
1. **판별**: 이 문장의 '눈'이 目인가 雪인가? → 문맥 이해가 필요
2. **생성**: 장음이라고 판별했다면, 음성에서 그 모음을 실제로 길게 만들 수 있는가?

## 2. 구조
```
문장 ─► [Kiwi 형태소 분석] ─► 후보 단어 + 어절 첫머리 여부
                                │  (표준 발음법 제6항: 장음은 단어 첫음절에서만)
                                ▼
                        [Gemini 문맥 판별] ─► 의미 → 장음 음절 위치
                                │
            ┌───────────────────┴────────────────────┐
            ▼                                        ▼
  방법 1: 상용 TTS 입력 변경                 방법 2: VITS 텐서 개입
  SSML <prosody rate> / 철자 변형('누운')    duration_predictor 출력에 hook,
  (모델은 블랙박스)                          장음 모음 토큰의 log_duration += log(w)
            │                                        │
            └──────────────► [멜 스펙트로그램 + DTW 비교] ◄─┘
```

| 파일 | 역할 |
|---|---|
| `src/morph.py` | Kiwi로 사전에 있는 동음이의어 명사를 찾고, 어절 첫머리인지 표시 |
| `src/disambiguate.py` | baseline(항상 짧은 의미)과 Gemini 판별, 정확도 평가 |
| `src/approach1_commercial.py` | SSML 생성(Google Cloud TTS), 철자 변형(gTTS) |
| `src/approach2_vits.py` | `facebook/mms-tts-kor` VITS의 duration 텐서에 forward hook으로 가중치 부여 |
| `src/analysis.py` | 멜 스펙트로그램 비교 그림, DTW로 국소 늘어남 비율 추정 |
| `data/lexicon.json` | 동음이의어 장단음 사전 (표준국어대사전 기준) |
| `data/sentences.json` | 평가 문장 12개, 사람이 붙인 정답 의미 25개 |
| `run_experiment.py` | 전체 실험 실행 |
| `tts_vowel_length.ipynb` | **Colab에서 단계별로 실행하는 노트북 (여기서 시작)** |
| `tests/test_vits_hook.py` | 사전학습 가중치 없이 hook 동작만 검증하는 테스트 |
| `docs/탐구노트.md` | 탐구 과정 기록과 면접 대비 정리 |

## 3. 실행 방법 (Google Colab 추천)
1. `tts_vowel_length.ipynb`를 Colab에서 연다.
2. 왼쪽 🔑 **보안 비밀**에 `GEMINI_API_KEY`를 등록한다 ([Google AI Studio](https://aistudio.google.com/)에서 무료 발급).
   - 선택: `GOOGLE_TTS_API_KEY`(Google Cloud Text-to-Speech). 없으면 gTTS와 철자 변형 방식으로 진행한다.
3. 위에서부터 셀을 차례로 실행한다.

로컬에서 실행하려면 다음과 같이 한다.
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
python run_experiment.py            # 결과는 outputs/ 에 저장
python tests/test_vits_hook.py      # 인터넷 없이 hook 동작 확인
```
Gemini 모델 이름은 환경변수 `GEMINI_MODEL`로 바꿀 수 있다(기본값 `gemini-3.8-flash`).

## 4. 두 방법 비교
| | 방법 1: 상용 TTS + 입력 변경 | 방법 2: VITS 텐서 개입 |
|---|---|---|
| 개입 지점 | 모델 바깥(입력 텍스트/SSML) | 모델 안(duration 텐서) |
| 조절 단위 | 음절 단위 속도, 또는 글자를 추가 | 모음 토큰 하나의 프레임 수 |
| 정확도 | 엔진이 SSML을 어떻게 해석하는지에 따라 달라짐 | 정확히 w배 (ceil 반올림 오차 ±1프레임) |
| 부작용 | 단어가 잘려 운율이 끊기거나 '누운'처럼 두 음절로 들림 | 늘린 부분이 기계적으로 들릴 수 있음 |
| 음질 | 상용 엔진이라 좋음 | MMS 모델이라 상대적으로 낮음 |
| 정렬 정보 | 없음 → DTW로 추정 | duration으로 정확히 앎 |

## 5. 이 환경에서 확인한 것과 Colab에서 확인할 것
- ✅ Kiwi 후보 추출, 어절 첫머리 규칙(예: '첫눈'의 '눈'은 장음에서 제외)
- ✅ baseline 정확도: 의미 56% (14/25), 장단 60% (15/25)
- ✅ SSML과 철자 변형 생성 (예: `계속 말을 걸었다` → `계속 마알을 걸었다`)
- ✅ 무작위 초기화 VITS에서 hook 검증: 지정 토큰만 늘어나고 나머지는 그대로인지, 음성 길이 변화가 `늘어난 프레임 × 256 샘플`과 정확히 같은지
- ✅ DTW 검증: 합성 신호에서 한 구간만 2배로 늘렸을 때 그 구간의 평균 배율 2.07, 나머지 0.98
- ⏳ **Colab에서 직접 실행할 것**: Gemini 정확도, 실제 MMS 한국어 모델의 음성, 상용 TTS 음성, 멜 스펙트로그램
