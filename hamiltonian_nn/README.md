# 해밀턴 경로로 연결한 은닉층 탐구

> 밀집층의 $n^2$개 연결을 **해밀턴 경로**($2n-1$개 간선)로 바꾸면 어떻게 될까?
> 성능, 수학적 성질(야코비안·라플라시안), 경로 학습(이중확률행렬·싱크혼), 실제 GPU 실행 시간까지 직접 확인한다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hatnimi/claude_repo/blob/claude/relaxed-knuth-q3zflq/hamiltonian_nn/hamiltonian_nn.ipynb)

## 핵심 아이디어
한 층의 연결을 입력 $n$개와 출력 $n$개로 이루어진 이분 그래프로 본다. 이 그래프의 정점 $2n$개를 모두 한 번씩 지나는 해밀턴 경로

$$x_{\pi(1)} \to y_1 \to x_{\pi(2)} \to y_2 \to \cdots \to x_{\pi(n)} \to y_n$$

만 남기면, 출력 하나는 입력 두 개와만 연결된다.

$$y_k = a_k\, x_{\pi(k)} + b_k\, x_{\pi(k+1)} + c_k$$

파라미터는 $n^2 \to 3n$, 연산은 $O(n^2) \to O(n)$으로 줄어든다.

## 파일
| 파일 | 역할 |
|---|---|
| `hamiltonian_nn.ipynb` | **Colab에서 위에서부터 실행하는 노트북 (여기서 시작)** |
| `탐구노트.md` | 탐구 과정과 결과 기록, 면접 대비 정리 (실행 결과를 직접 채워 넣는 곳) |
| `save_to_github.py` | Colab 실행 결과(노트북 + `outputs/`)를 GitHub에 저장 |
| `outputs/` | 실행하면 생기는 그림(`*.png`), `summary.md`, `results.json` |

## 노트북 구성
| 절 | 내용 | 수학 도구 |
|---|---|---|
| 1 | 해밀턴 경로로 층 만들기, 그래프로 검증 | 이분 그래프, 순열행렬, 이중대각 행렬 |
| 2 | 역전파 수식 직접 유도 → `autograd.Function`으로 구현 → gradcheck | 연쇄법칙, 야코비안 |
| 3 | 같은 간선 수의 무작위 연결·매칭과 비교 | 라플라시안 고윳값, 대수적 연결도 |
| 4 | 실험 1: FashionMNIST 정확도·파라미터 비교 | |
| 5 | 순열 → 이중확률행렬, 최적 운송, 싱크혼 역전파 유도 | 버코프 정리, 싱크혼 정리, 엔트로피 정규화 |
| 6 | 실험 2: 싱크혼으로 경로(순열) 학습 → 헝가리안으로 경화 | 선형 할당 문제 |
| 7 | 실험 3: 이론 연산량 vs 실제 GPU 시간, 커널 수, 배치 크기 | 산술 강도, 메모리 대역폭, GPU 병렬성 |

## 실행
1. 위 **Open in Colab** 배지를 누르거나, Colab에서 `파일 → 노트 열기 → GitHub`에 이 저장소를 입력해 `hamiltonian_nn/hamiltonian_nn.ipynb`를 연다.
   (저장소가 비공개라면 노트북 파일을 내려받아 Colab에 업로드한다.)
2. `런타임 → 런타임 유형 변경 → T4 GPU`
3. 위에서부터 차례로 실행 (약 20~30분)
4. 마지막 셀의 요약표를 `탐구노트.md`에 옮겨 적고 해석을 쓴다.

로컬에서 짧게 동작만 점검하려면 `HAM_QUICK=1 jupyter nbconvert --to notebook --execute hamiltonian_nn.ipynb`
(데이터 일부, 1 epoch만 쓰므로 정확도 숫자는 의미가 없다).
