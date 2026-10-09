"""3단계: 멜 스펙트로그램으로 '정말 길어졌는지' 눈으로 확인하고 숫자로 잰다.

- VITS(방법 2): duration을 알고 있으므로 음절이 몇 초~몇 초인지 정확히 표시할 수 있다.
- 상용 TTS(방법 1): 내부 정렬 정보가 없다. 그래서 DTW(동적 시간 정렬)로
  '원래 음성의 각 순간이 수정 음성에서 몇 배로 늘어났는지'를 추정한다.
"""
from __future__ import annotations

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
for _font in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]:
    if any(_font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _font
        break

SR, N_FFT, HOP, N_MELS = 16000, 1024, 256, 80


def mel_db(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def trim(wav: np.ndarray, top_db: float = 35) -> np.ndarray:
    """앞뒤 무음 제거. 상용 TTS는 앞뒤 무음 길이가 매번 달라 비교를 방해한다."""
    return librosa.effects.trim(wav, top_db=top_db)[0]


def dtw_stretch(base: np.ndarray, mod: np.ndarray, smooth: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """원래 음성의 각 프레임이 수정 음성에서 몇 프레임에 대응하는지 = 국소 늘어남 비율."""
    a, b = mel_db(base, SR), mel_db(mod, SR)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
    _, path = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    counts = np.bincount(path[:, 0], minlength=a.shape[1]).astype(float)
    ratio = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    times = np.arange(a.shape[1]) * HOP / SR
    return times, ratio


def plot_compare(base: tuple[np.ndarray, int], mod: tuple[np.ndarray, int], title: str, path: str,
                 base_spans=(), mod_spans=(), labels=("기본", "장음 반영"), show_dtw: bool = True):
    """위: 기본 음성 / 가운데: 장음 반영 음성 / 아래: DTW 늘어남 비율. 시간축을 같은 눈금으로 맞춘다."""
    (wb, srb), (wm, srm) = base, mod
    wb = librosa.resample(wb, orig_sr=srb, target_sr=SR) if srb != SR else wb
    wm = librosa.resample(wm, orig_sr=srm, target_sr=SR) if srm != SR else wm
    tmax = max(len(wb), len(wm)) / SR

    rows = 3 if show_dtw else 2
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.6 * rows), constrained_layout=True)
    for ax, w, spans, lab in [(axes[0], wb, base_spans, labels[0]), (axes[1], wm, mod_spans, labels[1])]:
        img = librosa.display.specshow(mel_db(w, SR), sr=SR, hop_length=HOP, x_axis="time",
                                       y_axis="mel", ax=ax, cmap="magma")
        for s, e in spans:
            ax.axvspan(s, e, color="cyan", alpha=0.25)
            ax.axvline(s, color="cyan", lw=1)
            ax.axvline(e, color="cyan", lw=1)
        ax.set_xlim(0, tmax)
        ax.set_title(f"{lab}  (길이 {len(w) / SR:.2f}초)")
    fig.colorbar(img, ax=axes[:2], format="%+2.0f dB")

    if show_dtw:
        t, r = dtw_stretch(wb, wm)
        axes[2].plot(t, r, color="tab:orange")
        axes[2].axhline(1.0, color="gray", ls="--", lw=1)
        for s, e in base_spans:
            axes[2].axvspan(s, e, color="cyan", alpha=0.25)
        axes[2].set_xlim(0, tmax)
        axes[2].set_ylabel("늘어난 배율")
        axes[2].set_xlabel("기본 음성 기준 시간(초)")
        axes[2].set_title("DTW 정렬: 기본 음성의 각 구간이 수정 음성에서 몇 배로 늘어났나")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
