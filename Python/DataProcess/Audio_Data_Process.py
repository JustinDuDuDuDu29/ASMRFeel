from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from multiprocessing import Pipe, Process, Queue
import numpy as np
from queue import Full, Empty
import scipy.signal as sig
import matplotlib.pyplot as plt
from Plot_Window.RollingPlot import RollingPlot
from collections import deque
import librosa
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
import queue
import time


def VoiceToVibrateData(ev:Event, q_Audio:Queue, parent_conn:Connection, vibrate_queue:Queue, filePath: str = "", sr: int = 16000, fmin: int = 70, fmax: int = 400, frame_length:int = 2048, hop_length:int = 160, window_ms:int = 200, saveToCSV: bool = False, saveCSVPath: str = "./freq_amp.csv"):
    rms_hist = deque(maxlen=1000)

    while not ev.is_set():
        y = q_Audio.get()
        if y.size < frame_length:
            continue

        # ---- Pitch (10 ms hop) ----
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length, center=True
        )
        f0_hz = np.nan_to_num(f0, nan=0.0)
        voiced = voiced_flag.astype(bool)

        # ---- RMS @ 10 ms hop (raw units) ----
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length, center=True
        )[0]

        # --- Update history BEFORE normalization ---
        rms_hist.extend(rms.tolist())

        # Bootstrap until we have enough history
        if len(rms_hist) < 100:  # ~1 s
            floor = np.percentile(rms, 10)
            p95g  = np.percentile(rms, 95)
        else:
            floor = np.percentile(rms_hist, 10)
            p95g  = np.percentile(rms_hist, 95)

        denom = max(p95g - floor, 1e-8)
        amp_10ms = np.clip((rms - floor) / denom, 0.0, 1.0)

        # ---- Pool both to 200 ms ----
        hop_ms = (1000 * hop_length) // sr  # ≈10 ms
        step = max(1, window_ms // hop_ms)

        n_frames = min(len(f0_hz), len(amp_10ms))
        n_full = (n_frames // step) * step
        if n_full == 0:
            continue

        f0_blocks = f0_hz[:n_full].reshape(-1, step)
        v_blocks  = voiced[:n_full].reshape(-1, step)
        amp_blocks = amp_10ms[:n_full].reshape(-1, step)

        def agg_f(f_block, v_block):
            return np.median(f_block[v_block]) if np.any(v_block) else 0.0

        f0_200ms  = np.array([agg_f(fb, vb) for fb, vb in zip(f0_blocks, v_blocks)])
        amp_200ms = np.median(amp_blocks, axis=1)

        # Light smoothing
        if len(f0_200ms)  >= 3: f0_200ms  = sig.medfilt(f0_200ms,  kernel_size=3)
        if len(amp_200ms) >= 3: amp_200ms = sig.medfilt(amp_200ms, kernel_size=3)

        # Gate by voicing: if no voiced frames, set amp = 0
        voiced_any = np.array([np.any(vb) for vb in v_blocks], dtype=bool)
        if len(amp_200ms) and not voiced_any[-1]:
            amp_200ms[-1] = 0.0

        # --- newest 200 ms values ---
        f = float(f0_200ms[-1])  if len(f0_200ms)  else 0.0
        a = float(amp_200ms[-1]) if len(amp_200ms) else 0.0
        print(f"f={f:.1f}Hz a={a:.3f}")
        # (optional) send audio to playback
        try:
            parent_conn.send(y)
        except (BrokenPipeError, EOFError):
            pass

        # ---- non-blocking latest-value-wins put ----
        try:
            vibrate_queue.put_nowait((f, a))
        except Full:
            try:
                vibrate_queue.get_nowait()   # drop stale
            except Empty:
                pass
            try:
                vibrate_queue.put_nowait((f, a))
            except Full:
                pass

        if saveToCSV:
            freq_amp_200ms = np.stack([f0_200ms, amp_200ms], axis=1)
            np.savetxt(saveCSVPath, freq_amp_200ms, delimiter=",", comments="", fmt="%.6f,%.6f")

def autocorr_pitch(frame, sr, fmin=50.0, fmax=800.0, RMS_GATE = 0):
    """Very lightweight pitch estimator using autocorrelation; returns 0 if unvoiced."""
    x = frame - np.mean(frame)
    rms = np.sqrt(np.mean(x*x))
    if rms < RMS_GATE:
        return 0.0

    n = len(x)
    size = 1
    while size < 2*n:
        size <<= 1
    X = np.fft.rfft(x, n=size)
    r = np.fft.irfft(X * np.conj(X))
    r = r[:n]

    lag_min = int(sr / fmax)
    lag_max = min(n-1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0

    seg = r[lag_min:lag_max+1]
    if len(seg) == 0:
        return 0.0
    peak_idx = int(np.argmax(seg)) + lag_min

    if r[0] <= 0:
        return 0.0
    clarity = r[peak_idx] / r[0]
    if clarity < 0.2:
        return 0.0

    f0 = sr / lag if (lag := peak_idx) > 0 else 0.0
    return float(f0 if (f0 >= fmin and f0 <= fmax) else 0.0)

def _hz_to_bin(hz, nfft, sr):
    return int(np.clip(np.floor(hz/(sr/nfft)), 0, nfft//2))

def tone_features(frame, sr, nfft=512):
    w = np.hanning(len(frame))
    spec = np.fft.rfft(frame*w, n=nfft)
    mag  = np.abs(spec) + 1e-12
    pow_ = mag*mag
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)

    total = float(np.sum(pow_) + 1e-12)

    centroid_hz = float(np.sum(freqs * mag) / np.sum(mag))
    centroid_norm = float(np.clip(centroid_hz / (sr/2), 0.0, 1.0))
    sfm = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag)))
    hf_bin = _hz_to_bin(3000, nfft, sr)
    hf_ratio = float(np.sum(pow_[hf_bin:]) / total)

    lf_max = _hz_to_bin(2000, nfft, sr)
    mh_max = _hz_to_bin(8000, nfft, sr)
    lf   = float(np.sum(pow_[:lf_max]) + 1e-12)
    midh = float(np.sum(pow_[lf_max:mh_max]) + 1e-12)
    harmonicity = float(lf / (lf + midh))  # 越大=越有聲/諧波清楚

    return centroid_hz, centroid_norm, sfm, hf_ratio, harmonicity

def VoiceToThermalData(ev:Event, q_Audio:Queue, parent_conn:Connection, thermal_queue:Queue, filePath: str = "", sr: int = 16000, frame_length:int = 2048, hop_length:int = 160, block_length:int = 200, rms_gate: float = 2e-5, smooth_strength: float = 0.15, saveToCSV: bool = False, saveCSVPath: str = "./tone.csv"):
    
    hop_s = hop_length / float(sr)
    if smooth_strength <= 0:
        ema_alpha = 1.0
    else:
        ema_alpha = 1.0 - np.exp(-hop_s / smooth_strength)

    tone_mix = 0.0 ; tone_smooth = 0.0
    
    # Plot window
    # plt.ion()
    # fig, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    # tone_plot  = RollingPlot(ax3, "Tone (Breathy-aware, 0..1)", "Tone (0..1)", 0.0, 1.0, 3.0, hop_s,
    #                           with_second=True, second_label="Tone (EMA)")
    # fig.tight_layout()

    while True:
        try:
            start_time = time.monotonic()
            y = q_Audio.get(timeout=0.01)
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < rms_gate:
                tone_mix = 0.0
            else:
                # rms_norm = float(rms / (rms + 0.2))
                # f0 = autocorr_pitch(y, sr, rms_gate)

                _, cn, sfm, hfr, harm = tone_features(y, sr)
                tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
                tone_mix = float(np.clip(tone_mix, 0.0, 1.0))

            """平滑處理(EMA)"""

            tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
            print(time.monotonic() - start_time)
            # tone_plot.update(tone_mix, tone_smooth)

            # ---- non-blocking latest-value-wins put ----
            # try:
            #     thermal_queue.put_nowait(tone_smooth)
            # except Full:
            #     try:
            #         thermal_queue.get_nowait()   # drop stale
            #     except Empty:
            #         pass
            #     try:
            #         thermal_queue.put_nowait(tone_smooth)
            #     except Full:
            #         pass

            # if saveToCSV:
            #     tone_stack = np.stack([tone_smooth], axis=1)
            #     np.savetxt(saveCSVPath, tone_stack, delimiter=",", comments="", fmt="%.6f,%.6f")
        except queue.Empty:
            pass
        # plt.pause(0.01)

