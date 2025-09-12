from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import queue as pyqueue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Plot_Window.RollingPlot import RollingPlot

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config

def dsp_vib(stop_evt: Event, q_audio_vib: Queue, q_vib: Queue):
    """Extract RMS from latest audio and push to q_vib."""
    while not stop_evt.is_set():
        try:
            arr = q_audio_vib.get(timeout=0.1)
            rms = float(np.sqrt(np.mean(arr**2)))
            vib_level = rms * Config.VIB_OUT_SCALE
            try:
                # print(f"vib_level: {vib_level}")
                q_vib.put_nowait(vib_level)
            except pyqueue.Full:
                print("q_vib full!!!")
        except pyqueue.Empty:
            print("audio vib empty, waiting...")
            continue

def dsp_therm(stop_evt: Event, q_audio_therm: Queue, q_therm: Queue, rms_gate: float = 2e-5, smooth_strength: float = 0.15):
    """Extract tone features from latest audio and push to q_therm."""
    def _hz_to_bin(hz: float, nfft: int, sr: int) -> int:
        """Clamp a frequency (Hz) to a valid FFT bin index [0, nfft//2]."""
        return int(np.clip(np.floor(hz / (sr / float(nfft))), 0, nfft // 2))

    def tone_features(frame: np.ndarray, sr: int, nfft: int) -> tuple[float, float, float, float, float]:
        """
        Compute tone-related features on a mono frame:
        - centroid_hz, centroid_norm in [0,1]
        - spectral flatness measure (SFM) in [0,1]
        - high-frequency ratio (>=3 kHz)
        - harmonicity proxy: low/(low+mid-high) with split at 2 kHz and 8 kHz
        """
        # Hann window
        w = np.hanning(len(frame))
        # rFFT with chosen nfft (>= len(frame) preferred)
        spec = np.fft.rfft(frame * w, n=nfft)
        mag = np.abs(spec) + 1e-12
        pow_ = mag * mag
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

        total_pow = float(np.sum(pow_) + 1e-12)

        # Spectral centroid
        centroid_hz = float(np.sum(freqs * mag) / np.sum(mag))
        centroid_norm = float(np.clip(centroid_hz / (sr / 2.0), 0.0, 1.0))

        # Spectral flatness (0=peaky/tonal, 1=flat/noisy)
        sfm = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag)))

        # High-frequency ratio (>= 3 kHz)
        hf_bin = _hz_to_bin(3000, nfft, sr)
        hf_ratio = float(np.sum(pow_[hf_bin:]) / total_pow)

        # Harmonicity proxy: more low vs mid-high energy => "more voiced"
        lf_max = _hz_to_bin(2000, nfft, sr)   # < 2 kHz
        mh_max = _hz_to_bin(8000, nfft, sr)   # < 8 kHz (cap if sr low)
        lf = float(np.sum(pow_[:lf_max]) + 1e-12)
        midh = float(np.sum(pow_[lf_max:mhh_max if (mhh_max := mh_max) else lf_max]) + 1e-12)
        harmonicity = float(lf / (lf + midh))  # larger => more harmonic/voiced

        return centroid_hz, centroid_norm, sfm, hf_ratio, harmonicity
    

    hop_s = Config.AUDIO_CHUNK_MS / 1000.0
    # EMA coefficient derived from continuous-time RC equivalent
    ema_alpha = 1.0 if smooth_strength <= 0 else (1.0 - np.exp(-hop_s / smooth_strength))

    tone_smooth = 0.0

    # plt.ion()
    # fig, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    # tone_plot  = RollingPlot(ax3, "Tone (Breathy-aware, 0..1)", "Tone (0..1)", 0.0, 1.0, 3.0, hop_s,
    #                           with_second=True, second_label="Tone (EMA)")
    # fig.tight_layout()


    while not stop_evt.is_set():
        try:
            arr = q_audio_therm.get(timeout=0.1)
        except pyqueue.Empty:
            # no new audio; just loop
            continue

        # quick gate on very low energy
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms < rms_gate:
            tone_mix = 0.0
        else:
            # choose nfft >= len(arr), power of two, min 512
            nfft = max(512, 1 << (len(arr) - 1).bit_length())
            _, cn, sfm, hfr, harm = tone_features(arr, Config.SAMPLERATE, nfft=nfft)

            # mix: brighter / noisier / high-freq -> hotter; harmonic (voiced) -> cooler
            # from your prototype: 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1 - harm)
            tone_mix = 0.45 * cn + 0.25 * sfm + 0.20 * hfr + 0.10 * (1.0 - harm)
            tone_mix = float(np.clip(tone_mix, 0.0, 1.0))

        # EMA smoothing
        tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
        # tone_plot.update(tone_mix, tone_smooth)
        therm = tone_smooth * Config.THERM_OUT_SCALE
        try:
            q_therm.put_nowait(therm)
        except pyqueue.Full:
            # if consumer is lagging, drop (we only want freshest)
            print("q_therm full!!!")
            pass
        # plt.pause(0.01)

