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

def VoiceToThermalData(ev:Event, q_Audio:Queue, parent_conn:Connection, thermal_queue:Queue, filePath: str = "", sr: int = 16000, frame_length:int = 512, hop_length:int = 128, block_length:int = 128, rms_gate: float = 2e-5, smooth_strength: float = 0.15, saveToCSV: bool = False, saveCSVPath: str = "./tone.csv"):
    
    hop_s = hop_length / float(sr)
    if smooth_strength <= 0:
        ema_alpha = 1.0
    else:
        ema_alpha = 1.0 - np.exp(-hop_s / smooth_strength)

    tone_mix = 0.0 ; tone_smooth = 0.0


    while True:
        try:
            start_time = time.monotonic()
            y = q_Audio.get(timeout=0.01)
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < rms_gate:
                tone_mix = 0.0
            else:

                _, cn, sfm, hfr, harm = tone_features(y, sr)
                tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
                tone_mix = float(np.clip(tone_mix, 0.0, 1.0))

            """平滑處理(EMA)"""

            tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
            print(time.monotonic() - start_time)
        except queue.Empty:
            pass
        # plt.pause(0.01)

