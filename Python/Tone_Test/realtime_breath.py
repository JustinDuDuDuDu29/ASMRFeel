#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import numpy as np
import sounddevice as sd
import collections
import queue

from time import sleep
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms

# ---------------- Defaults ----------------
SR    = 16000
FRAME = 512     # ~11.6ms/256
HOP   = 128     # ~5.8ms/128
BLOCK = 128     # device callback hop
RMS_GATE = 2e-5 # 靜音門檻（依麥克風調整）
SMOOTH_STRENGTH = 0.15

# ---------------- Thermal Setting ----------------
ACTIVE_THERMAL = False
NORMAL = 34
HOTEST = 36
TONE_THRESHOLD = 0.25
DURATION_THRESHOLD = 0.1

if ACTIVE_THERMAL:
    devices = discover_devices(4)
    device = devices[0]

# ---------------- Feature helpers ----------------
def _hz_to_bin(hz, nfft, sr):
    return int(np.clip(np.floor(hz/(sr/nfft)), 0, nfft//2))

def tone_features(frame, sr, nfft=512):
    """計算 Tone 所需特徵：譜心(Hz/Norm)、SFM、HF_ratio、Harmonicity(代理)"""
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

def spectral_centroid(frame, sr):
    nfft = 512
    w = np.hanning(len(frame))
    spec = np.fft.rfft(frame*w, n=nfft)
    mag  = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(nfft, d=1.0/sr)
    return float(np.sum(freqs * mag) / np.sum(mag))

def autocorr_pitch(frame, sr, fmin=50.0, fmax=800.0):
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

# ---------------- Plot wrapper ----------------
class RollingPlot:
    """維護一個折線圖的滾動視窗資料（長度 = window_s）。支援第二條線（黃色）。"""
    def __init__(self, ax, title, y_label, y_min, y_max, window_s, hop_s, with_second=False, second_label=None):
        import matplotlib.pyplot as plt
        self.ax = ax
        self.window_s = window_s
        self.hop_s = hop_s
        self.max_points = max(2, int(window_s / hop_s))
        self.xs = collections.deque(maxlen=self.max_points)
        self.ys = collections.deque(maxlen=self.max_points)
        (self.line,) = self.ax.plot([], [], linewidth=1.6)
        self.line2 = None
        self.ys2 = None
        if with_second:
            (self.line2,) = self.ax.plot([], [], 'y', linewidth=1.6)  # 黃色線
            self.ys2 = collections.deque(maxlen=self.max_points)
            if second_label:
                self.line2.set_label(second_label)
                self.ax.legend(loc="upper right", frameon=False, fontsize=9)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(y_label)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(0, window_s)
        self._t0 = time.time()

    def update(self, y, y2=None):
        import numpy as np
        t = time.time() - self._t0
        self.xs.append(t)
        self.ys.append(y)
        if self.line2 is not None and y2 is not None:
            self.ys2.append(y2)
        # 顯示最近 window_s 秒
        x0 = self.xs[-1] - self.window_s if len(self.xs) > 0 else 0.0
        xs = np.array(self.xs) - max(x0, 0.0)
        ys = np.array(self.ys)
        self.line.set_data(xs, ys)
        if self.line2 is not None and self.ys2 is not None and len(self.ys2) > 0:
            ys2 = np.array(self.ys2)
            self.line2.set_data(xs[-len(ys2):], ys2)  # 對齊長度
        self.ax.set_xlim(0, self.window_s)

# ---------------- File loader ----------------
def load_audio_mono(filepath, target_sr):
    """
    讀入本地音檔並回傳 mono float32、取樣率=target_sr 的 numpy array。
    - MP3/FLAC/WAV 皆可（MP3 透過 pydub 需 ffmpeg）
    """
    try:
        from pydub import AudioSegment
    except Exception as e:
        raise RuntimeError("需要 pydub 來讀 MP3/各種壓縮格式，請先 `pip install pydub` 並安裝 ffmpeg") from e

    audio = AudioSegment.from_file(filepath)
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    audio = audio.set_channels(1).set_sample_width(2)
    raw = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return raw, target_sr

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--frame", type=int, default=FRAME)
    parser.add_argument("--hop", type=int, default=HOP)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--device", type=str, default=None, help="input device index/name (mic mode)")
    parser.add_argument("--outdev", type=str, default=None, help="output device index/name (file play)")
    parser.add_argument("--win", type=float, default=3.0, help="seconds displayed in plots")
    parser.add_argument("--lowlatency", action="store_true")
    parser.add_argument("--file", type=str, default=None, help="path to local audio file (e.g., .mp3/.wav)")
    parser.add_argument("--play", action="store_true", help="play audio while analyzing (file mode)")
    parser.add_argument("--smooth_tau", type=float, default=SMOOTH_STRENGTH, help="EMA time constant for tone smoothing (seconds); 0 to disable")
    args = parser.parse_args()

    sr = args.sr
    frame = args.frame
    hop = args.hop
    block = args.block
    if args.lowlatency:
        frame = 128; hop = 64; block = 64

    hop_s = hop / float(sr)

    # 由 tau 推 alpha；tau<=0 表示關閉平滑（alpha=1 → 直接等於當前值）
    if args.smooth_tau <= 0:
        ema_alpha = 1.0
    else:
        ema_alpha = 1.0 - np.exp(-hop_s / args.smooth_tau)

    # Plot window（Tone 軸啟用第二條黃色線）
    import matplotlib
    import matplotlib.pyplot as plt
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    amp_plot  = RollingPlot(ax1, "Amplitude (RMS, gated)", "RMS (0..1)", 0.0, 1.0, args.win, hop_s)
    pitch_plot = RollingPlot(ax2, "Pitch (F0)", "Frequency (Hz)", 0.0, 800.0, args.win, hop_s)
    tone_plot  = RollingPlot(ax3, "Tone (Breathy-aware, 0..1)", "Tone (0..1)", 0.0, 1.0, args.win, hop_s,
                              with_second=True, second_label="Tone (EMA)")
    fig.tight_layout()

    q = queue.Queue(maxsize=64)
    sample_buf = np.zeros(frame, dtype=np.float32)

    def push_to_queue(x):
        try:
            while q.qsize() > 4:
                q.get_nowait()
            q.put_nowait(x)
        except queue.Full:
            pass

    # ---------- 麥克風模式 ----------
    def mic_mode():
        def audio_callback(indata, frames, time_info, status):
            nonlocal sample_buf

            # start = time.perf_counter()

            if status: pass
            mono = indata.reshape(-1)
            need = hop
            if len(mono) >= need:
                new = mono[:need]
            else:
                new = np.pad(mono, (0, need-len(mono)), mode='constant')

            sample_buf[:-need] = sample_buf[need:]
            sample_buf[-need:] = new

            '''
            rms = float(np.sqrt(np.mean(sample_buf**2)))
            if rms < RMS_GATE:
                rms_norm = 0.0; f0 = 0.0; tone_mix = 0.0
            else:
                rms_norm = float(rms / (rms + 0.2))
                f0 = autocorr_pitch(sample_buf, sr)
                _, cn, sfm, hfr, harm = tone_features(sample_buf, sr)
                tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
                tone_mix = float(np.clip(tone_mix, 0.0, 1.0))
            push_to_queue(rms_norm, f0, tone_mix)
            '''
            push_to_queue(sample_buf.copy())

            
            # elapsed = (time.perf_counter() - start) * 1000  # ms
            # print(f"Callback took {elapsed:.3f} ms")

        stream_kwargs = dict(samplerate=sr, channels=1, blocksize=block,
                             dtype='float32', callback=audio_callback)
        stream_kwargs["device"] = 1

        with sd.InputStream(**stream_kwargs):
            print("Running (mic). Ctrl+C to stop.")
            last_log = time.time()
            last_trigger = time.time()
            lastCool_trigger = time.time()
            start = False
            startCool = False
            tone_smooth = 0.0  # EMA 狀態
            state = 0
            rms_norm = 0.0 ; f0 = 0.0 ; tone_mix = 0.0 
            try:
                while True:
                    try:
                        # ------Sample Process------------
                        start_time = time.perf_counter()
                        # rms_norm, f0, tone_mix = q.get(timeout=0.05)
                        y = q.get(timeout=0.05)
                        rms = float(np.sqrt(np.mean(y**2)))
                        if rms < RMS_GATE:
                            rms_norm = 0.0; f0 = 0.0; tone_mix = 0.0
                        else:
                            rms_norm = float(rms / (rms + 0.2))
                            f0 = autocorr_pitch(y, sr)
                            _, cn, sfm, hfr, harm = tone_features(y, sr)
                            tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
                            tone_mix = float(np.clip(tone_mix, 0.0, 1.0))

                        # ---- 平滑處理（EMA） ----
                        tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
                        
                        # print(f"process timeout: {time.perf_counter() - start_time}")
                        # ------Sample Process------------

                        # ------Thermal Feedback------------------
                        start_time = time.perf_counter()
                        if ACTIVE_THERMAL:
                            if abs(device.registers.get_skin_temperature() - NORMAL) < 1:
                                print("Stop Thermal...")
                                state = 0
                                device.registers.set_thermal_mode(ThermalMode.OFF)
                            elif device.registers.get_skin_temperature() >= HOTEST + 2:
                                state = -1
                                print("Force Cooling...")
                                device.registers.set_thermal_mode(ThermalMode.MANUAL)
                                device.registers.set_thermal_intensity(-1.0)
                            elif device.registers.get_skin_temperature() < NORMAL:
                                state = 1
                                print("Force Return...")
                                device.registers.set_thermal_mode(ThermalMode.MANUAL)
                                device.registers.set_thermal_intensity(1.0)
                            else :
                                state = 0
                            
                            if tone_smooth >= TONE_THRESHOLD:
                                if start == False:
                                    last_trigger = time.time()
                                    start = True
                                elif time.time() - last_trigger >= DURATION_THRESHOLD:
                                    if device.registers.get_thermal_power() <= 0.01 and state == 0:
                                        print("Heating...")
                                        device.registers.set_thermal_mode(ThermalMode.MANUAL)
                                        device.registers.set_thermal_intensity(1.0)
                                    # elif device.registers.get_skin_temperature() >= HOTEST:
                                    #     print("Stop Heating...")
                                    #     start = False
                                    #     device.registers.set_thermal_mode(ThermalMode.OFF)
                            elif tone_smooth < TONE_THRESHOLD:
                                if startCool == False:
                                    lastCool_trigger = time.time()
                                    startCool = True
                                elif time.time() - lastCool_trigger >= DURATION_THRESHOLD*10:
                                    start = False
                                    startCool = False
                                    if device.registers.get_thermal_power() >= 0.5 and state == 0:
                                        print("Cooling...")
                                        device.registers.set_thermal_mode(ThermalMode.MANUAL)
                                        device.registers.set_thermal_intensity(-1.0)
                                    # elif device.registers.get_skin_temperature() <= NORMAL:
                                    #     print("Stop Thermal...")
                                    #     device.registers.set_thermal_mode(ThermalMode.OFF)

                        # print(f"thermal timeout: {time.perf_counter() - start_time}")
                        # ------Thermal Feedback------------------

                        amp_plot.update(rms_norm)
                        pitch_plot.update(f0)
                        tone_plot.update(tone_mix, tone_smooth)  # 畫兩條：白色=原值、黃色=EMA
                    except queue.Empty:
                        pass
                    plt.pause(0.01)
                    if time.time() - last_log >= 0.5:
                        last_log = time.time()
                        if ACTIVE_THERMAL: print(f"RMS={rms_norm:.4f}, F0={f0:.1f} Hz, Tone={tone_mix:.2f}, ToneEMA={tone_smooth:.2f}, NowTemp={device.registers.get_skin_temperature()}")
                        else : print(f"RMS={rms_norm:.4f}, F0={f0:.1f} Hz, Tone={tone_mix:.2f}, ToneEMA={tone_smooth:.2f}")
            except KeyboardInterrupt:
                pass

    # ---------- 檔案模式 ----------
    # def file_mode(path, play=False):
    #     x, sr_file = load_audio_mono(path, sr)
    #     print(f"Loaded file: {path}  len={len(x)/sr:.2f}s  sr={sr}")

    #     pos = 0
    #     ana_accum = 0
    #     local_buf = np.zeros(frame, dtype=np.float32)
    #     ana_hop = hop

    #     def out_callback(outdata, frames, time_info, status):
    #         nonlocal pos, ana_accum, local_buf
    #         if status: pass

    #         end = min(pos + frames, len(x))
    #         chunk = x[pos:end]
    #         if len(chunk) < frames:
    #             pad = np.zeros(frames - len(chunk), dtype=np.float32)
    #             chunk = np.concatenate([chunk, pad])
    #         outdata[:] = chunk.reshape(-1, 1)
    #         pos = end

    #         # 分析與播放同步推進
    #         remain = frames
    #         offset = 0
    #         while remain > 0:
    #             take = min(ana_hop - ana_accum, remain)
    #             local_buf[:-take] = local_buf[take:]
    #             local_buf[-take:] = chunk[offset:offset+take]
    #             ana_accum += take
    #             offset += take
    #             remain -= take

    #             if ana_accum >= ana_hop:
    #                 ana_accum = 0
    #                 rms = float(np.sqrt(np.mean(local_buf**2)))
    #                 if rms < RMS_GATE:
    #                     rms_norm = 0.0; f0 = 0.0; tone_mix = 0.0
    #                 else:
    #                     rms_norm = float(rms / (rms + 0.2))
    #                     f0 = autocorr_pitch(local_buf, sr)
    #                     _, cn, sfm, hfr, harm = tone_features(local_buf, sr)
    #                     tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
    #                     tone_mix = float(np.clip(tone_mix, 0.0, 1.0))
    #                 try:
    #                     while q.qsize() > 4:
    #                         q.get_nowait()
    #                     q.put_nowait((rms_norm, f0, tone_mix))
    #                 except queue.Full:
    #                     pass

    #         if pos >= len(x):
    #             raise sd.CallbackStop

    #     out_kwargs = dict(samplerate=sr, channels=1, dtype='float32',
    #                       blocksize=hop, callback=out_callback)
    #     if play and hasattr(args, "outdev") and args.outdev is not None:
    #         try:
    #             out_kwargs["device"] = int(args.outdev)
    #         except ValueError:
    #             cands = [i for i, d in enumerate(sd.query_devices())
    #                      if d['max_output_channels'] > 0 and d['name'] == args.outdev]
    #             if len(cands) != 1:
    #                 raise ValueError(f"Output device name must match exactly one device; got {len(cands)} matches.")
    #             out_kwargs["device"] = cands[0]

    #     print("Running (file mode, callback-driven).")
    #     try:
    #         with sd.OutputStream(**out_kwargs):
    #             last_log = time.time()
    #             tone_smooth = 0.0  # EMA 狀態
    #             while True:
    #                 try:
    #                     rms_norm, f0, tone_mix = q.get(timeout=0.05)
    #                     # ---- 平滑處理（EMA） ----
    #                     tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
    #                     # ------------------------
    #                     amp_plot.update(rms_norm)
    #                     pitch_plot.update(f0)
    #                     tone_plot.update(tone_mix, tone_smooth)
    #                 except queue.Empty:
    #                     pass
    #                 plt.pause(0.01)
    #                 if time.time() - last_log >= 0.5:
    #                     last_log = time.time()
    #                     print(f"RMS={rms_norm:.4f}, F0={f0:.1f} Hz, Tone={tone_mix:.2f}, ToneEMA={tone_smooth:.2f}")
    #     except sd.CallbackStop:
    #         print("File finished.")

    # 選模式
    # if args.file:
    #     file_mode(args.file, play=args.play)
    # else:
    mic_mode()

if __name__ == "__main__":
    main()
