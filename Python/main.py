#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-process, multi-thread pipeline WITH detailed timing debug.

What we print:
- Audio callback:
  seq, cb_ns, adc_pts_ms, fixed_pts_ms, chosen_pts_ms, ΔPTS, Q sizes
- DSP (vib/therm):
  seq, wait_from_pts_ms, proc_ms, feature summary
- Commander:
  pending sizes, per frame: age_ms, until_due_ms, rate-limit decisions,
  send ΔPTS, lookahead, payload
- Playback:
  seq, schedule_delay_ms(target), actual_play_delay_ms, drift

Tip: set Cfg.DEBUG_LEVEL = 1~3 for verbosity control.
"""

import time
import threading
import queue
import math
from dataclasses import dataclass
from typing import Optional, List, Dict

import pyaudio
import numpy as np
import serial  # pyserial
from datafeel.device import Dot, VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms

# ===================== Config =====================

class Cfg:
    # Debug
    DEBUG = True
    DEBUG_LEVEL = 1             # 1=concise, 2=normal, 3=verbose

    
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CAP_HOP_MS = 10             # Hop = analysis hop
    CAP_WINDOW_MS = 20          # Window = frame size sent to DSP
    PLAYBACK_LATENCY_MS = 300  # audio playback buffer target (large for demo)
    LOOKAHEAD_MS = 25           # commander lookahead window
    TX_MAX_RATE_HZ = 150        # limit serial update rate (>= 100 for 10ms hop)
    RX_QUEUE_MAX = 256
    TX_QUEUE_MAX = 256
    VIB_QUEUE_MAX = 256
    THERM_QUEUE_MAX = 256
    DROP_POLICY = "drop_old"    # "drop_old" or "drop_new"

    DURATION_THRESHOLD = 0.1
    SMOOTH_STRENGTH = 0.15
    RMS_GATE = 2e-5
    EMA_ALPHA = 0.0


    NORMAL = 34
    HOTTEST = 36
    TONE_THRESHOLD = 0.025



# ===================== Utils ======================

def now_ns() -> int:
    return time.monotonic_ns()

def ms_to_ns(ms: float) -> int:
    return int(ms * 1e6)

def ns_to_ms(ns: int) -> float:
    return ns / 1e6

def dprint(level: int, *a):
    if Cfg.DEBUG and Cfg.DEBUG_LEVEL >= level:
        print(*a, flush=False)

def q_put(q: "queue.Queue", item, drop_policy="drop_old"):
    """Non-blocking put with bounded queue; apply drop policy consistently."""
    try:
        q.put_nowait(item)
    except queue.Full:
        if drop_policy == "drop_old":
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass
        # elif drop_new -> drop silently

def InitDots() -> list[Dot]:
    return discover_devices(4)

# ===================== Message Types ======================

@dataclass
class AudioFrame:
    seq: int
    pts_ns: int                 # chosen PTS for this hop (see AudioIO)
    hop_ns: int
    data: Optional[bytes]
    cb_ns: int                  # callback arrival monotonic time
    adc_time_s: Optional[float] = None   # PortAudio device time (sec) if available
    fixed_pts_ns: Optional[int] = None   # fixed-step pts (seq-based)
    adc_pts_ns: Optional[int] = None     # adc time mapped to monotonic
    # helper for debug
    prev_pts_ns: Optional[int] = None

@dataclass
class VibCmd:
    seq: int
    pts_ns: int
    duration_ms: int
    amp: float = 0.0
    band_energies: Optional[List[float]] = None

@dataclass
class ThermCmd:
    seq: int
    pts_ns: int
    duration_ms: int
    tone_smooth: float = 0.0

@dataclass
class CombinedCmd:
    seq: int
    pts_ns: int
    duration_ms: int
    vib: Optional[VibCmd] = None
    therm: Optional[ThermCmd] = None

# ===================== Global Queues ======================

Q_AUDIO_TO_VIB = queue.Queue(maxsize=Cfg.VIB_QUEUE_MAX)
Q_AUDIO_TO_THERM = queue.Queue(maxsize=Cfg.THERM_QUEUE_MAX)
Q_VIB_TO_CMD = queue.Queue(maxsize=Cfg.TX_QUEUE_MAX)
Q_THERM_TO_CMD = queue.Queue(maxsize=Cfg.TX_QUEUE_MAX)
Q_AUDIO_FOR_PLAYBACK = queue.Queue(maxsize=Cfg.RX_QUEUE_MAX)

STOP = threading.Event()

# ===================== Audio Capture ======================

class AudioIO:
    def __init__(self):
        self.pa = None
        self.stream_in = None
        self.stream_out = None
        self.frame_size = int(Cfg.SAMPLE_RATE * (Cfg.CAP_WINDOW_MS / 1000.0))
        self.hop_size = int(Cfg.SAMPLE_RATE * (Cfg.CAP_HOP_MS / 1000.0))
        self.bytes_per_sample = 4  # float32
        self._lock = threading.Lock()
        self._capture_buf = bytearray(self.frame_size * self.bytes_per_sample)
        self._buf_filled = 0

        # timing bases
        self._seq = 0
        self._fixed_base_ns = None
        self._base_adc_s = None
        self._base_mono_ns = None
        self._prev_pts_ns = None

    def start(self):
        self.pa = pyaudio.PyAudio()

        dprint(1, "=== Audio Devices ===")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            dprint(1, f"[{i}] {info['name']}")

        self.stream_in = self.pa.open(
            format=pyaudio.paFloat32,
            channels=Cfg.CHANNELS,
            rate=Cfg.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.hop_size,
            stream_callback=self._on_capture
        )
        self.stream_out = self.pa.open(
            format=pyaudio.paFloat32,
            channels=Cfg.CHANNELS,
            rate=Cfg.SAMPLE_RATE,
            output=True,
            frames_per_buffer=self.hop_size
        )
        self.stream_in.start_stream()
        self.stream_out.start_stream()
        threading.Thread(target=self._playback_loop, daemon=True).start()

    def stop(self):
        if self.stream_in:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.stream_out:
            self.stream_out.stop_stream()
            self.stream_out.close()
        if self.pa:
            self.pa.terminate()

    def _map_adc_to_mono_ns(self, adc_s: float) -> int:
        """Align device ADC time to monotonic ns axis."""
        if self._base_adc_s is None:
            self._base_adc_s = adc_s
            self._base_mono_ns = now_ns()
        return self._base_mono_ns + int((adc_s - self._base_adc_s) * 1e9)

    def _on_capture(self, in_data, frame_count, time_info, status):
        # Called in PortAudio thread context; keep it lightweight.
        if STOP.is_set():
            return (None, pyaudio.paComplete)

        cb_ns = now_ns()

        # Candidate 1: ADC time from device
        adc_s = time_info.get("input_buffer_adc_time", None)
        adc_pts_ns = self._map_adc_to_mono_ns(adc_s) if adc_s is not None else None

        # Candidate 2: fixed-step pts
        if self._fixed_base_ns is None:
            self._fixed_base_ns = cb_ns
        fixed_pts_ns = self._fixed_base_ns + self._seq * ms_to_ns(Cfg.CAP_HOP_MS)

        # Choose best PTS: prefer adc if available, else fixed
        chosen_pts = adc_pts_ns if adc_pts_ns is not None else fixed_pts_ns

        fr = AudioFrame(
            seq=self._seq,
            pts_ns=chosen_pts,
            hop_ns=ms_to_ns(Cfg.CAP_HOP_MS),
            data=in_data,
            cb_ns=cb_ns,
            adc_time_s=adc_s,
            fixed_pts_ns=fixed_pts_ns,
            adc_pts_ns=adc_pts_ns,
            prev_pts_ns=self._prev_pts_ns
        )
        self._prev_pts_ns = chosen_pts
        self._seq += 1

        # Sliding window buffer (if you later need window > hop)
        with self._lock:
            needed = len(self._capture_buf) - self._buf_filled
            take = min(needed, len(in_data))
            self._capture_buf[self._buf_filled:self._buf_filled+take] = in_data[:take]
            self._buf_filled += take

        # Enqueue to pipelines
        q_put(Q_AUDIO_TO_VIB, fr, Cfg.DROP_POLICY)
        q_put(Q_AUDIO_TO_THERM, fr, Cfg.DROP_POLICY)
        q_put(Q_AUDIO_FOR_PLAYBACK, fr, Cfg.DROP_POLICY)

        # Debug: print key times & jitter
        if fr.prev_pts_ns is None:
            delta_ms = None
        else:
            delta_ms = ns_to_ms(fr.pts_ns - fr.prev_pts_ns)

        q_sizes = (Q_AUDIO_TO_VIB.qsize(), Q_AUDIO_TO_THERM.qsize(), Q_AUDIO_FOR_PLAYBACK.qsize())
        dprint(2, f"[A-CB] seq={fr.seq} cb_ms={ns_to_ms(cb_ns):.3f} "
                  f"adc_pts_ms={(adc_pts_ns and ns_to_ms(adc_pts_ns)) or None:.3f} "
                  f"fixed_pts_ms={ns_to_ms(fixed_pts_ns):.3f} "
                  f"chosen_pts_ms={ns_to_ms(chosen_pts):.3f} "
                  f"ΔPTS={(delta_ms is not None) and f'{delta_ms:.3f}' or '—'} "
                  f"Q(vib,therm,play)={q_sizes}")

        # Maintain overlap-add buffer position (optional)
        with self._lock:
            if self._buf_filled >= len(self._capture_buf):
                slide_bytes = self.hop_size * self.bytes_per_sample
                self._capture_buf = self._capture_buf[slide_bytes:] + bytearray(slide_bytes)
                self._buf_filled = len(self._capture_buf) - slide_bytes

        return (None, pyaudio.paContinue)

    def _playback_loop(self):
        if self.stream_out is None:
            return
        target_delay_ns = ms_to_ns(Cfg.PLAYBACK_LATENCY_MS)
        while not STOP.is_set():
            try:
                fr: AudioFrame = Q_AUDIO_FOR_PLAYBACK.get(timeout=0.05)
            except queue.Empty:
                continue

            due_ns = fr.pts_ns + target_delay_ns
            now = now_ns()
            if due_ns > now:
                time.sleep((due_ns - now) / 1e9)
            try:
                self.stream_out.write(fr.data)
            except Exception as e:
                dprint(1, "[Playback] write error:", e)

            # Debug: actual play delay vs target
            play_ns = now_ns()
            actual_delay_ms = ns_to_ms(play_ns - fr.pts_ns)
            drift_ms = actual_delay_ms - Cfg.PLAYBACK_LATENCY_MS
            dprint(2, f"[Play ] seq={fr.seq} target_ms={Cfg.PLAYBACK_LATENCY_MS} "
                      f"actual_ms={actual_delay_ms:.3f} drift_ms={drift_ms:.3f} "
                      f"Qplay={Q_AUDIO_FOR_PLAYBACK.qsize()}")

# ===================== DSP Threads ======================

def dsp_vib_thread():
    hop_ms = Cfg.CAP_HOP_MS
    while not STOP.is_set():
        try:
            fr: AudioFrame = Q_AUDIO_TO_VIB.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = now_ns()
        # simple RMS
        amp = 0.0
        if fr.data:
            try:
                arr = np.frombuffer(fr.data, dtype=np.float32)
                rms = math.sqrt(float(np.mean(arr * arr)) + 1e-12)
                amp = min(1.0, rms * 30.0)
            except Exception:
                pass
        t1 = now_ns()

        wait_ms = ns_to_ms(t0 - fr.pts_ns)       # from capture pts to DSP start
        proc_ms = ns_to_ms(t1 - t0)              # DSP compute time
        dprint(3, f"[DSP-V] seq={fr.seq} wait_ms={wait_ms:.3f} proc_ms={proc_ms:.3f} amp={amp:.3f} "
                  f"Qvib_in={Q_AUDIO_TO_VIB.qsize()}")

        vib = VibCmd(
            seq=fr.seq,
            pts_ns=fr.pts_ns,
            duration_ms=hop_ms,
            amp=amp,
            band_energies=None
        )
        q_put(Q_VIB_TO_CMD, vib, Cfg.DROP_POLICY)

def dsp_therm_thread():

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


    hop_length = int(Cfg.SAMPLE_RATE * (Cfg.CAP_HOP_MS / 1000.0))
    hop_s = hop_length / float(Cfg.SAMPLE_RATE)

    if Cfg.SMOOTH_STRENGTH <= 0:
        ema_alpha = 1.0
    else:
        ema_alpha = 1.0 - np.exp(-hop_s / Cfg.SMOOTH_STRENGTH)

    tone_mix = 0.0
    tone_smooth = 0.0

    hop_ms = Cfg.CAP_HOP_MS
    while not STOP.is_set():
        try:
            fr: AudioFrame = Q_AUDIO_TO_THERM.get(timeout=0.1)
        except queue.Empty:
            continue

        t0 = now_ns()
        tone_smooth = 0.0
        if fr.data:
            try:
                # arr = np.frombuffer(fr.data, dtype=np.float32)
                # s = np.sign(arr)
                # crossings = np.where(np.diff(s) != 0)[0].size
                # est_f0 = crossings * (Cfg.SAMPLE_RATE / (2.0 * len(arr) + 1e-9))
                # tone_hz = float(est_f0)
                arr = np.frombuffer(fr.data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(arr*arr)))
                if rms < Cfg.RMS_GATE:
                    tone_mix = 0.0
                else:
                    _, cn, sfm, hfr, harm = tone_features(arr, Cfg.SAMPLE_RATE)
                    tone_mix = 0.45*cn + 0.25*sfm + 0.20*hfr + 0.10*(1.0 - harm)
                    tone_mix = float(np.clip(tone_mix, 0.0, 1.0))

                tone_smooth = (1.0 - ema_alpha) * tone_smooth + ema_alpha * tone_mix
                tone_smooth = float(tone_smooth)
            except Exception:
                pass

        t1 = now_ns()

        wait_ms = ns_to_ms(t0 - fr.pts_ns)
        proc_ms = ns_to_ms(t1 - t0)
        dprint(3, f"[DSP-T] seq={fr.seq} wait_ms={wait_ms:.3f} proc_ms={proc_ms:.3f} "
                  f"tone={tone_smooth:.1f} Qtherm_in={Q_AUDIO_TO_THERM.qsize()}")

        therm = ThermCmd(
            seq=fr.seq,
            pts_ns=fr.pts_ns,
            duration_ms=hop_ms,
            tone_smooth=tone_smooth,
        )
        q_put(Q_THERM_TO_CMD, therm, Cfg.DROP_POLICY)

# ===================== Commander (Serial owner) ======================

class Commander:
    def __init__(self):
        self.device = None
        self.last_tx_ns = 0
        self.min_tx_interval_ns = int(1e9 / Cfg.TX_MAX_RATE_HZ)
        self.lookahead_ns = ms_to_ns(Cfg.LOOKAHEAD_MS)
        self.prev_vib_amp = 0.0
        self.prev_therm_lvl = 0.0
        self.max_amp_delta = 0.1
        self.max_therm_delta = 0.05
        self._last_sent_pts: Optional[int] = None

        self.state = 0
        self.last_trigger = time.monotonic()
        self.lastCool_trigger = time.monotonic()
        self.startWarm = False
        self.startCool = False
        self.tone_smooth = 0.0

        self._last_seq_sent = None

    def start(self):
        try:
            devices = InitDots()
            self.device = devices[0]
            self.device.registers.set_vibration_mode(VibrationMode.MANUAL)
            self.device.registers.set_thermal_mode(ThermalMode.MANUAL)
            dprint(1, f"[Cmd  ] Dot initialized: {self.device}")
        except Exception as e:
            dprint(1, "[Cmd  ] Device open failed, using stub:", e)
            self.device = None
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        if self.device:
            try:
                self.device.stop_vibration()
                self.device.disable_all_thermal()
                self.device.set_led_off()
            except Exception:
                dprint(1, "[Cmd  ] Error stopping device")
                pass

    def _loop(self):
        pending_vib: Dict[int, VibCmd] = {}
        pending_therm: Dict[int, ThermCmd] = {}

        while not STOP.is_set():
            # 1) drain incoming queues
            got = False
            while True:
                pulled = False
                try:
                    v = Q_VIB_TO_CMD.get_nowait()
                    pending_vib[v.pts_ns] = v
                    pulled = True
                except queue.Empty:
                    pass
                try:
                    t = Q_THERM_TO_CMD.get_nowait()
                    pending_therm[t.pts_ns] = t
                    pulled = True
                except queue.Empty:
                    pass
                if not pulled:
                    break
                got = True

            now = now_ns()
            cutoff = now + self.lookahead_ns

            due_pts = sorted(set(list(pending_vib.keys()) + list(pending_therm.keys())))
            # Debug pending
            if got and due_pts:
                dprint(3, f"[Cmd  ] pending: vib={len(pending_vib)} therm={len(pending_therm)} "
                          f"next_pts_ms={ns_to_ms(due_pts[0]):.3f} now_ms={ns_to_ms(now):.3f}")

            for pts in due_pts:
                if pts > cutoff:
                    break

                # Peek (do not pop yet)
                vib = pending_vib.get(pts)
                thr = pending_therm.get(pts)
                seq = vib.seq if vib else (thr.seq if thr else -1)
                comb = CombinedCmd(seq=seq, pts_ns=pts, duration_ms=Cfg.CAP_HOP_MS, vib=vib, therm=thr)

                # rate limit: if too soon since last send, wait (don't drop)
                if now - self.last_tx_ns < self.min_tx_interval_ns:
                    rl_ms = ns_to_ms(self.min_tx_interval_ns - (now - self.last_tx_ns))
                    dprint(3, f"[Cmd  ] seq={seq} RATE-LIMIT wait~{rl_ms:.3f} ms")
                    break  # wait to next loop to try again

                # Apply slew-rate limits
                self._apply_slew(comb)

                # Pop only when sending
                pending_vib.pop(pts, None)
                pending_therm.pop(pts, None)

                age_ms = ns_to_ms(now - pts)
                until_due_ms = ns_to_ms((pts + self.lookahead_ns) - now)
                dprint(2, f"[Cmd  ] seq={seq} age_ms={age_ms:.3f} until_due_ms={until_due_ms:.3f} "
                          f"Qout(v,t)={len(pending_vib)},{len(pending_therm)}")


                send_start = now_ns()
                self._send_combined(comb)
                send_ms = ns_to_ms(now_ns() - send_start)
                dprint(1, f"[Cmd  ] send_ms={send_ms:.3f}ms Qin(v,t)={Q_VIB_TO_CMD.qsize()},{Q_THERM_TO_CMD.qsize()}")
                
                self.last_tx_ns = now_ns()

            time.sleep(0.001)

    def _apply_slew(self, comb: CombinedCmd):
        if comb.vib:
            delta = comb.vib.amp - self.prev_vib_amp
            if abs(delta) > self.max_amp_delta:
                comb.vib.amp = self.prev_vib_amp + math.copysign(self.max_amp_delta, delta)
            self.prev_vib_amp = comb.vib.amp
        if comb.therm:
            delta = comb.therm.tone_smooth - self.prev_therm_lvl
            if abs(delta) > self.max_therm_delta:
                comb.therm.tone_smooth = self.prev_therm_lvl + math.copysign(self.max_therm_delta, delta)
            self.prev_therm_lvl = comb.therm.tone_smooth

    def _send_combined(self, comb: CombinedCmd):
        payload = {
            "seq": comb.seq,
            "pts_ms": round(ns_to_ms(comb.pts_ns), 3),
            "dur_ms": comb.duration_ms,
            "vib_amp": comb.vib.amp if comb.vib else None,
            "therm_lvl": comb.therm.tone_smooth if comb.therm else None,
        }
        line = (str(payload) + "\n").encode("utf-8")

        if self.device:
            try:
                dprint(1, f"[Cmd  ] seq={comb.seq} vib_amp={comb.vib.amp:.3f} therm_lvl={comb.therm.tone_smooth:.3f}")
                self.device.play_frequency(70, comb.vib.amp)
                
                if abs(self.device.registers.get_skin_temperature() - Cfg.NORMAL) < 1:
                    # print("Stop Thermal...")
                    state = 0
                    self.device.registers.set_thermal_intensity(0.0)
                elif self.device.registers.get_skin_temperature() >= Cfg.HOTTEST + 2:
                    state = -1
                    # print("Force Cooling...")
                    self.device.registers.set_thermal_intensity(-1.0)
                elif self.device.registers.get_skin_temperature() < Cfg.NORMAL:
                    state = 1
                    # print("Force Return...")
                    self.device.registers.set_thermal_intensity(1.0)
                else :
                    state = 0

                # Thermal Actuation
                if comb.therm.tone_smooth >= Cfg.TONE_THRESHOLD:
                    if self.startWarm == False:
                        self.last_trigger = time.monotonic()
                        self.startWarm = True
                    elif time.monotonic() - self.last_trigger >= Cfg.DURATION_THRESHOLD:
                        if self.device.registers.get_thermal_power() <= 0.01 and state == 0:
                            print("Heating...")
                            self.device.registers.set_thermal_intensity(1.0)
                elif comb.therm.tone_smooth < Cfg.TONE_THRESHOLD:
                    if self.startCool == False:
                        self.lastCool_trigger = time.monotonic()
                        self.startCool = True
                    elif time.monotonic() - self.lastCool_trigger >= Cfg.DURATION_THRESHOLD*10:
                        self.startWarm = False
                        self.startCool = False
                        if self.device.registers.get_thermal_power() >= 0.5 and state == 0:
                            print("Cooling...")
                            self.device.registers.set_thermal_intensity(-1.0)

            except Exception as e:
                dprint(1, "[Cmd  ] device write error:", e)
        # Print stub or mirror even if serial present (debug)
        now = now_ns()
        delta_pts = None if self._last_sent_pts is None else ns_to_ms(comb.pts_ns - self._last_sent_pts)
        self._last_sent_pts = comb.pts_ns
        age_ms = ns_to_ms(now - comb.pts_ns)
        dprint(2, f"[Cmd->] {payload} age_ms={age_ms:.3f} ΔPTS={delta_pts and f'{delta_pts:.3f}' or '—'}")

# ===================== Bootstrapping ======================

def main():
    audio = AudioIO()
    cmd = Commander()

    audio.start()
    cmd.start()

    vib_t = threading.Thread(target=dsp_vib_thread, daemon=True)
    thm_t = threading.Thread(target=dsp_therm_thread, daemon=True)
    vib_t.start()
    thm_t.start()

    dprint(1, "Running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        STOP.set()
        audio.stop()
        cmd.stop()
        dprint(1, "Stopped.")

if __name__ == "__main__":
    main()
