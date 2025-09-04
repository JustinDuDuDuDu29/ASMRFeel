import multiprocessing
import sys
import serial
from serial.tools import list_ports
from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import random

import queue as pyqueue
import pyaudio
import time
import numpy as np
import matplotlib.pyplot as plt
import Plot_Window.RollingPlot

class Config:
    INPUT_DEVICE_INDEX = 0
    OUTPUT_DEVICE_INDEX = 1
    SAMPLERATE = 16000
    AUDIO_CHUNK_MS = 70

    AUDIO_PLAYBACK_DELAY_S = 0.5


    VIB_OUT_SCALE = 100


    THERM_OUT_SCALE = 1

# -------------------- Audio Capture --------------------

def AudioCapture(stop_evt: Event, q_audio_playback: Queue, q_audio_vib: Queue, q_audio_therm: Queue, sr=Config.SAMPLERATE, chunk_ms=Config.AUDIO_CHUNK_MS):
    """Continuously capture audio in 70 ms frames and put the latest into q_audio."""
    pa = pyaudio.PyAudio()
    # print all device
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
    framesize = int(Config.SAMPLERATE * chunk_ms / 1000)

    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=sr,
                     input=True,
                     input_device_index=Config.INPUT_DEVICE_INDEX,
                     frames_per_buffer=framesize)

    while not stop_evt.is_set():
        data = stream.read(framesize, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.float32)
        try:
            q_audio_playback.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_playback queue is full!!!")

        try:
            q_audio_vib.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_vib queue is full!!!")

        try:
            q_audio_therm.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_therm queue is full!!!")

    stream.stop_stream()
    stream.close()
    pa.terminate()

# -------------------- Audio Playback --------------------

def AudioPlayback(stop_evt: Event, q_audio_playback: Queue, playback_delay=Config.AUDIO_PLAYBACK_DELAY_S, sr=Config.SAMPLERATE):
    """Play audio from q_audio with a delay (default 300ms)."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=sr,
                     output=True,
                     output_device_index=Config.OUTPUT_DEVICE_INDEX)

    buffer = []
    start_time = time.time()

    while not stop_evt.is_set():
        try:
            arr = q_audio_playback.get(timeout=0.05)
            buffer.append(arr)
        except pyqueue.Empty:
            pass

        # Wait until playback_delay has passed
        if buffer and (time.time() - start_time) > playback_delay:
            stream.write(buffer.pop(0).tobytes())

    stream.stop_stream()
    stream.close()
    pa.terminate()


def dsp_vib(stop_evt: Event, q_audio_vib: Queue, q_vib: Queue):
    """Extract RMS from latest audio and push to q_vib."""
    while not stop_evt.is_set():
        try:
            arr = q_audio_vib.get(timeout=0.1)
            rms = float(np.sqrt(np.mean(arr**2)))
            vib_level = rms * Config.VIB_OUT_SCALE
            try:
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
    # tone_plot  = Plot_Window.RollingPlot.RollingPlot(ax3, "Tone (Breathy-aware, 0..1)", "Tone (0..1)", 0.0, 1.0, 3.0, hop_s,
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

def worker(stop_evt:Event, q_cmd: Queue):
    dfc = DataFeelCenter(numOfDots=4)  
    
    while not stop_evt.is_set():
        try:
            while not q_cmd.empty():
                # print current time stamp ms
                # print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.{int(time.time() * 1000) % 1000:03d}")
                cmd = q_cmd.get_nowait()
                if not cmd:
                    return
                method, args = cmd

                getattr(dfc, method)(*args)

        except Exception as e:
            print(f"worker Err! {e}")
            stop_evt.set()
            break

def choose_port(default=None):
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    prompt = f"Select port index [{0 if default is None else default}]: "
    try:
        idx = input(prompt).strip()
        idx = int(idx) if idx else (0 if default is None else default)
    except ValueError:
        idx = 0
    return ports[idx].device


def Commander(stop_evt: Event, q_pres:Queue, q_vib:Queue, q_therm:Queue, q_cmd:Queue):
    while not stop_evt.is_set():
        t = q_pres.get()
        temp, pres1, pres2 = t.split(";")
        p = pres1.split(",")
        p1 = pres2.split(",")
        vib =None 
        if not q_vib.empty():
            vib = q_vib.get(block=False)
            

        therm=None 
        if not q_therm.empty():
            therm = q_therm.get(block=False)


        # print(q_pres.empty(), q_vib.empty(), q_therm.empty())
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())

        # print(f"pres {t1 - t0:.3f}s, vib {t2 - t1:.3f}s, therm {t3 - t2:.3f}s")

        led = [[0,0,0]]*8

        for i, val in enumerate(p):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                led[i] = [255, 0, 0]

        led1 = [[0,0,0]]*8

        for i, val in enumerate(p1):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                led1[i] = [255, 0, 0]


        # print(vib, therm)
        t0 = token(superDotID = 0, vibFrequency=70, vibIntensity=vib, therIntensity=0, ledList=led)
        t1 = token(superDotID = 1, vibFrequency=70, vibIntensity=vib, therIntensity=0, ledList=led1)
        # print(t0)
        try:
            q_cmd.put_nowait(("useToken", (t0, )))
            q_cmd.put_nowait(("useToken", (t1, )))
            # print(time.perf_counter()-last)
            # last = time.perf_counter()

        except pyqueue.Full:
            print("q_cmd full!!!")


def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
    """Line-framed reader. Reconnects on failure."""

    while not stop_evt.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                # set receive buffer to 1
                # ser.set_buffer_size(rx_size=0, tx_size=0)
                ser.reset_input_buffer()
                
                # read buffer size
                while not stop_evt.is_set():
                    try:
                        # print(f"is Empty? {q.empty()}")
                        line = ser.readline()  
                        # buffer_size = ser.in_waiting
                        # print(f"[serial] connected to {port} at {baud} baud, buffer size {buffer_size}")
                        if line:
                            d = line.rstrip(b"\r\n").decode("utf-8")
                            q.put(d)


                            # workaround: because arduino clock is different from pc, we need to adjust the timing
                            # try:
                            #     q.put_nowait(d)
                            # except pyqueue.Full:
                            #     try:
                            #         q.get_nowait()
                            #     except pyqueue.Empty:
                            #         pass
                            #     try:
                            #         q.put_nowait(d)
                            #     except pyqueue.Full:
                            #         pass
                    except serial.SerialException as e:
                        print(f"[serial] read error: {e}; will reconnect")
                        break  
        except serial.SerialException as e:
            print(f"[serial] open error on {port}: {e}; retrying...")
        stop_evt.wait(1.0)

def main():
    baud = 115200
    port = choose_port()
    print(f"Starting connection at {port} {baud}…")

    q_audio_playback = Queue(maxsize=1)
    q_audio_vib = Queue(maxsize=1)
    q_audio_therm = Queue(maxsize=1)

    q_pres = Queue()
    q_vib = Queue()
    q_therm = Queue()
    q_cmd = Queue()
    stop_evt = multiprocessing.Event()

    p_worker = Process(target=worker, args=(stop_evt, q_cmd,), daemon= True) 
    p_commander = Process(target=Commander, args=(stop_evt, q_pres, q_vib, q_therm, q_cmd,), daemon= True) 
    p_vib = Process(target=dsp_vib, args=(stop_evt, q_audio_vib, q_vib,), daemon=True)
    p_therm = Process(target=dsp_therm, args=(stop_evt, q_audio_therm, q_therm,), daemon=True)
    p_audiocapture = Process(target=AudioCapture, args=(stop_evt, q_audio_playback, q_audio_vib, q_audio_therm), daemon=True)
    p_audioplayback = Process(target=AudioPlayback, args=(stop_evt, q_audio_playback), daemon=True)
    p_serial = Process(target=read_from_serial, args=(stop_evt, q_pres, port, baud,), daemon=True)
    
    

    p_worker.start()
    p_commander.start()
    p_vib.start()
    p_therm.start()
    p_audiocapture.start()
    p_audioplayback.start()
    p_serial.start()

    # workaround: because there's 5 mysterious data in q_pres, we clean them all first
    # time.sleep(1)
    # while not q_pres.empty():
    #     q_pres.get_nowait()
    # while not q_vib.empty():
    #     q_vib.get_nowait()
    # while not q_therm.empty():
    #     q_therm.get_nowait()
    # while not q_cmd.empty():
    #     q_cmd.get_nowait()

    print("Press 'q' then Enter to quit.")
    try:
        for line in sys.stdin:
            if line.strip().lower() == "q":
                print("Quit requested.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        p_serial.join(timeout=2)
        p_commander.join()
        p_worker.join()
        p_audiocapture.join()
        p_audioplayback.join()
        p_vib.join()
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
