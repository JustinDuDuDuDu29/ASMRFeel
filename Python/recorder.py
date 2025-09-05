# pip install pyaudio soundfile pyserial numpy
import time
import threading
from collections import deque
import sys
from serial.tools import list_ports
import numpy as np
import pyaudio
from DataFeelCenter import token
from DataFeelProcess.DFHandler import Worker, Commander
import soundfile as sf
import serial
from multiprocessing.synchronize import Event
from multiprocessing import  Event, Process, Queue
import multiprocessing
# ===================== Config =====================
SAMPLE_RATE      = 44100
BLOCK_FRAMES     = 1024
DURATION_SEC     = 10

# Set your two mono mic device indices
DEVICE_LEFT      = 0
DEVICE_RIGHT     = 2

SER_PORT         = "COM5"       # e.g. "/dev/ttyUSB0"
SER_BAUD         = 115200
SER_SLEEP_S      = 0.002        # small yield for serial thread

WAV_OUT          = "mics_plus_serial.wav"
TOTAL_CHANNELS   = 10           # 2 mics + 8 serial channels (ch3..ch10)

# Scale raw serial ints (e.g., 0..1023) to audio range [-1, 1] or [0, 1]
SERIAL_SCALE     = 1.0 / 1023.0 # set to 1.0 to write raw ints as floats
CENTER_AT_ZERO   = False        # True -> map to [-1, 1] instead of [0, 1]

# How to fill between serial rows within a block: "hold" or "linear"
INTERP_MODE      = "hold"
BLOCK_SIZE       = 1024 * 8
# ==================================================
def play():
    
    q_cmd = Queue()
    stop_evt = multiprocessing.Event()
    play_= Process(target=playAudio, args=(stop_evt, q_cmd,), daemon= True)
    p_worker = Process(target=Worker, args=(stop_evt, q_cmd,), daemon= True)
    play_.start()
    p_worker.start()

    play_.join()


def playAudio(stop_evt: Queue, q_cmd: Queue):
    # ser_q = queue.Queue()
    # stop_evt = threading.Event()
    # th = threading.Thread(target=serial_reader, args=( ser_q, stop_evt), daemon=True)
    with sf.SoundFile(WAV_OUT, 'r') as f: 
        sr = f.samplerate
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paFloat32, channels=2, rate=sr, output=True)
        i = 0
        try:
            while True:
                block = f.read(BLOCK_SIZE, dtype='float32', always_2d=True)
                if len(block) == 0:
                    break
                stereo = block[:, :2]
                pd = block[:, 3:]
                
                stream.write(stereo.tobytes())
                for e in pd[0]:
                    if e * 1023 > 60:
                        print(e)
                        t0 = token(superDotID=0, therDiff=0, therIntensity=0, vibFrequency=100, vibIntensity=e, ledList=[[0,0,0]]*8)
                        q_cmd.put_nowait(("useToken", (t0, )))
                        break
                else:
                    t0 = token(superDotID=0, therDiff=0, therIntensity=0, vibFrequency=100, vibIntensity=0, ledList=[[0,0,0]]*8)
                    q_cmd.put_nowait(("useToken", (t0, )))

        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

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

def list_audio_devices():
    pa = pyaudio.PyAudio()
    print("\n=== Audio Devices ===")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info.get('name')}")
    pa.terminate()


def serial_reader(ser, fifo: deque, stop_evt: threading.Event, remainder: bytearray):
    """
    Continuously read_all() from serial, parse comma-separated ints,
    and append (timestamp_seconds, np.float32[8]) into fifo (FIFO order).
    """
    while not stop_evt.is_set():
        try:
            data = ser.read_all()
            if data:
                remainder.extend(data)
                *lines, rem = remainder.split(b'\n')
                remainder[:] = rem  # keep partial tail

                for ln in lines:
                    ln = ln.rstrip(b'\r')
                    if not ln:
                        continue
                    try:
                        parts = [p for p in ln.decode("utf-8").split(',') if p]
                        vals_i = [int(p) for p in parts]
                        if len(vals_i) >= 8:
                            ts = time.perf_counter()  # seconds (high-res, monotonic)
                            fifo.append((ts, np.asarray(vals_i[:8], dtype=np.float32)))

                    except Exception:
                        # bad line -> skip
                        continue
            else:
                time.sleep(SER_SLEEP_S)
        except Exception:
            time.sleep(0.01)


def fill_serial_block(
    ser_fifo: deque,
    n: int,
    block_t0: float,
    Ts: float,
    last_vals: np.ndarray,
    mode: str = "hold"
):
    """
    Build (n, 8) serial block from queued rows with timestamps.
    - Pops any rows with ts <= block_t1 as needed.
    - Returns (ser_block, updated_last_vals).
    """
    block_t1 = block_t0 + n * Ts
    ser_block = np.empty((n, 8), dtype=np.float32)

    # 1) Consume and fold in any serial rows that happened BEFORE this block,
    #    so we start with the latest value as of block_t0.
    while ser_fifo and ser_fifo[0][0] <= block_t0:
        _, last_vals = ser_fifo.popleft()

    # 2) Start with last known values (sample & hold baseline).
    print(last_vals)
    ser_block[:] = last_vals

    if mode not in ("hold", "linear"):
        mode = "hold"

    # 3) Now place rows that land inside this block.
    #    Process in time order while ts < block_t1.
    prev_t = block_t0
    prev_v = last_vals

    while ser_fifo and ser_fifo[0][0] < block_t1:
        ts, v = ser_fifo.popleft()
        # segment from prev_t -> ts maps to indices [start:end)
        start = max(0, int((prev_t - block_t0) / Ts))
        end   = min(n, int((ts     - block_t0) / Ts))

        if end > start:
            if mode == "linear":
                # linear interp from prev_v to v over [start, end)
                w = np.linspace(0.0, 1.0, end - start, endpoint=False, dtype=np.float32)
                ser_block[start:end] = prev_v + (v - prev_v) * w[:, None]
            else:
                # hold prev value
                ser_block[start:end] = prev_v

        prev_t, prev_v = ts, v

    # 4) Fill the tail [last_change : n) with prev_v
    last_idx = max(0, int((prev_t - block_t0) / Ts))
    if last_idx < n:
        ser_block[last_idx:] = prev_v

    return ser_block, prev_v  # updated last_vals = prev_v


def record():
    # list_audio_devices()  # uncomment to see device indices

    # --- Serial setup ---
    ser = serial.Serial(choose_port())  # non-blocking
    ser_fifo = deque()            # holds (ts_seconds, np.float32[8]) in order
    remainder = bytearray()
    stop_evt = threading.Event()
    th = threading.Thread(target=serial_reader,
                          args=(ser, ser_fifo, stop_evt, remainder),
                          daemon=True)
    th.start()

    # --- Audio setup ---
    pa = pyaudio.PyAudio()
    stream_left = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                          input=True, input_device_index=DEVICE_LEFT,
                          frames_per_buffer=BLOCK_FRAMES)
    stream_right = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                           input=True, input_device_index=DEVICE_RIGHT,
                           frames_per_buffer=BLOCK_FRAMES)

    wav = sf.SoundFile(WAV_OUT, mode="w", samplerate=SAMPLE_RATE,
                       channels=TOTAL_CHANNELS, subtype="PCM_16")

    Ts = 1.0 / SAMPLE_RATE
    blocks = int(np.ceil(SAMPLE_RATE / BLOCK_FRAMES * DURATION_SEC))
    last_vals = np.zeros(8, dtype=np.float32)  # used before first serial row arrives

    print(f"Recording {DURATION_SEC}s to {WAV_OUT} ({TOTAL_CHANNELS} ch)...")
    try:
        for _ in range(blocks):
            # 1) read mic audio
            t0 = time.perf_counter()  # approximate time of the FIRST frame in this block
            left  = np.frombuffer(stream_left.read(BLOCK_FRAMES,  exception_on_overflow=False), dtype=np.float32)
            right = np.frombuffer(stream_right.read(BLOCK_FRAMES, exception_on_overflow=False), dtype=np.float32)
            n = min(len(left), len(right), BLOCK_FRAMES)
            left, right = left[:n], right[:n]

            # 2) build serial block aligned by timestamps
            ser_block, last_vals = fill_serial_block(
                ser_fifo, n, t0, Ts, last_vals, mode=INTERP_MODE
            )

            # 3) scale / center serial values to audio range
            if CENTER_AT_ZERO:
                ser_block = (ser_block * SERIAL_SCALE) * 2.0 - 1.0   # map to [-1, 1]
            else:
                ser_block = (ser_block * SERIAL_SCALE)               # map to [0, 1]

            # 4) combine and write -> shape (n, 10): [L, R, ch3..ch10]
            frame = np.column_stack([left, right, ser_block])
            wav.write(frame)

    finally:
        wav.close()
        stream_left.close()
        stream_right.close()
        pa.terminate()

        stop_evt.set()
        th.join(timeout=0.5)
        ser.close()

    print("Done.")


if __name__ == "__main__":
    play()
    # record()

