from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
import pyaudio
import time
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config
import queue as pyqueue


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