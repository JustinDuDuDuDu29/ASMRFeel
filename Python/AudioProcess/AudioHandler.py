import math
from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from typing import Optional
import pyaudio
import time
import numpy as np

import sys
import os
import wave
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config
import queue as pyqueue

# -------------------- Audio Capture --------------------

def AudioCapture(stop_evt: Event, init_evt: Event, q_audio_playback: Queue, q_audio_vib: Queue, q_audio_therm: Queue, q_pres:Queue, sr=Config.SAMPLERATE, chunk_ms=Config.AUDIO_CHUNK_MS, vibra_delay=Config.VIBRATION_DELAY_S):
    """Continuously capture audio in 70 ms frames and put the latest into q_audio."""
    pa = pyaudio.PyAudio()
    # print all device
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']} | "
            f"host API: {pa.get_host_api_info_by_index(int(info['hostApi']))['name']} | "
            f"maxInputChannels: {info['maxInputChannels']}")
    framesize = int(Config.SAMPLERATE * chunk_ms / 1000)


    stream:Optional[pyaudio.Stream] = None
    raw:bytes = b''
    dtype = np.float32
    n_channels = 18
    all_channels:Optional[np.ndarray] = None


    if not Config.PLAY_RECORD:
        # if Config.MIMIC_STEREO:
        #     channel = 1
        # else:
        #     channel = 2
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=2,
                         rate=sr,
                         input=True,
                         input_device_index=Config.INPUT_DEVICE_INDEX,
                         frames_per_buffer=framesize)
    else:
        # use record here
        PATH = Config.RECORD_PATH
        with wave.open(PATH, "r") as w:
            n_channels = w.getnchannels()
            # print(w.readframes(w.getnframes()))
            fr = w.getframerate()
            sampwidth = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sampwidth == 1:
            dtype = np.uint8   # 8-bit PCM
        elif sampwidth == 2:
            dtype = np.float16# 16-bit PCM
        elif sampwidth == 4:
            dtype = np.float32# 32-bit PCM
        else:
            raise ValueError("Unsupported sample width")

        all_channels = np.frombuffer(raw, dtype=np.int16)
        all_channels = all_channels.reshape(-1, n_channels)
        all_channels = all_channels.T
        all_channels = all_channels.astype(np.float32) / 32768.0

        wave_time = all_channels.T.shape[0] / fr
        lens = math.floor(wave_time * 1000 / Config.ARDUINO_CLK)
        print(wave_time)
        p_play_recorded_Pres = Process(target=play_recorded_press, args=(q_pres, all_channels[2:], lens,), daemon=True)
        p_play_recorded_Pres.start()

        # all_channels = np.frombuffer(raw, dtype=np.int16).copy()   # now writable
        # all_channels = all_channels.reshape(-1, n_channels)
        # all_channels = all_channels.T
        # all_channels[:2] = all_channels[:2].astype(np.float32) / 32768.0



    buffer = []
    start_time = time.time()

    index = 0
    lastTime = 0
    init_evt.wait()  # wait until AudioCapture is ready

    while not stop_evt.is_set():
        # print(framesize)
        arr:Optional[np.ndarray] = None
        if not Config.PLAY_RECORD:
            if stream is not None:
                data = stream.read(framesize, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.float32)
                l = arr[0::2]
                r = arr[1::2]
                arr = np.stack([l, r], axis=0)
                # print("streamsahpe", arr.shape)
            else:
                print("streaming is none! exiting the program")
                exit()
        else:
            if all_channels is None:
                print("all_channels is none! exiting the program")
                exit()

            if not index > math.ceil(all_channels.shape[-1] / framesize):
                while (time.monotonic() - lastTime) * 1000 < Config.AUDIO_CHUNK_MS: 
                    pass

                arr = all_channels[0:2, index * framesize : (index + 1) * framesize]
                index += 1
                lastTime = time.monotonic()

        # HERE
        # if Config.MIMIC_STEREO:
        #     if arr is not None:
        #         arr = np.stack([arr, arr], axis=0)
        try:
            q_audio_playback.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_playback queue is full!!!")

        if arr is not None:
            buffer.append(arr)

        if (time.time() - start_time) > vibra_delay:
            try:
                q_audio_vib.put_nowait(buffer.pop(0))
            except pyqueue.Full:
                print("q_audio_vib queue is full!!!")
        else:
            q_audio_vib.put_nowait(arr)


        try:
            q_audio_therm.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_therm queue is full!!!")

    if stream is not None:
        stream.stop_stream()
        stream.close()
    pa.terminate()

# -------------------- Audio Playback --------------------

def AudioPlayback(stop_evt: Event, q_audio_playback: Queue, playback_delay=Config.AUDIO_PLAYBACK_DELAY_S, sr=Config.SAMPLERATE):
    """Play audio from q_audio with a delay (default 300ms)."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=2,
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
            # play stereo audio Transpose stereo data from (2, framesize) to (framesize, 2) 
            data = buffer.pop(0).T.tobytes()
            stream.write(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

# -------------------- Audio Dual mic capture--------------------
def AudioCaptureDualMics(stop_evt: Event,
                        init_evt: Event,
                         q_audio_playback: Queue,
                         q_audio_vib: Queue,
                         q_audio_therm: Queue,
                         sr=Config.SAMPLERATE,
                         chunk_ms=Config.AUDIO_CHUNK_MS,
                         vibra_delay=Config.VIBRATION_DELAY_S):

    pa = pyaudio.PyAudio()
    # List all devices so you can pick indexes
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']} | maxInputChannels={info['maxInputChannels']}")

    # Replace these with the actual indices for left/right microphones
    LEFT_MIC_INDEX  = Config.LEFT_MIC_INDEX
    RIGHT_MIC_INDEX = Config.RIGHT_MIC_INDEX

    framesize = int(sr * chunk_ms / 1000)

    # open two independent 1-channel input streams
    left_stream = pa.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=sr,
                          input=True,
                          input_device_index=LEFT_MIC_INDEX,
                          frames_per_buffer=framesize)
    right_stream = pa.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=sr,
                           input=True,
                           input_device_index=RIGHT_MIC_INDEX,
                           frames_per_buffer=framesize)

    buffer = []
    start_time = time.time()

    init_evt.wait()  # wait until AudioCapture is ready


    while not stop_evt.is_set():
        # --- read from both mics ---
        left_data  = left_stream.read(framesize, exception_on_overflow=False)
        right_data = right_stream.read(framesize, exception_on_overflow=False)

        left_arr  = np.frombuffer(left_data,  dtype=np.float32)
        right_arr = np.frombuffer(right_data, dtype=np.float32)

        # stack to shape (2, framesize): [0]=left, [1]=right
        stereo_arr = np.stack([left_arr, right_arr], axis=0)

        # feed queues
        try:
            q_audio_playback.put_nowait(stereo_arr)
        except pyqueue.Full:
            print("q_audio_playback full!")

        buffer.append(stereo_arr)

        if (time.time() - start_time) > vibra_delay:
            try:
                q_audio_vib.put_nowait(buffer.pop(0))
            except pyqueue.Full:
                print("q_audio_vib full!")
        else:
            q_audio_vib.put_nowait(stereo_arr)

        try:
            q_audio_therm.put_nowait(stereo_arr)
        except pyqueue.Full:
            print("q_audio_therm full!")


def play_recorded_press(q_pres: Queue, pres_Channel: np.ndarray, lens:int ):

    print("play recorded press!")
    
    pressIndex = 0
    lastPress = 0
    start_time = time.time()

    while time.time() - start_time < Config.AUDIO_PLAYBACK_DELAY_S:
        pass

    while pressIndex < lens:
        while 1000 * (time.monotonic() - lastPress) > Config.ARDUINO_CLK: 
            lastPress = time.monotonic()
            # print(pres_Channel[:, math.floor(pressIndex * 16000 * 70 / 1000)])
            press = pres_Channel[:, math.floor(pressIndex * 16000 * 70 / 1000)] * 1023.0
            print(press)
            pdL = press[0:8]
            pdR = press[8:]
            row_a = ",".join(str(int(x)) for x in pdR)   # right half
            row_b = ",".join(str(int(x)) for x in pdL)   # left half

            pressIndex += 1

            

            q_pres.put(f"1;{row_a};{row_b}")
            
