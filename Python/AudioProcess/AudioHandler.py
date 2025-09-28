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

def AudioCapture(stop_evt: Event, q_audio_playback: Queue, q_audio_vib: Queue, q_audio_therm: Queue, sr=Config.SAMPLERATE, chunk_ms=Config.AUDIO_CHUNK_MS, vibra_delay=Config.VIBRATION_DELAY_S):
    """Continuously capture audio in 70 ms frames and put the latest into q_audio."""
    pa = pyaudio.PyAudio()
    # print all device
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(f"{i}: {info['name']} | "
            f"host API: {pa.get_host_api_info_by_index(int(info['hostApi']))['name']} | "
            f"maxInputChannels: {info['maxInputChannels']}")
    framesize = int(Config.SAMPLERATE * chunk_ms / 1000)


    if Config.MIMIC_STEREO:
        channel = 1
    else:
        channel = 2
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=channel,
                     rate=sr,
                     input=True,
                     input_device_index=Config.INPUT_DEVICE_INDEX,
                     frames_per_buffer=framesize)

    buffer = []
    start_time = time.time()

    while not stop_evt.is_set():
        data = stream.read(framesize, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.float32)
        if Config.MIMIC_STEREO:
            arr = np.stack([arr, arr], axis=0)
        try:
            q_audio_playback.put_nowait(arr)
        except pyqueue.Full:
            print("q_audio_playback queue is full!!!")

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

# # -------------------- Audio Dual mic capture--------------------
# def AudioCaptureDualMics(stop_evt: Event,
#                          q_audio_playback: Queue,
#                          q_audio_vib: Queue,
#                          q_audio_therm: Queue,
#                          sr=Config.SAMPLERATE,
#                          chunk_ms=Config.AUDIO_CHUNK_MS,
#                          vibra_delay=Config.VIBRATION_DELAY_S):

#     pa = pyaudio.PyAudio()
#     # List all devices so you can pick indexes
#     for i in range(pa.get_device_count()):
#         info = pa.get_device_info_by_index(i)
#         print(f"{i}: {info['name']} | maxInputChannels={info['maxInputChannels']}")

#     # Replace these with the actual indices for left/right microphones
#     LEFT_MIC_INDEX  = Config.LEFT_MIC_INDEX
#     RIGHT_MIC_INDEX = Config.RIGHT_MIC_INDEX

#     framesize = int(sr * chunk_ms / 1000)

#     # open two independent 1-channel input streams
#     left_stream = pa.open(format=pyaudio.paFloat32,
#                           channels=1,
#                           rate=sr,
#                           input=True,
#                           input_device_index=LEFT_MIC_INDEX,
#                           frames_per_buffer=framesize)
#     right_stream = pa.open(format=pyaudio.paFloat32,
#                            channels=1,
#                            rate=sr,
#                            input=True,
#                            input_device_index=RIGHT_MIC_INDEX,
#                            frames_per_buffer=framesize)

#     buffer = []
#     start_time = time.time()

#     while not stop_evt.is_set():
#         # --- read from both mics ---
#         left_data  = left_stream.read(framesize, exception_on_overflow=False)
#         right_data = right_stream.read(framesize, exception_on_overflow=False)

#         left_arr  = np.frombuffer(left_data,  dtype=np.float32)
#         right_arr = np.frombuffer(right_data, dtype=np.float32)

#         # stack to shape (2, framesize): [0]=left, [1]=right
#         stereo_arr = np.stack([left_arr, right_arr], axis=0)

#         # feed queues
#         try:
#             q_audio_playback.put_nowait(stereo_arr)
#         except pyqueue.Full:
#             print("q_audio_playback full!")

#         buffer.append(stereo_arr)

#         if (time.time() - start_time) > vibra_delay:
#             try:
#                 q_audio_vib.put_nowait(buffer.pop(0))
#             except pyqueue.Full:
#                 print("q_audio_vib full!")
#         else:
#             q_audio_vib.put_nowait(stereo_arr)

#         try:
#             q_audio_therm.put_nowait(stereo_arr)
#         except pyqueue.Full:
#             print("q_audio_therm full!")

#     left_stream.stop_stream()
#     left_stream.close()
#     right_stream.stop_stream()
#     right_stream.close()
#     pa.terminate()
