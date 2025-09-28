from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
import pyaudio
import time
import numpy as np
import wave
import struct
from datetime import datetime

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config
import queue as pyqueue

def Recorder(stop_evt: Event, q_pres: Queue):
    pa = pyaudio.PyAudio()

    framesize = int(Config.SAMPLERATE * Config.AUDIO_CHUNK_MS / 1000)
    current_pres_data = np.zeros(16, dtype=np.float32)

    # Initialize single stereo stream
    stereo_stream = pa.open(format=pyaudio.paInt16,
                           channels=2,
                           rate=Config.SAMPLERATE,
                           input=True,
                           input_device_index=Config.INPUT_DEVICE_INDEX,
                           frames_per_buffer=framesize)
    
    # Setup WAV file for 2 channels (stereo audio only)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"Record/audio_recording_{timestamp}.wav"
    
    # WAV file parameters
    channels = 18  # Left and right audio only
    sample_width = 2  # 16-bit PCM (2 bytes)
    
    wav_file = wave.open(wav_filename, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(Config.SAMPLERATE)
    
    print(f"Audio recording started: {wav_filename}")
    print(f"Channels: {channels}, Sample rate: {Config.SAMPLERATE} Hz, Frame size: {framesize}")

    try:
        while not stop_evt.is_set():
            # Read stereo audio data
            stereo_data = stereo_stream.read(framesize, exception_on_overflow=False)
            stereo_arr = np.frombuffer(stereo_data, dtype=np.int16)
            
            # Separate left and right channels from interleaved stereo data
            left_arr = stereo_arr[0::2]  # Even indices (0, 2, 4, ...)
            right_arr = stereo_arr[1::2]  # Odd indices (1, 3, 5, ...)

            # Try to get new pressure data from queue (non-blocking)
            try:
                # Get all available pressure data to use the latest
                new_pres_data = q_pres.get_nowait()
                # print(new_pres_data)
                _, pres1, pres2 = new_pres_data.split(";")
                p = pres1.split(",")
                p1 = pres2.split(",")

                # combine p and p1
                current_pres_data = np.array(p + p1, dtype=np.float32)

                # map int(val) from 0-1023 to 0-1, then to int16 range
                current_pres_data = current_pres_data / 1023.0
                current_pres_data = (current_pres_data * 32767).astype(np.int16)

            except pyqueue.Empty:
                # No new pressure data available, use forward fill (keep current_pres_data)
                pass
            
            # Create 18-channel interleaved data
            # Each sample contains: [left, right] + 16 pressure values
            frame = []

            for i in range(framesize):
                # Add left and right audio samples (already int16)
                frame.extend([left_arr[i], right_arr[i]])
                # Add 16 pressure values (now int16)
                frame.extend(current_pres_data)

            # Convert to bytes and write to WAV file
            wav_data = np.array(frame, dtype=np.int16).tobytes()
            wav_file.writeframes(wav_data)
            
    except Exception as e:
        print(f"Audio recording error: {e}")
    finally:
        # Cleanup
        stereo_stream.stop_stream()
        stereo_stream.close()
        pa.terminate()
        wav_file.close()
        print(f"Audio recording stopped: {wav_filename}")


def RecorderAudioOnly(stop_evt: Event, q_pres: Queue):
    """
    Audio-only recorder that captures stereo audio (left and right channels)
    and saves to a standard 2-channel WAV file.
    """
    pa = pyaudio.PyAudio()

    framesize = int(Config.SAMPLERATE * Config.AUDIO_CHUNK_MS / 1000)
    
    # Initialize single stereo stream
    stereo_stream = pa.open(format=pyaudio.paInt16,
                           channels=2,
                           rate=Config.SAMPLERATE,
                           input=True,
                           input_device_index=Config.INPUT_DEVICE_INDEX,
                           frames_per_buffer=framesize)
    
    # Setup WAV file for 2 channels (stereo audio only)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"Record/audio_recording_{timestamp}.wav"
    
    # WAV file parameters
    channels = 2  # Left and right audio only
    sample_width = 2  # 16-bit PCM (2 bytes)
    
    wav_file = wave.open(wav_filename, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(Config.SAMPLERATE)
    
    print(f"Audio recording started: {wav_filename}")
    print(f"Channels: {channels}, Sample rate: {Config.SAMPLERATE} Hz, Frame size: {framesize}")

    try:
        while not stop_evt.is_set():
            # Read stereo audio data
            stereo_data = stereo_stream.read(framesize, exception_on_overflow=False)
            stereo_arr = np.frombuffer(stereo_data, dtype=np.int16)
            
            # Separate left and right channels from interleaved stereo data
            left_arr = stereo_arr[0::2]  # Even indices (0, 2, 4, ...)
            right_arr = stereo_arr[1::2]  # Odd indices (1, 3, 5, ...)

            # Try to get new pressure data from queue (non-blocking)
            try:
                # Get all available pressure data to use the latest
                new_pres_data = q_pres.get_nowait()
                if len(new_pres_data) == 16:
                    current_pres_data = np.array(new_pres_data, dtype=np.float32)
            except pyqueue.Empty:
                # No new pressure data available, use forward fill (keep current_pres_data)
                pass
            
            # Create 2-channel interleaved data
            # Each sample contains: [left, right]
            combined_data = []

            # Ensure both channels have the same length
            # min_len = min(len(left_arr), len(right_arr))
            # if min_len < framesize:
            #     left_arr = np.pad(left_arr, (0, framesize - len(left_arr)))
            #     right_arr = np.pad(right_arr, (0, framesize - len(right_arr)))

            for i in range(framesize):
                # Add left and right audio samples
                combined_data.extend([left_arr[i], right_arr[i]])
            
            # Convert to bytes and write to WAV file
            combined_array = np.array(combined_data, dtype=np.int16)
            wav_data = combined_array.tobytes()
            wav_file.writeframes(wav_data)
            
    except Exception as e:
        print(f"Audio recording error: {e}")
    finally:
        # Cleanup
        stereo_stream.stop_stream()
        stereo_stream.close()
        pa.terminate()
        wav_file.close()
        print(f"Audio recording stopped: {wav_filename}")
