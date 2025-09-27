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
    
    # Initialize audio streams
    left_stream = pa.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=Config.SAMPLERATE,
                          input=True,
                          input_device_index=Config.LEFT_MIC_INDEX,
                          frames_per_buffer=framesize)
    right_stream = pa.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=Config.SAMPLERATE,
                           input=True,
                           input_device_index=Config.RIGHT_MIC_INDEX,
                           frames_per_buffer=framesize)
    
    # Setup WAV file for 18 channels (2 audio + 16 pressure)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recording_{timestamp}.wav"
    
    # WAV file parameters
    channels = 18  # 2 audio + 16 pressure
    sample_width = 4  # 32-bit float (4 bytes)
    
    wav_file = wave.open(wav_filename, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(Config.SAMPLERATE)
    
    # Initialize pressure data array with zeros (forward fill buffer)
    current_pres_data = np.zeros(16, dtype=np.float32)
    
    print(f"Recording started: {wav_filename}")
    print(f"Channels: {channels}, Sample rate: {Config.SAMPLERATE} Hz, Frame size: {framesize}")

    try:
        while not stop_evt.is_set():
            # Read audio data
            left_data = left_stream.read(framesize, exception_on_overflow=False)
            right_data = right_stream.read(framesize, exception_on_overflow=False)
            left_arr = np.frombuffer(left_data, dtype=np.float32)
            right_arr = np.frombuffer(right_data, dtype=np.float32)
            
            # Try to get new pressure data from queue (non-blocking)
            try:
                # Get all available pressure data to use the latest
                new_pres_data = q_pres.get_nowait()
                if len(new_pres_data) == 16:
                    current_pres_data = np.array(new_pres_data, dtype=np.float32)
            except pyqueue.Empty:
                # No new pressure data available, use forward fill (keep current_pres_data)
                pass
            
            # Create 18-channel interleaved data
            # Each sample contains: [left, right, pres0, pres1, ..., pres15]
            combined_data = []
            
            for i in range(framesize):
                # Add left and right audio samples
                combined_data.extend([left_arr[i], right_arr[i]])
                # Add 16 pressure values (forward filled)
                combined_data.extend(current_pres_data)
            
            # Convert to bytes and write to WAV file
            combined_array = np.array(combined_data, dtype=np.float32)
            wav_data = combined_array.tobytes()
            wav_file.writeframes(wav_data)
            
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        # Cleanup
        left_stream.stop_stream()
        left_stream.close()
        right_stream.stop_stream()
        right_stream.close()
        pa.terminate()
        wav_file.close()
        print(f"Recording stopped: {wav_filename}")