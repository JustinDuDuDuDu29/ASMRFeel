from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
import pyaudio
from typing import Optional
import numpy as np

class AudioHandler(object):
    """即時錄音與播放，把麥克風輸入分割並儲存。可接收從音訊資料並送到指定輸出裝置播放。可提供列出裝置清單的功能。"""
    def __init__(self,q_Audio: Queue, channels: int = 1, sr: int = 16000, window_ms: int = 200):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = channels 
        self.SR= sr
        self.STREAM:Optional[pyaudio.Stream] = None
        self.TIMELAP = window_ms / 1000.0
        self.Q_AUDIO = q_Audio

    @staticmethod
    def ListDevice():
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                di = p.get_device_info_by_index(i)
                max_in  = int(di.get('maxInputChannels')  or 0)
                max_out = int(di.get('maxOutputChannels') or 0)
                name    = str(di.get('name') or '')
                print(f"[{i:2}] {name}  in:{max_in} out:{max_out}")
        finally:
            p.terminate()

    def RecordAudio(self, ev:Event,frames_per_buffer=1024):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.SR,
                             input=True,
                             input_device_index=1,
                             frames_per_buffer=frames_per_buffer,)
        try: 
            bytes_per_frame = 4 * self.CHANNELS  # float32
            target_frames   = int(self.SR * self.TIMELAP)
            target_bytes    = target_frames * bytes_per_frame
            buf = bytearray()
            while (self.stream is not None and self.stream.is_active() and not ev.is_set()): 
                chunk = self.stream.read(frames_per_buffer, exception_on_overflow=False)
                buf.extend(chunk)
                # time.sleep(self.TIMELAP)
                while len(buf) >= target_bytes:
                    window_bytes = bytes(buf[:target_bytes])
                    del buf[:target_bytes]
                    samples = np.frombuffer(window_bytes, dtype=np.float32)
                    self.Q_AUDIO.put(samples.copy())  

        finally:
            try:
                if self.stream is not None and self.stream.is_active():
                    self.stream.stop_stream()
            finally:
                if self.stream is not None:
                    self.stream.close()
                p.terminate()
                print("finally, the AudioHandler has stopped...")
    
    def PlayAudio(self, ev:Event, child_conn: Connection):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32,
                        channels=self.CHANNELS,
                        rate=self.SR,
                        output_device_index=6,
                        input=False,
                        output=True)
        try:
        # while not q_Audio.empty():
            while not ev.is_set():
                if child_conn.poll(0.05):
                    arr = child_conn.recv()
                    self.stream.write(arr.astype(np.float32, copy=False).tobytes())
        finally:
            try:
                if self.stream is not None and self.stream.is_active():
                    self.stream.stop_stream()
            finally:
                if self.stream is not None:
                    self.stream.close()
                    self.stream = None
                if p is not None:
                    p.terminate()
                print("finally, the AudioHandler has stopped...")

    def callback(self, in_data, frame_count, time_info, flag):
        return (in_data, pyaudio.paContinue)