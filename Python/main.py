import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from typing import Optional
from datafeel import device
import numpy as np
import librosa
import scipy.signal as sig
import time
from time import sleep
from datafeel.device import Dot, VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
import pyaudio
from multiprocessing import Pipe, Process, Queue

class AudioHandler(object):
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
                        output_device_index=2,
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



def InitDots() -> list[Dot]:
    return discover_devices(4)
    
def VoiceToVibrateData(ev:Event, q_Audio:Queue, parent_conn:Connection, filePath: str = "", sr: int = 16000, fmin: int = 70, fmax: int = 400, frame_length:int = 2048, hop_length:int = 160, window_ms:int = 200, saveToCSV: bool = False, saveCSVPath: str = "./freq_amp.csv"):
    # y, sr = librosa.load(filePath, sr=sr)
    while not ev.is_set():
        y = q_Audio.get()
        if y.size < frame_length:
            continue
        # ---- Pitch (10 ms hop) ----
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length, center=True
        )
        f0_hz = np.nan_to_num(f0, nan=0.0)
        voiced = voiced_flag.astype(bool)

        # ---- Amplitude at 10 ms hop (RMS -> [0,1]) ----
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]  # shape matches f0_hz
        p95 = np.percentile(rms, 95) + 1e-12  # robust normalization
        amp_10ms = np.clip(rms / p95, 0.0, 1.0)

        # ---- Pool both to 200 ms (20 frames of 10 ms) ----
        step = window_ms // (1000 * hop_length // sr) # = 20 when hop=10 ms

        n_frames = min(len(f0_hz), len(amp_10ms))
        n_full = (n_frames // step) * step
        if n_full == 0:
            continue
        # freq
        f0_blocks = f0_hz[:n_full].reshape(-1, step)
        v_blocks  = voiced[:n_full].reshape(-1, step)

        def agg_block(f_block, v_block):
            return np.median(f_block[v_block]) if np.any(v_block) else 0.0

        f0_200ms = np.array([agg_block(fb, vb) for fb, vb in zip(f0_blocks, v_blocks)])
        if len(f0_200ms) >= 3:
            f0_200ms = sig.medfilt(f0_200ms, kernel_size=3)

        # amplitude
        amp_blocks = amp_10ms[:n_full].reshape(-1, step)
        amp_200ms = np.median(amp_blocks, axis=1)
        if len(amp_200ms) >= 3:
            amp_200ms = sig.medfilt(amp_200ms, kernel_size=3)

        print(f'{f0_200ms}, {amp_200ms}')
        parent_conn.send(y)
        
        # ---- Stack to 2D array: [freq_hz, amp_0_1] ----

        if saveToCSV:
            freq_amp_200ms = np.stack([f0_200ms, amp_200ms], axis=1)
        # Save (CSV)
            np.savetxt(saveCSVPath, freq_amp_200ms, delimiter=",",
                       comments="", fmt="%.6f,%.6f")


def VibrateDataFeel(devices: list[Dot]):
    device = devices[0]
    # frequency = [100] * 100

    frequency = []
    with open("./freq_amp_200ms.csv", 'r') as f:
        lines = f.readlines()
        for r in lines:
            r = r.strip().rstrip().split(',')
            frequency.append([int(float(r[0])),float(r[1])])

    t = 0
    device.registers.set_vibration_mode(VibrationMode.MANUAL)   
    a = input("Please input")
    tt = time.time()
    for freq in frequency:
        # print(freq)
        total = time.time()
        device.play_frequency(freq[0], freq[1])
        # if(freq[1] < 0.1):
        #     device.stop_vibration()
        #     pass
        # device.play_frequency(freq[0], freq[1])
        st = (time.time() - total)
        sleep(0.2-st)
        # sleep(0.2)

        # t+=st
        # print(0.1 - (time.time() - total))
        
    print(time.time() - tt)

    device.stop_vibration()

    sleep(1)





if __name__ == "__main__":
    print("Starting ASMRFeel !")

    q_Audio = Queue()
    parent_conn, child_conn = Pipe()

    AudioHandler.ListDevice()

    audioHandler = AudioHandler(q_Audio)
    audioHandler1 = AudioHandler(q_Audio)

    stopReadAudioProcessEvent = multiprocessing.Event()
    ReadAudioProcess = Process(target=audioHandler.RecordAudio, args=(stopReadAudioProcessEvent, ), daemon=True)
    ReadAudioProcess.start()

    # AudioToDotParamProcess = Process(target=VoiceToVibrateData, args=(q_Audio,), daemon=True) 
    VoiceToVibrateDataProcess = Process(target=VoiceToVibrateData, args=(stopReadAudioProcessEvent, q_Audio, parent_conn, ), daemon=True) 
    VoiceToVibrateDataProcess.start()
    
    PlayAudioProcess = Process(target=audioHandler1.PlayAudio, args=(stopReadAudioProcessEvent, child_conn, ), daemon=True) 
    PlayAudioProcess.start()

    
    print("Press q to  exit!")
    while True:
        k = input()
        k = k.lower().strip()
        if(k == "q"):
            print("Exiting...")
            stopReadAudioProcessEvent.set()
            ReadAudioProcess.join()
            VoiceToVibrateDataProcess.join()
            PlayAudioProcess.join()
            break
    # devices = InitDots()
    # VoiceToVibrateData("./file.wav")
    # VibrateDataFeel(devices)
    print("ASMRFeel done.")

