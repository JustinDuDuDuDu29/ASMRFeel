from Config import Config
import math
import numpy as np 
import wave

def PlayRecord():
    print("starting PlayRecord process!")
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

    # Interpret raw data as interleaved samples
    all_channels = np.frombuffer(raw, dtype=dtype)
    all_channels = all_channels.reshape(-1, n_channels)

    print(all_channels.T.shape)
    wave_time = all_channels.shape[0] / fr
    print(wave_time)

    # create the press array, sent each element in the array every Config.ARDUINO_CLK to the serial queue to simulate the real world arduino 
    PressARR = []
    for i in range(math.floor(wave_time * 1000 / Config.ARDUINO_CLK)):
        PressARR.append(all_channels[math.floor(i * Config.ARDUINO_CLK * fr / 1000)])

    



PlayRecord()
    
