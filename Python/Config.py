from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class Config:
    INPUT_DEVICE_INDEX = 0
    ARDUINO_CLK = 70

    # LEFT_MIC_INDEX = 1
    # RIGHT_MIC_INDEX = 3

    OUTPUT_DEVICE_INDEX = 1
    SAMPLERATE = 16000
    AUDIO_CHUNK_MS = 70

    AUDIO_PLAYBACK_DELAY_S = 1
    VIBRATION_DELAY_S = 1

    VIB_OUT_SCALE = 1.2
    HVIB_OUT_SCALE = 1

    HEAT_TIME = 3
    HAND_HEAT_TIME = 3

    THERM_OUT_SCALE = 1
    HOLD_COUNT = 3

    RECORD_OUT_PATH = f"Record/audio_recording_{timestamp}.wav"
    # RECORD_PATH = "./Record/mock.wav"
    RECORD_PATH = "./Record/asmrfeelfinal2.wav"
    RECORD = False
    PLAY_RECORD = True
    # MIMIC_STEREO = False

    STREAMING = False # USE mic
