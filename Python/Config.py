from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class Config:
    INPUT_DEVICE_INDEX = 12
    ARDUINO_CLK = 70

    # LEFT_MIC_INDEX = 1
    # RIGHT_MIC_INDEX = 3

    OUTPUT_DEVICE_INDEX = 17
    SAMPLERATE = 16000
    AUDIO_CHUNK_MS = 70

    AUDIO_PLAYBACK_DELAY_S = 2
    VIBRATION_DELAY_S = 2

    VIB_OUT_SCALE = 1
    HVIB_OUT_SCALE = 1

    HEAT_TIME = 1.5
    THERM_OUT_SCALE = 1
    HOLD_COUNT = 3

    RECORD_OUT_PATH = f"Record/audio_recording_{timestamp}.wav"
    # RECORD_PATH = "./Record/mock.wav"
    RECORD_PATH = "./Record/audio_recording_20250929_161817.wav"
    RECORD = False
    PLAY_RECORD = False
    # MIMIC_STEREO = False
