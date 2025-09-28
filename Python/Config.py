from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class Config:
    INPUT_DEVICE_INDEX = 5
    ARDUINO_CLK = 70

    LEFT_MIC_INDEX = 3
    RIGHT_MIC_INDEX = 7

    OUTPUT_DEVICE_INDEX = 9
    SAMPLERATE = 16000
    AUDIO_CHUNK_MS = 70

    AUDIO_PLAYBACK_DELAY_S = 2
    VIBRATION_DELAY_S = 2

    VIB_OUT_SCALE = 1
    HVIB_OUT_SCALE = 1

    HEAT_TIME = 1.5
    THERM_OUT_SCALE = 1


    RECORD_OUT_PATH = f"Record/audio_recording_{timestamp}.wav"
    RECORD_PATH = "./Record/audio_recording_20250929_030223.wav"
    RECORD = False
    PLAY_RECORD = True
    # MIMIC_STEREO = False
