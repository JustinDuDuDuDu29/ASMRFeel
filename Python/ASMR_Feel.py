import multiprocessing
import threading
from datafeel.device import Dot, VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
from multiprocessing import Pipe, Process, Queue

from AudioProcess.AudioHandler import AudioHandler
from DataProcess.Audio_Data_Process import VoiceToThermalData, VoiceToVibrateData
from DataFeelProcess.Vibration_Thermal_DataFeel import VibrateDataFeel, ThermalDataFeel

def InitDots() -> list[Dot]:
    return discover_devices(4)


if __name__ == "__main__":
    print("Starting ASMRFeel !")
    q_Audio = Queue()
    parent_conn, child_conn = Pipe()

    vibrate_queue = multiprocessing.Queue(maxsize=1)
    thermal_queue = multiprocessing.Queue(maxsize=1)


    AudioHandler.ListDevice()

    audioHandler = AudioHandler(q_Audio)
    audioHandler1 = AudioHandler(q_Audio)

    # device = InitDots()[0]

    stopReadAudioProcessEvent = multiprocessing.Event()
    ReadAudioProcess = Process(target=audioHandler.RecordAudio, args=(stopReadAudioProcessEvent, ), daemon=True)
    ReadAudioProcess.start()

    # stop_evt = threading.Event()

    # VoiceToVibrateDataProcess = Process(target=VoiceToVibrateData, args=(stopReadAudioProcessEvent, q_Audio, parent_conn, vibrate_queue), daemon=True)
    # VoiceToVibrateDataProcess.start()

    VoiceToThermalDataProcess = Process(target=VoiceToThermalData, args=(stopReadAudioProcessEvent, q_Audio, parent_conn, thermal_queue, "", 16000, 512, 128, 128, 2e-5, 0.15), daemon=True)
    VoiceToThermalDataProcess.start()
    
    # PlayAudioProcess = Process(target=audioHandler1.PlayAudio, args=(stopReadAudioProcessEvent, child_conn, ), daemon=True)
    # PlayAudioProcess.start()

    # VibrateDataFeelThread = threading.Thread(target=VibrateDataFeel, args=(device, vibrate_queue, stop_evt))
    # VibrateDataFeelProcess = Process(target=VibrateDataFeel, args=(vibrate_queue,), daemon=True)
    # VibrateDataFeelThread.start()

    # ThermalDataFeelThread = threading.Thread(target=ThermalDataFeel,  args=(device, thermal_queue, stop_evt))
    # ThermalDataFeelProcess = Process(target=ThermalDataFeel, args=(thermal_queue,), daemon=True)
    # ThermalDataFeelThread.start()
    

    print("Press q to  exit!")
    while True:
        k = input()
        k = k.lower().strip()
        if(k == "q"):
            print("Exiting...")
            stopReadAudioProcessEvent.set()
            ReadAudioProcess.join()
            # VoiceToVibrateDataProcess.join()
            VoiceToThermalDataProcess.join()
            # PlayAudioProcess.join()
            # stop_evt.set()
            # VibrateDataFeelThread.join()
            # ThermalDataFeelThread.join()
            break
    print("ASMRFeel done.")

