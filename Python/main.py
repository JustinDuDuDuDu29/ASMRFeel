import multiprocessing
import sys
from serial.tools import list_ports
from multiprocessing import  Process, Queue

from Config import Config
from AudioProcess.AudioHandler import AudioCapture, AudioPlayback
from DataFeelProcess.DFHandler import Worker, Commander
from DataProcess.DataHandler import dsp_therm, dsp_vib
from SerialProcess.SerialHandler import read_from_serial
from DataFeelCenter import DataFeelCenter, token
from SocketToUnity import SocketToUnity
from RecordProcess.Recorder import Recorder, RecorderAudioOnly

import time

def choose_port(default=None):
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    prompt = f"Select port index [{0 if default is None else default}]: "
    try:
        idx = input(prompt).strip()
        idx = int(idx) if idx else (0 if default is None else default)
    except ValueError:
        idx = 0
    return ports[idx].device


def main():
    baud = 115200
    port = choose_port()
    print(f"Starting connection at {port} {baud}…")

    q_audio_playback = Queue()
    q_audio_vib = Queue()
    q_audio_therm = Queue()

    q_pres = Queue()
    q_pres_record = Queue()
    q_vib = Queue()
    q_therm = Queue()
    q_cmd = Queue()
    q_unity = Queue()
    # q_wav = Queue()
    stop_evt = multiprocessing.Event()
    init_evt = multiprocessing.Event()

    p_worker = Process(target=Worker, args=(stop_evt, q_cmd,), daemon= True) 
    p_commander = Process(target=Commander, args=(stop_evt, q_pres, q_vib, q_therm, q_cmd, q_unity,), daemon= True) 
    p_vib = Process(target=dsp_vib, args=(stop_evt, q_audio_vib, q_vib, init_evt), daemon=True)
    p_therm = Process(target=dsp_therm, args=(stop_evt, q_audio_therm, q_therm,), daemon=True)
    # p_audiocapture = Process(target=AudioCapture, args=(stop_evt, q_audio_playback, q_audio_vib, q_audio_therm), daemon=True)
    p_audiocapture = Process(target=AudioCapture, args=(stop_evt, init_evt, q_audio_playback, q_audio_vib, q_audio_therm, q_pres, ), daemon=False)
    p_audioplayback = Process(target=AudioPlayback, args=(stop_evt, q_audio_playback), daemon=True)

    if Config.RECORD:
        p_recorder = Process(target=Recorder, args=(stop_evt, q_pres_record,), daemon=True)
    if not Config.PLAY_RECORD:
        p_serial = Process(target=read_from_serial, args=(stop_evt, q_pres, q_pres_record, port, baud,), daemon=True)
    # p_socket = Process(target=SocketToUnity, args=(stop_evt, q_unity, 1688, ), daemon=True)
    # p_wsocket = Process(target=SocketToUnity, args=(stop_evt, q_wav, 1689, ), daemon=True)
    
    

    p_worker.start()
    p_commander.start()
    p_vib.start()
    p_therm.start()
    p_audiocapture.start()
    p_audioplayback.start()
    # p_serial.start()
    if Config.RECORD:
        p_recorder.start()
    if not Config.PLAY_RECORD:
        p_serial.start()
    # p_socket.start()
    # p_wsocket.start()

    # workaround: because there's 5 mysterious data in q_pres, we clean them all first
    time.sleep(4)
    while not q_pres.empty():
        q_pres.get_nowait()
    while not q_vib.empty():
        q_vib.get_nowait()
    while not q_therm.empty():
        q_therm.get_nowait()
    while not q_cmd.empty():
        q_cmd.get_nowait()
    while not q_unity.empty():
        q_unity.get_nowait()
    while not q_pres_record.empty():
        q_pres_record.get_nowait()
    # while not q_wav.empty():
    #     q_wav.get_nowait()

    print("Press 'q' then Enter to quit.")
    try:
        for line in sys.stdin:
            if line.strip().lower() == "q":
                print("Quit requested.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        
        stop_evt.set()
        p_serial.join(timeout=2)
        p_commander.join()
        p_worker.join()
        p_audiocapture.join()
        p_audioplayback.join()
        p_vib.join()
        p_therm.join()
        if not Config.RECORD:
            p_recorder.join()
        # p_socket.join()
        # p_wsocket.join()
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
