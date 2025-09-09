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

    q_audio_playback = Queue(maxsize=1)
    q_audio_vib = Queue(maxsize=1)
    q_audio_therm = Queue(maxsize=1)

    q_pres = Queue()
    q_vib = Queue()
    q_therm = Queue()
    q_cmd = Queue()
    stop_evt = multiprocessing.Event()

    p_worker = Process(target=Worker, args=(stop_evt, q_cmd,), daemon= True) 
    p_commander = Process(target=Commander, args=(stop_evt, q_pres, q_vib, q_therm, q_cmd,), daemon= True) 
    p_vib = Process(target=dsp_vib, args=(stop_evt, q_audio_vib, q_vib,), daemon=True)
    p_therm = Process(target=dsp_therm, args=(stop_evt, q_audio_therm, q_therm,), daemon=True)
    p_audiocapture = Process(target=AudioCapture, args=(stop_evt, q_audio_playback, q_audio_vib, q_audio_therm), daemon=True)
    p_audioplayback = Process(target=AudioPlayback, args=(stop_evt, q_audio_playback), daemon=True)
    p_serial = Process(target=read_from_serial, args=(stop_evt, q_pres, port, baud,), daemon=True)
    
    

    p_worker.start()
    p_commander.start()
    p_vib.start()
    p_therm.start()
    p_audiocapture.start()
    p_audioplayback.start()
    p_serial.start()

    # workaround: because there's 5 mysterious data in q_pres, we clean them all first
    time.sleep(3)
    while not q_pres.empty():
        q_pres.get_nowait()
    while not q_vib.empty():
        q_vib.get_nowait()
    while not q_therm.empty():
        q_therm.get_nowait()
    while not q_cmd.empty():
        q_cmd.get_nowait()

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
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
