import multiprocessing
import sys, time, threading
import serial
from serial.tools import list_ports
from time import sleep
from multiprocessing.synchronize import Event
from multiprocessing import Pipe, Process, Queue
from DataFeelCenter import DataFeelCenter
from concurrent.futures import ThreadPoolExecutor
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms


def candy(dfc, method, args):
    getattr(dfc, method)(*args)

def worker(stop_evt:Event, q_cmd: Queue):
    dfc = DataFeelCenter(numOfDots=4)  # Owns devices
    # with ThreadPoolExecutor() as tpe:
    while not stop_evt.is_set():
        try:
            while not q_cmd.empty():
                cmd = q_cmd.get()
                if not cmd:
                    return
                method, args = cmd
                # tpe.submit(candy, dfc, method, args)

                getattr(dfc, method)(*args)

        except Exception as e:
            print(f"worker Err! {e}")
            stop_evt.set()
            break

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

def pressureLED(stop_evt: Event, q:Queue, q_cmd:Queue):
    
    # device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    while not stop_evt.is_set():
        d = q.get()
        d = d.split(",")

        for i, val in enumerate(d):
            if int(val)>60:
                # device.registers.set_individual_led(i, 255, 0, 0)
                q_cmd.put(("led_no_timing", (0, i, 255, 0, 0)))
                q_cmd.put(("led_no_timing", (1, i, 255, 0, 0)))
            else:
                q_cmd.put(("led_no_timing", (0, i, 0, 255, 0)))
                q_cmd.put(("led_no_timing", (1, i, 0, 255, 0)))

        print(f"size of Q is: {q_cmd.qsize()}")
                # device.registers.set_individual_led(i, 0, 0, 0)



def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
    """Line-framed reader. Reconnects on failure."""

    # device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    while not stop_evt.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                ser.reset_input_buffer()
                while not stop_evt.is_set():
                    try:
                        line = ser.readline()  # b'' on timeout

                        if line:
                            d = line.rstrip(b"\r\n").decode("utf-8")
                            q.put(d)
                    except serial.SerialException as e:
                        print(f"[serial] read error: {e}; will reconnect")
                        break  
        except serial.SerialException as e:
            print(f"[serial] open error on {port}: {e}; retrying...")
        stop_evt.wait(1.0)

def main():
    baud = 115200
    port = choose_port()
    print(f"Starting connection at {port} {baud}…")

    q = Queue()
    q_cmd = Queue()
    stop_evt = multiprocessing.Event()
    w = Process(target=worker, args=(stop_evt, q_cmd,), daemon= True) 
    w.start()

    t = Process(target=read_from_serial, args=(stop_evt, q, port, baud,), daemon=True)
    t.start()
    p =Process(target=pressureLED, args=(stop_evt, q, q_cmd,), daemon= True) 
    p.start()
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
        t.join(timeout=2)
        p.join()
        w.join()
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
