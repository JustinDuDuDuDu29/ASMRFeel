import multiprocessing
import sys, time, threading
import serial
from serial.tools import list_ports
from time import sleep
from multiprocessing.synchronize import Event
from multiprocessing import Pipe, Process, Queue
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms

devices = discover_devices(4)

device = devices[0]

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

def pressureLEDxxx(stop_evt: Event, q:Queue):
    
    device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    while not stop_evt.is_set():
        # if(q.empty()):
        #     device.set_led_off()

        d = q.get()
        print(d)
        d = d.split(",")

        for i, val in enumerate(d):
            # print(f"{i}: {val}")
            if int(val)>100:
                device.registers.set_individual_led(i, 255, 0, 0)
            else:
                device.registers.set_individual_led(i, 0, 0, 0)
            # sleep(.1)



def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
    """Line-framed reader. Reconnects on failure."""

    device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    while not stop_evt.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                # Optional: set DTR/RTS here if your device needs it
                ser.reset_input_buffer()
                while not stop_evt.is_set():
                    try:
                        line = ser.readline()  # b'' on timeout
                        if line:
                            d = line.rstrip(b"\r\n").decode("utf-8")
                            q.put_nowait(d)

                            d = q.get()
                            print(ser.in_waiting);
                            print(f"d: {d}")

                            d = d.split(",")

                            for i, val in enumerate(d):
                                # print(1)
                                if int(val)>100:
                                    device.set_led( 255, 0, 0, i)
                                else:
                                    device.registers.set_individual_led( 0, 0, 0,i)
                    except serial.SerialException as e:
                        print(f"[serial] read error: {e}; will reconnect")
                        break  # break inner loop → outer reconnects
        except serial.SerialException as e:
            print(f"[serial] open error on {port}: {e}; retrying...")
        # small backoff before reconnect
        stop_evt.wait(1.0)

def main():
    baud = 115200
    port = choose_port()
    print(f"Starting connection at {port} {baud}…")

    q = Queue(12)
    stop_evt = multiprocessing.Event()
    t = threading.Thread(target=read_from_serial, args=(stop_evt, q, port, baud), daemon=True)
    t.start()
    # p =Process(target=pressureLED, args=(stop_evt, q), daemon= True) 
    # p.start()
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
        # p.join()
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()
