import multiprocessing
import sys, time, threading
import serial
from serial.tools import list_ports
from time import sleep
from multiprocessing.synchronize import Event
from multiprocessing import Pipe, Process, Queue
from DataFeelCenter import DataFeelCenter, token
from concurrent.futures import ThreadPoolExecutor
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
import random


def candy(dfc, method, args):
    getattr(dfc, method)(*args)



def worker(stop_evt:Event, q_cmd: Queue):
    dfc = DataFeelCenter(numOfDots=4)  # Owns devices
    
    while not stop_evt.is_set():
        try:
            while not q_cmd.empty():
                cmd = q_cmd.get_nowait()
                if not cmd:
                    return
                method, args = cmd

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
    
    ledFlag = False 
    ledFlag1 = False 
    vibFlag = False 
    # device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    while not stop_evt.is_set():
        d = q.get()
        d = d.split(",")

        led1 = []
        led2 = []
        led3 = []
        led4 = []


        t0 = token(superDotID = 0)
        t1 = token(superDotID = 1)
        t2 = token(superDotID = 2)
        t3 = token(superDotID = 3)

        
        for i, val in enumerate(d):
            if int(val)>60:
                vibFlag = False
                t0.vibFrequency = 100
                t0.vibIntensity = random.uniform(0.5, 1.0)
                t1.vibFrequency = 100
                t1.vibIntensity = random.uniform(0.5, 1.0)
                t2.vibFrequency = 100
                t2.vibIntensity = random.uniform(0.5, 1.0)
                t3.vibFrequency = 100
                t3.vibIntensity = random.uniform(0.5, 1.0)
                break;
        else:
            if not vibFlag:
                t0.vibFrequency = 0 
                t0.vibIntensity = 0
                t1.vibFrequency = 0
                t1.vibIntensity = 0
                t2.vibFrequency = 0
                t2.vibIntensity = 0
                t3.vibFrequency = 0
                t3.vibIntensity = 0
                vibFlag = True


        # LED Example
        for i, val in enumerate(d):
            if int(val)>60:
                led1.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
                led2.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
                led3.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
                led4.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
            else:
                led1.append([0, 0, 0])
                led2.append([0, 0, 0])
                led3.append([0, 0, 0])
                led4.append([0, 0, 0])


        t0.ledList = led1
        t1.ledList = led2
        t2.ledList = led3
        t3.ledList = led4

        
        q_cmd.put_nowait(("useToken", (t0, )))
        q_cmd.put_nowait(("useToken", (t1, )))
        q_cmd.put_nowait(("useToken", (t2, )))
        q_cmd.put_nowait(("useToken", (t3, )))

        # if not ledFlag1:
        #     q_cmd.put_nowait(("led_Arr_no_timing", (0, led5)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (1, led6)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (2, led7)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (3, led8)))
        #     ledFlag1 = True

        # for i, val in enumerate(d):
        #     if int(val)>60:
        #         vibFlag = False
        #         q_cmd.put_nowait(("vibrate_no_timeing", (0, 100, random.uniform(0.5, 1.0))))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (1, 100, random.uniform(0.5, 1.0))))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (2, 100, random.uniform(0.5, 1.0))))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (3, 100, random.uniform(0.5, 1.0))))
        #         break;
        #     elif i == 7:
        #         if vibFlag:
        #             return
        #         q_cmd.put_nowait(("vibrate_no_timeing", (0, 100, 0)))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (1, 100, 0)))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (2, 100, 0)))
        #         q_cmd.put_nowait(("vibrate_no_timeing", (3, 100, 0)))
        #         vibFlag = True
        #
        #
        #
        # # LED Example
        # for i, val in enumerate(d):
        #     if int(val)>60:
        #         led1.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led2.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led3.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led4.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         ledFlag = False
        #     else:
        #         led1.append([0, 0, 0])
        #         led2.append([0, 0, 0])
        #         led3.append([0, 0, 0])
        #         led4.append([0, 0, 0])
        #
        #
        # for i, val in enumerate(d):
        #     if int(val)>60:
        #         led5.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led6.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led7.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         led8.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
        #         ledFlag1 = False
        #     else:
        #         led5.append([0, 0, 0])
        #         led6.append([0, 0, 0])
        #         led7.append([0, 0, 0])
        #         led8.append([0, 0, 0])
        #
        #
        #
        #
        #
        #
        # if not ledFlag:
        #     q_cmd.put_nowait(("led_Arr_no_timing", (0, led1)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (1, led2)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (2, led3)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (3, led4)))
        #     ledFlag = True
        #
        # if not ledFlag1:
        #     q_cmd.put_nowait(("led_Arr_no_timing", (0, led5)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (1, led6)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (2, led7)))
        #     q_cmd.put_nowait(("led_Arr_no_timing", (3, led8)))
        #     ledFlag1 = True
        #

        # print(f"size of Q is: {q_cmd.qsize()}")
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
