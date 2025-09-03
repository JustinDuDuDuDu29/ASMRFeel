import multiprocessing
import sys
import serial
from serial.tools import list_ports
from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import random


def candy(dfc, method, args):
    getattr(dfc, method)(*args)



def worker(stop_evt:Event, q_cmd: Queue):
    dfc = DataFeelCenter(numOfDots=4)  
    
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
    first_pass_done = False
    while not stop_evt.is_set():
        tokenFlag = False
        d = q.get()
        therState, _d0, _d1 = d.split(";")
        d0 = _d0.split(",")
        d1 = _d1.split(",")

        led0 = [[0,0,0]]*8
        led1 = [[0,0,0]]*8


        t0 = token(superDotID = 0, vibFrequency=0, vibIntensity=0, therIntensity=0, ledList=led0 )
        t1 = token(superDotID = 1, vibFrequency=0, vibIntensity=0, therIntensity=0, ledList=led1 )
        # t1 = token(superDotID = 1, vibFrequency=0, vibIntensity=0, ledList=led1)

        
        # if random.random() < .5:
        #
        #     t0.vibFrequency = 200
        #     t0.vibIntensity = 1.0 
        #     # t0.vibIntensity = random.uniform(0.5, 1.0)
        #     t0.therIntensity = 1.0
        #     t0.therDiff= -3.0
        #     q_cmd.put_nowait(("useToken", (t0, )))
        #     q_cmd.put_nowait(("useToken", (t0, )))
        #     q_cmd.put_nowait(("useToken", (t0, )))
        #     q_cmd.put_nowait(("useToken", (t0, )))


        for i, val in enumerate(d0):
            if int(val)>60:

                first_pass_done = False 
                tokenFlag = True
                t0.vibFrequency = 100
                t0.vibIntensity = 1.0 
                t0.therIntensity = 1.0
                print(int(therState))
                if(int(therState) == 1):
                    t0.therDiff = 3.0
                else:
                    t0.therDiff = -3.0 

                if t0.ledList is None:
                    t0.ledList = [[0,0,0] * 8]
                    raise ValueError(f"t0.ledList is None!, have set to [[0,0,0]] * 8")
                t0.ledList[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

        for i, val in enumerate(d1):
            if int(val)>60:

                first_pass_done = False 
                tokenFlag = True
                t1.vibFrequency = 100
                t1.vibIntensity = 1.0 
                t1.therIntensity = 1.0
                if(int(therState) == 1):
                    t1.therDiff = 3.0
                else:
                    t1.therDiff = -3.0 

                if t1.ledList is None:
                    t1.ledList = [[0,0,0] * 8]
                    raise ValueError(f"t0.ledList is None!, have set to [[0,0,0]] * 8")
                t1.ledList[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                
                
        # for i, val in enumerate(d1):
        #     if int(val)>60:
        #         t1.vibFrequency = 100
        #         t1.vibIntensity = random.uniform(0.5, 1.0)
        #         if t1.ledList is None:
        #             t1.ledList = [[0,0,0] * 8]
        #             raise ValueError(f"t0.ledList is None!, have set to [[0,0,0]] * 8")
        #         t1.ledList[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        

        # if not tokenFlag:
        #     if not first_pass_done:
        #         first_pass_done = True
        #     else:
        #         continue  # skip sending after the first pass


        q_cmd.put_nowait(("useToken", (t0, )))
        q_cmd.put_nowait(("useToken", (t1, )))




def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
    """Line-framed reader. Reconnects on failure."""

    while not stop_evt.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                ser.reset_input_buffer()
                while not stop_evt.is_set():
                    try:
                        line = ser.readline()  

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
