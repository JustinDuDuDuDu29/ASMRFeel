from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import serial
import queue as pyqueue

def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
    """Line-framed reader. Reconnects on failure."""

    while not stop_evt.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=1) as ser:
                # set receive buffer to 1
                # ser.set_buffer_size(rx_size=0, tx_size=0)
                ser.reset_input_buffer()
                
                # read buffer size
                while not stop_evt.is_set():
                    try:
                        # print(f"is Empty? {q.empty()}")
                        line = ser.readline()  
                        # buffer_size = ser.in_waiting
                        # print(f"[serial] connected to {port} at {baud} baud, buffer size {buffer_size}")
                        if line:
                            d = line.rstrip(b"\r\n").decode("utf-8")
                            # q.put(d)


                            # workaround: because arduino clock is different from pc, we need to adjust the timing
                            try:
                                q.put_nowait(d)
                            except pyqueue.Full:
                                try:
                                    q.get_nowait()
                                except pyqueue.Empty:
                                    pass
                                try:
                                    q.put_nowait(d)
                                except pyqueue.Full:
                                    pass
                    except serial.SerialException as e:
                        print(f"[serial] read error: {e}; will reconnect")
                        break  
        except serial.SerialException as e:
            print(f"[serial] open error on {port}: {e}; retrying...")
        stop_evt.wait(1.0)