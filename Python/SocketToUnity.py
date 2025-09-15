import multiprocessing
from multiprocessing import queues
import socket
from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
import time
import threading
import queue

class socketConfig:
    HOST = '127.0.0.1'
    PORT = 1688
    LISTEN = 1

def SocketToUnity(stop_evt: Event, unityQueue: Queue):
    while not stop_evt.is_set():
        try:
            with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as s:
                s.bind((socketConfig.HOST, socketConfig.PORT))
                s.listen(socketConfig.LISTEN)

                print(f"Python listening on {socketConfig.HOST}: {socketConfig.PORT}")
                conn, addr = s.accept()
                with conn:
                    print(f"New connection on {addr}!")
                    print("Cleaning queue...")

                    while not stop_evt.is_set():

                        recv = unityQueue.get()
                        if not recv:
                            continue
                        
                        conn.sendall(bytes(str(recv), "utf-8"))
                        print("sent")
                
            
        except TimeoutError:
            print(f"The socket has encounter a timeout!")
        except InterruptedError:
            print(f"The socket has encounter a Interrupt!")
        except BrokenPipeError:
            print(f"The pipe is broken! Try reconnecting")

        



# if __name__ == "__main__":
#     stop_evt = multiprocessing.Event()
#     unityQueue = Queue(maxsize=1)
#     t = threading.Thread(target=SocketToUnity, args=(stop_evt, unityQueue, ), daemon=True)
#     t.start()
#     while True:
#         try:
#             unityQueue.put("abc")
#             time.sleep(1)
#         except queue.Full:
#             try:
#                 unityQueue.get_nowait()
#             except queue.Empty:
#                 pass
#             try:
#                 unityQueue.put_nowait("abc")
#             except queue.Full:
#                 pass
#     t.join()
