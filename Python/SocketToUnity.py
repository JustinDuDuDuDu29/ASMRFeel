import json
import socket
import threading
import time
from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from queue import Empty as QueueEmpty

class socketConfig:
    HOST = '127.0.0.1'
    PORT = 1688
    LISTEN = 8           # backlog 大一點
    ACCEPT_TIMEOUT = 1.0 # 秒；用來可中斷 accept
    GET_TIMEOUT = 0.1    # 秒；取 queue 可中斷
    SEND_RETRY_SLEEP = 0.05

def _to_bytes(payload) -> bytes:
    """
    將 payload 轉為 UTF-8 bytes：
    - str: 直接 encode
    - dict/list/其他: 轉成 JSON 再 encode
    """
    if payload is None:
        return b""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    # 其他型別 → JSON
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except Exception:
        # 最後退路：轉字串
        return str(payload).encode("utf-8")

def _send_framed(conn: socket.socket, data: bytes):
    """
    傳送 4 bytes big-endian 長度 + data
    """
    n = len(data)
    header = n.to_bytes(4, byteorder="big", signed=False)
    conn.sendall(header + data)

def SocketToUnity(stop_evt: Event, unityQueue: Queue, port: int):
    """
    長度前置的 TCP 伺服器。
    從 unityQueue 取資料，封包後送到 Unity。
    """
    while not stop_evt.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((socketConfig.HOST, port))
                s.listen(socketConfig.LISTEN)
                s.settimeout(socketConfig.ACCEPT_TIMEOUT)

                print(f"[Python] Listening on {socketConfig.HOST}:{socketConfig.PORT}")

                # 等待連線（可被 stop_evt 中斷）
                while not stop_evt.is_set():
                    try:
                        conn, addr = s.accept()
                        conn.settimeout(None)  # 用阻塞模式
                        print(f"[Python] New connection from {addr}")
                    except socket.timeout:
                        continue  # 回去檢查 stop_evt
                    except OSError as e:
                        print(f"[Python] accept OSError: {e}")
                        break

                    # 進入傳送 loop
                    try:
                        with conn:
                            while not stop_evt.is_set():
                                try:
                                    item = unityQueue.get(timeout=socketConfig.GET_TIMEOUT)
                                except QueueEmpty:
                                    continue

                                if item is None:
                                    # 你可用 None 當成特殊訊號（忽略或關閉）
                                    continue

                                data = _to_bytes(item)
                                # 空字串就不送
                                if not data:
                                    continue

                                _send_framed(conn, data)
                                # print("[Python] sent one frame")
                    except (BrokenPipeError, ConnectionResetError) as e:
                        print(f"[Python] Connection lost: {e}. Waiting for new client...")
                        # 回到 accept 等下一個連線
                        continue
                    except OSError as e:
                        print(f"[Python] Connection OSError: {e}.")
                        continue

        except OSError as e:
            print(f"[Python] Socket error (restart server loop): {e}")
            time.sleep(0.3)  # 稍微喘口氣再重建 socket
        except Exception as e:
            print(f"[Python] Unexpected error: {e}")
            time.sleep(0.3)

    print("[Python] Server stopped cleanly.")