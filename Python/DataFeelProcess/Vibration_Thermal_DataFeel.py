import time
from multiprocessing import Pipe, Process, Queue
from queue import Full, Empty
import time
from datafeel.device import Dot, VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
import threading
import minimalmodbus

DEV_LOCK = threading.Lock()

def VibrateDataFeel(device, vibrate_queue:Queue, stop_evt: threading.Event):
    with DEV_LOCK:
        device.registers.set_vibration_mode(VibrationMode.MANUAL)

    period = 0.2
    next_deadline = time.monotonic()
    last_f, last_a = 0.0, 0.0

    try:
        while not stop_evt.is_set():
            try:
                item = vibrate_queue.get(timeout=0.02)
                f, a = item
                # Drain any newer items so we act on the latest
                while True:
                    try:
                        f, a = vibrate_queue.get_nowait()
                    except Empty:
                        break
                last_f, last_a = float(f), float(a)
            except Empty:
                # nothing new; keep last_f/last_a
                pass

            # Actuate
            t0 = time.monotonic()
            with DEV_LOCK:
                if last_a > 0.1:
                    device.play_frequency(int(last_f), last_a)
                else:
                    device.stop_vibration()

            next_deadline += period
            rem = next_deadline - time.monotonic()
            if rem > 0:
                time.sleep(rem)
    finally:
        with DEV_LOCK:
            device.registers.set_vibration_mode(VibrationMode.OFF)

            
def ThermalDataFeel(device, thermal_queue:Queue, stop_evt: threading.Event, NORMAL : int = 34, HOTEST : int = 36, TONE_THRESHOLD : float = 0.25, DURATION_THRESHOLD : float = 0.1):
    with DEV_LOCK:
        device.registers.set_thermal_mode(ThermalMode.MANUAL)

    state = 0
    last_trigger = time.monotonic()
    lastCool_trigger = time.monotonic()
    start = False
    startCool = False
    tone_smooth = 0.0
    
    try:
        while not stop_evt.is_set():
            try:
                ts = thermal_queue.get(timeout=0.02)
                while True:
                    try:
                        ts = thermal_queue.get_nowait()
                    except Empty:
                        break
                tone_smooth = float(ts)
            except Empty:
                pass
            
            try:
                with DEV_LOCK:
                    # Thermal Protection and State Handle
                    if abs(device.registers.get_skin_temperature() - NORMAL) < 1:
                        # print("Stop Thermal...")
                        state = 0
                        device.registers.set_thermal_intensity(0.0)
                    elif device.registers.get_skin_temperature() >= HOTEST + 2:
                        state = -1
                        # print("Force Cooling...")
                        device.registers.set_thermal_intensity(-1.0)
                    elif device.registers.get_skin_temperature() < NORMAL:
                        state = 1
                        # print("Force Return...")
                        device.registers.set_thermal_intensity(1.0)
                    else :
                        state = 0

                    # Thermal Actuate
                    if tone_smooth >= TONE_THRESHOLD:
                        if start == False:
                            last_trigger = time.monotonic()
                            start = True
                        elif time.monotonic() - last_trigger >= DURATION_THRESHOLD:
                            if device.registers.get_thermal_power() <= 0.01 and state == 0:
                                # print("Heating...")
                                device.registers.set_thermal_intensity(1.0)
                    elif tone_smooth < TONE_THRESHOLD:
                        if startCool == False:
                            lastCool_trigger = time.monotonic()
                            startCool = True
                        elif time.monotonic() - lastCool_trigger >= DURATION_THRESHOLD*10:
                            start = False
                            startCool = False
                            if device.registers.get_thermal_power() >= 0.5 and state == 0:
                                # print("Cooling...")
                                device.registers.set_thermal_intensity(-1.0)
            
            except minimalmodbus.InvalidResponseError:
                time.sleep(0.005)

    finally:
        with DEV_LOCK:
            device.registers.set_thermal_mode(ThermalMode.OFF)