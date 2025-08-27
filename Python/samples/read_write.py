# import sys, time, threading
# import serial
# import multiprocessing
# from serial.tools import list_ports
# from time import sleep
# from multiprocessing.synchronize import Event
# from multiprocessing import Pipe, Process, Queue
# from datafeel.deviceXXX import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
# import asyncio
#
# devices = []
#
# def choose_port(default=None):
#     ports = list(list_ports.comports())
#     if not ports:
#         print("No serial ports found.")
#         sys.exit(1)
#     print("Available ports:")
#     for i, p in enumerate(ports):
#         print(f"  [{i}] {p.device}  {p.description}")
#     prompt = f"Select port index [{0 if default is None else default}]: "
#     try:
#         idx = input(prompt).strip()
#         idx = int(idx) if idx else (0 if default is None else default)
#     except ValueError:
#         idx = 0
#     return ports[idx].device
#
# async def pressureLED(devices, stop_evt: Event, q:Queue):
#
#     print(f"in pro: {len(devices)}")
#     await devices[0].registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL, devices[0].id)
#     while not stop_evt.is_set():
#         if(q.empty()):
#             await devices[0].set_led_off()
#
#         d = q.get()
#         print(d)
#         d = d.split(",")
#
#         for i, val in enumerate(d):
#             # print(f"{i}: {val}")
#             if int(val)>100:
#                 await devices[0].registers.set_individual_led(i, 255, 0, 0, devices[0].id)
#             else:
#                 await devices[0].registers.set_individual_led(i, 0, 0, 0, devices[0].id)
#             # sleep(.1)
#
#
#
# def read_from_serial(stop_evt: Event, q:Queue, port: str, baud: int = 115200):
#     """Line-framed reader. Reconnects on failure."""
#     while not stop_evt.is_set():
#         try:
#             with serial.Serial(port, baudrate=baud, timeout=1) as ser:
#                 # Optional: set DTR/RTS here if your device needs it
#                 while not stop_evt.is_set():
#                     try:
#                         line = ser.readline()  # b'' on timeout
#                         if line:
#                             # TODO: replace with your handling
#                             q.put(line.rstrip(b"\r\n").decode("utf-8"))
#                     except serial.SerialException as e:
#                         print(f"[serial] read error: {e}; will reconnect")
#                         break  # break inner loop → outer reconnects
#         except serial.SerialException as e:
#             print(f"[serial] open error on {port}: {e}; retrying...")
#         # small backoff before reconnect
#         stop_evt.wait(1.0)
#
# async def main():
#
#     devices = await discover_devices(4)
#
#     print(len(devices))
#
#     baud = 115200
#     port = choose_port()
#     print(f"Starting connection at {port} {baud}…")
#
#     q = Queue()
#     stop_evt = multiprocessing.Event()
#     t = threading.Thread(target=read_from_serial, args=(stop_evt, q, port, baud), daemon=True)
#     t.start()
#     p =threading.Thread(target=asyncio.run, args=(pressureLED(devices, stop_evt, q),), daemon= True) 
#     # p =threading.Thread(target=pressureLED, args=(devices, stop_evt, q), daemon= True) 
#     p.start()
#     print("Press 'q' then Enter to quit.")
#     try:
#         for line in sys.stdin:
#             if line.strip().lower() == "q":
#                 print("Quit requested.")
#                 break
#     except KeyboardInterrupt:
#         pass
#     finally:
#         stop_evt.set()
#         t.join(timeout=2)
#         p.join()
#         print("Stopped cleanly.")
#
# if __name__ == "__main__":
#     asyncio.run(main())
#
import asyncio
from time import sleep
from datafeel.deviceXXX import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
async def main():
    devices = await discover_devices(4)

    print("xxxxx")
    device = devices[0]

    print("======")
    print(devices)

    print("======")
    print("found", len(devices), "devices")
    print(device)
    # print("Reading all data...")
    # print("skin temperature:", await device.registers.get_skin_temperature(device.id))
    # print("sink temperature:", device.registers.get_sink_temperature())
    # print("mcu temperature:", device.registers.get_mcu_temperature())
    # print("gate driver temperature:", device.registers.get_gate_driver_temperature())
    # print("thermal power:", device.registers.get_thermal_power())
    # print("thermal mode:", device.registers.get_thermal_mode())
    # print("thermal intensity:", device.registers.get_thermal_intensity())
    # print("vibration mode:", device.registers.get_vibration_mode())
    # print("vibration frequency:", device.registers.get_vibration_frequency())
    # print("vibration intensity:", device.registers.get_vibration_intensity())
    # print("vibration go:", device.registers.get_vibration_go())
    # print("vibration sequence 0123:", device.registers.get_vibration_sequence_0123())
    # print("vibration sequence 3456:", device.registers.get_vibration_sequence_3456())

# Set the global LED color
    await device.set_led(255, 0, 0)
    sleep(1)

# Set the color of a specific LED
    await device.set_led(255, 0, 0, 5)
    sleep(1)

# Set the LED to breathe mode
    await device.set_led_breathe()
    sleep(1)

    await device.set_led_off()

# play an A major scale
    # frequencies = [110, 123, 139, 147, 165, 185, 208, 220]
    #
    # device.registers.set_vibration_mode(VibrationMode.MANUAL)   
    # for frequency in frequencies:
    #     device.play_frequency(frequency, 1.0)
    #     sleep(0.5)
    #
    # vibration_sequence = [
    #     VibrationWaveforms.STRONG_BUZZ_P100,
    #     VibrationWaveforms.Rest(0.5),
    #     VibrationWaveforms.TRANSITION_HUM1_P100,
    #     VibrationWaveforms.Rest(0.5),
    #     VibrationWaveforms.TRANSITION_RAMP_DOWN_MEDIUM_SHARP2_P50_TO_P0,
    #     VibrationWaveforms.TRANSITION_RAMP_UP_MEDIUM_SHARP2_P0_TO_P50,
    #     VibrationWaveforms.DOUBLE_CLICK_P100,
    #     VibrationWaveforms.TRANSITION_RAMP_UP_SHORT_SMOOTH2_P0_TO_P100,
    # ]
    #
    # device.play_vibration_sequence(vibration_sequence)
    # sleep(3)
    # device.set_vibration_sequence(vibration_sequence)
    # sleep(3)
    #
    # device.play_frequency(250, 1.0)
    #
    # device.stop_vibration()
    #
    # sleep(1)
    #
    await device.set_led(0, 255, 0, device.id)
    print("aaa")
    print("global led:", await device.registers.get_global_led(id=device.id))
    sleep(1)

    await device.registers.set_led_mode(LedMode.BREATHE, device.id)
    sleep(1)

#     device.activate_thermal_intensity_control(1.0) # Heating
#     sleep(5)
#     device.activate_thermal_intensity_control(-1.0) # Cooling
#     sleep(5)
#     device.disable_all_thermal()
#
# # If you don't like the high-level API, you can use the low-level API for any of the features
#
# # Thermal Low-Level API
#     device.registers.set_thermal_mode(ThermalMode.MANUAL)
#     device.registers.set_thermal_intensity(1.0)
#     print("\r\n\r\nHeating for 10 seconds...\r\n\r\n")
#
#     for i in range(20):
#         print("skin temperature:", device.registers.get_skin_temperature())
#         print("thermal power:", device.registers.get_thermal_power())
#         sleep(0.5)
#
#     device.registers.set_thermal_intensity(-1.0)
#     print("\r\n\r\nCooling for 10 seconds...\r\n\r\n")
#     for i in range(20):
#         print("skin temperature:", device.registers.get_skin_temperature())
#         print("thermal power:", device.registers.get_thermal_power())
#         sleep(0.5)
#     device.registers.set_thermal_mode(ThermalMode.OFF)
#
#
#
# # Vibration Low-Level API
#     device.registers.set_vibration_mode(VibrationMode.MANUAL)
#     device.registers.set_vibration_frequency(200)
#     device.registers.set_vibration_intensity(1.0)
#     sleep(3)
#
# LED Low-Level API
    await device.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL, device.id)
    await device.registers.set_individual_led(0, 255, 0, 0, device.id)
    sleep(.5)
    await device.registers.set_individual_led(1, 0, 255, 0, device.id)
    sleep(.5)
    await device.registers.set_individual_led(2, 0, 0, 255, device.id)
    sleep(.5)
    await device.registers.set_individual_led(3, 255, 255, 0, device.id)
    sleep(.5)
    await device.registers.set_individual_led(4, 255, 0, 255, device.id)
    sleep(.5)
    await device.registers.set_individual_led(5, 0, 255, 255, device.id)
    sleep(.5)
    await device.registers.set_individual_led(6, 255, 255, 255, device.id)
    sleep(.5)
    await device.registers.set_individual_led(7, 255, 255, 255, device.id)

asyncio.run(main())
