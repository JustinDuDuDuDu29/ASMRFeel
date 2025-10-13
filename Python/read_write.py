from time import sleep
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
devices = discover_devices(4)

device = devices[0]
device2 = devices[1]

device.registers.set_led_mode(LedMode.GLOBAL_MANUAL)
device2.registers.set_led_mode(LedMode.GLOBAL_MANUAL)
device.registers.set_global_led(255,0,0)
device2.registers.set_global_led(255,0,0)
