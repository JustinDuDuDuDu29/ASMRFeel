from time import sleep
from datafeel.device import VibrationMode, discover_devices, LedMode, ThermalMode, VibrationWaveforms
devices = discover_devices(4)

device = devices[0]

# Vibration Low-Level API
device.registers.set_vibration_mode(VibrationMode.MANUAL)
device.registers.set_vibration_frequency(100)
device.registers.set_vibration_intensity(0.4)

