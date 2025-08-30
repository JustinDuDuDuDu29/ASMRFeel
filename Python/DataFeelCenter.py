from time import sleep
import time
from typing import List
from datafeel.device import Dot, LedMode, ThermalMode, VibrationMode
import serial
import serial.tools.list_ports

class SuperDot(Dot):
    def __init__(self, port, id):
        super().__init__(port, id)
        self.isVibrating = False;
        self.isThermaling= False;
        self.isLEDArr= [False] * 8;
        self.registers.set_vibration_mode(VibrationMode.MANUAL)
        self.registers.set_thermal_mode(ThermalMode.MANUAL)
        self.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
    

    
class DataFeelCenter():
    def __init__(self, numOfDots):
        self.superDotArr = self.discover_devices_Super(numOfDots)

    def vibrate_no_timeing(self, superDotID: int, frequence: float, intensity: float):
        targetDot = self.superDotArr[superDotID]
        targetDot.registers.set_vibration_frequency(frequence)
        targetDot.registers.set_vibration_intensity(intensity)


    def thermal_no_timing(self, superDotID:int, intensity: float):
        targetDot = self.superDotArr[superDotID]
        targetDot.registers.set_thermal_intensity(intensity)
        targetDot.isThermaling = True

    def led_no_timing(self, superDotID: int, ledID:int, r:int, g:int, b:int):
        print(f"{superDotID}, {ledID}, {r}, {g}, {b}")
        targetDot = self.superDotArr[superDotID]
        targetDot.registers.set_individual_led(ledID, r, g, b)
        targetDot.isLEDArr[superDotID] = True

    @staticmethod
    def discover_devices_Super(maxAddress) -> List[SuperDot]:
        """
        Discover all DataFeel Devices connected to the computer.
        """
        devices = []
        # first, find the serial port the Dot is connected to
        for port in serial.tools.list_ports.comports():
            if port.vid == 0x10c4 and port.pid == 0xea60:
                print("found DataFeel Device on port", port.device)
                
                for x in range(1, maxAddress + 1):
                    try:
                        dot = SuperDot(port.device, x)
                        devices.append(dot)
                    except Exception as e:
                        print(f"No device at address {x}")
                break
        return devices

