from time import sleep
import time
from typing import List
from datafeel.device import Dot, LedMode, ThermalMode, VibrationMode
import serial
import serial.tools.list_ports
import struct
from dataclasses import dataclass
class SuperDot(Dot):
    def __init__(self, port, id):
        super().__init__(port, id)
        self.isVibrating = False;
        self.isThermaling= False;
        self.registers.set_vibration_mode(VibrationMode.MANUAL)
        self.registers.set_thermal_mode(ThermalMode.MANUAL)
        self.registers.set_led_mode(LedMode.INDIVIDUAL_MANUAL)
        self.therBase = self.registers.get_skin_temperature()
        self.therCurrent = self.registers.get_skin_temperature()
        self.therTarget: float | None = None
        self.therIntehsity: float = 0
        self.vibFrequency: float = 0
        self.vibIntensity: float = 0
        self.ledList: List[List[int]] = [[0,0,0]] * 8
    

# x =-12 
# aaa = struct.unpack('>I', struct.pack('>f', x))[0]
# print(bin(aaa))
# print(bin(aaa & 0xFFFF))
# print(bin(aaa >> 16))

@dataclass
class token:
    superDotID: int | None = None 
    therIntensity: float | None = None 
    therDiff: float | None = None 
    vibFrequency: float | None = None 
    vibIntensity: float | None = None 
    vibIntensity: float | None = None 
    ledList: List[List[int]] | None =  None


class DataFeelCenter():
    def __init__(self, numOfDots):
        self.superDotArr = self.discover_devices_Super(numOfDots)

    def useToken(self, token:token):
        if token.superDotID is None:
            return
        print(token)

        targetDot = self.superDotArr[token.superDotID]

        if token.therDiff is not None:
            diff = targetDot.therCurrent - (targetDot.therBase + token.therDiff)
            if diff < 0.3:
                token.therIntensity = min(1.0, diff)
            elif diff > 0.3:
                token.therIntensity = max(-1.0, -diff)
            else:
                token.therIntensity = 0

        if token.therIntensity is None:
            token.therIntensity = targetDot.therIntehsity
        targetDot.therIntehsity = token.therIntensity
        
        if token.vibFrequency is None:
            token.vibFrequency = targetDot.vibFrequency
        targetDot.vibFrequency= token.vibFrequency
            
        if token.vibIntensity is None:
            token.vibIntensity= targetDot.vibIntensity
        targetDot.vibIntensity= token.vibIntensity

        if token.ledList is None:
            token.ledList = targetDot.ledList
        targetDot.ledList = token.ledList

        targetDot.therCurrent = targetDot.registers.set_all(targetDot.ledList, therIntensity=token.therIntensity, vibFrequency=token.vibFrequency, vibIntensity=token.vibIntensity)
        # if token.ledList is not None:
        #     self.led_Arr_no_timing(token.superDotID, token.ledList) 
        # if token.therIntensity:
        #     pass
        # if token.vibFrequency is not None and token.vibIntensity is not None:
        #     self.vibrate_no_timeing(token.superDotID, token.vibFrequency, token.vibIntensity) 



    def vibrate_no_timeing(self, superDotID: int, frequency: float, intensity: float):
        targetDot = self.superDotArr[superDotID]
        if targetDot.vibFrequency == frequency and targetDot.vibIntensity == intensity:
            return
        targetDot.registers.set_vibration_fast(frequency=frequency, intensity=intensity)
        targetDot.vibFrequency = frequency
        targetDot.vibIntensity = intensity

    def vibrate_no_timeing_Orig(self, superDotID: int, frequency: float, intensity: float):
        targetDot = self.superDotArr[superDotID]
        targetDot.registers.set_vibration_frequency(frequency)
        targetDot.registers.set_vibration_intensity(intensity)


    def thermal_no_timing(self, superDotID:int, intensity: float):
        targetDot = self.superDotArr[superDotID]
        targetDot.registers.set_thermal_intensity(intensity)
        targetDot.isThermaling = True

    def led_Arr_no_timing(self, superDotID: int, ledLists: List[List[int]]):
        targetDot = self.superDotArr[superDotID]
        if targetDot.ledList == ledLists:
            return
        targetDot.registers.set_individual_leds(ledLists)
        targetDot.ledList = ledLists

    def led_no_timing(self, superDotID: int, ledID:int, r:int, g:int, b:int):
        targetDot = self.superDotArr[superDotID]
        if not targetDot.ledList[ledID][0] == r or not targetDot.ledList[ledID][1] == g or targetDot.ledList[ledID][2] == b:
            targetDot.registers.set_individual_led(ledID, r, g, b)
            targetDot.ledList[ledID][0] = r
            targetDot.ledList[ledID][1] = g
            targetDot.ledList[ledID][2] = b

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

