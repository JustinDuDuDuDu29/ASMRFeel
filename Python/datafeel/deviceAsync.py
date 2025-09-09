from math import ceil
import struct
from time import time
from typing import List, Tuple
import serial
import serial.tools.list_ports
import minimalmodbus as modbus
from enum import IntEnum

def _from_IEEE754(int_val) -> int:
    return struct.unpack('!f', struct.pack('!I', int_val))[0]

def _to_Long_Int(val: int, littleEndianSwap: bool = False, swap: bool = False, littleEndian: bool = False) -> Tuple[int, int]:
    #TODO: Strange, need testing

    res = format(val, '032b')

    a = res[0:8]
    b = res[8:16]
    c = res[16:24]
    d = res[24:32]

    res = d + c + b + a

    if littleEndian:
        res = a + b + c + d
        return int(res[0: 16], 2), int(res[16:32], 2)
        
    if swap:
        res = c + d + a + b
        return int(res[0: 16], 2), int(res[16:32], 2)

    if littleEndianSwap:
        res = b + a + d + c
        return int(res[0: 16], 2), int(res[16:32], 2)

    
    return int(res[0: 16], 2), int(res[16:32], 2)


def _to_IEEE754(val: int | float | str) -> int:
    if isinstance(val, str):
       val = float(val) 
    return struct.unpack('>I', struct.pack('>f', val))[0]

def _fix_string_endianness(string):
    return ''.join(string[i:i+2][::-1] for i in range(0, len(string), 2))

class VibrationWaveforms(IntEnum):
    END_SEQUENCE = 0
    STRONG_CLICK_P100 = 1
    STRONG_CLICK_P60 = 2
    STRONG_CLICK_P30 = 3
    SHARP_CLICK_P100 = 4
    SHARP_CLICK_P60 = 5
    SHARP_CLICK_P30 = 6
    SOFT_BUMP_P100 = 7
    SOFT_BUMP_P60 = 8
    SOFT_BUMP_P30 = 9
    DOUBLE_CLICK_P100 = 10
    DOUBLE_CLICK_P60 = 11
    TRIPLE_CLICK_P100 = 12
    SOFT_FUZZ_P60 = 13
    STRONG_BUZZ_P100 = 14
    ALERT_750MS_P100 = 15
    ALERT_1000MS_P100 = 16
    STRONG_CLICK1_P100 = 17
    STRONG_CLICK2_P80 = 18
    STRONG_CLICK3_P60 = 19
    STRONG_CLICK4_P30 = 20
    MEDIUM_CLICK1_P100 = 21
    MEDIUM_CLICK2_P80 = 22
    MEDIUM_CLICK3_P60 = 23
    SHARP_TICK1_P100 = 24
    SHARP_TICK2_P80 = 25
    SHARP_TICK3_P60 = 26
    SHORT_DOUBLE_CLICK_STRONG1_P100 = 27
    SHORT_DOUBLE_CLICK_STRONG2_P80 = 28
    SHORT_DOUBLE_CLICK_STRONG3_P60 = 29
    SHORT_DOUBLE_CLICK_STRONG4_P30 = 30
    SHORT_DOUBLE_CLICK_MEDIUM1_P100 = 31
    SHORT_DOUBLE_CLICK_MEDIUM2_P80 = 32
    SHORT_DOUBLE_CLICK_MEDIUM3_P60 = 33
    SHORT_DOUBLE_SHARP_TICK1_P100 = 34
    SHORT_DOUBLE_SHARP_TICK2_P80 = 35
    SHORT_DOUBLE_SHARP_TICK3_P60 = 36
    LONG_DOUBLE_SHARP_CLICK_STRONG1_P100 = 37
    LONG_DOUBLE_SHARP_CLICK_STRONG2_P80 = 38
    LONG_DOUBLE_SHARP_CLICK_STRONG3_P60 = 39
    LONG_DOUBLE_SHARP_CLICK_STRONG4_P30 = 40
    LONG_DOUBLE_SHARP_CLICK_MEDIUM1_P100 = 41
    LONG_DOUBLE_SHARP_CLICK_MEDIUM2_P80 = 42
    LONG_DOUBLE_SHARP_CLICK_MEDIUM3_P60 = 43
    LONG_DOUBLE_SHARP_TICK1_P100 = 44
    LONG_DOUBLE_SHARP_TICK2_P80 = 45
    LONG_DOUBLE_SHARP_TICK3_P60 = 46
    BUZZ1_P100 = 47
    BUZZ2_P80 = 48
    BUZZ3_P60 = 49
    BUZZ4_P40 = 50
    BUZZ5_P20 = 51
    PULSING_STRONG1_P100 = 52
    PULSING_STRONG2_P60 = 53
    PULSING_MEDIUM1_P100 = 54
    PULSING_MEDIUM2_P60 = 55
    PULSING_SHARP1_P100 = 56
    PULSING_SHARP2_P60 = 57
    TRANSITION_CLICK1_P100 = 58
    TRANSITION_CLICK2_P80 = 59
    TRANSITION_CLICK3_P60 = 60
    TRANSITION_CLICK4_P40 = 61
    TRANSITION_CLICK5_P20 = 62
    TRANSITION_CLICK6_P10 = 63
    TRANSITION_HUM1_P100 = 64
    TRANSITION_HUM2_P80 = 65
    TRANSITION_HUM3_P60 = 66
    TRANSITION_HUM4_P40 = 67
    TRANSITION_HUM5_P20 = 68
    TRANSITION_HUM6_P10 = 69
    TRANSITION_RAMP_DOWN_LONG_SMOOTH1_P100_TO_P0 = 70
    TRANSITION_RAMP_DOWN_LONG_SMOOTH2_P100_TO_P0 = 71
    TRANSITION_RAMP_DOWN_MEDIUM_SMOOTH1_P100_TO_P0 = 72
    TRANSITION_RAMP_DOWN_MEDIUM_SMOOTH2_P100_TO_P0 = 73
    TRANSITION_RAMP_DOWN_SHORT_SMOOTH1_P100_TO_P0 = 74
    TRANSITION_RAMP_DOWN_SHORT_SMOOTH2_P100_TO_P0 = 75
    TRANSITION_RAMP_DOWN_LONG_SHARP1_P100_TO_P0 = 76
    TRANSITION_RAMP_DOWN_LONG_SHARP2_P100_TO_P0 = 77
    TRANSITION_RAMP_DOWN_MEDIUM_SHARP1_P100_TO_P0 = 78
    TRANSITION_RAMP_DOWN_MEDIUM_SHARP2_P100_TO_P0 = 79
    TRANSITION_RAMP_DOWN_SHORT_SHARP1_P100_TO_P0 = 80
    TRANSITION_RAMP_DOWN_SHORT_SHARP2_P100_TO_P0 = 81
    TRANSITION_RAMP_UP_LONG_SMOOTH1_P0_TO_P100 = 82
    TRANSITION_RAMP_UP_LONG_SMOOTH2_P0_TO_P100 = 83
    TRANSITION_RAMP_UP_MEDIUM_SMOOTH1_P0_TO_P100 = 84
    TRANSITION_RAMP_UP_MEDIUM_SMOOTH2_P0_TO_P100 = 85
    TRANSITION_RAMP_UP_SHORT_SMOOTH1_P0_TO_P100 = 86
    TRANSITION_RAMP_UP_SHORT_SMOOTH2_P0_TO_P100 = 87
    TRANSITION_RAMP_UP_LONG_SHARP1_P0_TO_P100 = 88
    TRANSITION_RAMP_UP_LONG_SHARP2_P0_TO_P100 = 89
    TRANSITION_RAMP_UP_MEDIUM_SHARP1_P0_TO_P100 = 90
    TRANSITION_RAMP_UP_MEDIUM_SHARP2_P0_TO_P100 = 91
    TRANSITION_RAMP_UP_SHORT_SHARP1_P0_TO_P100 = 92
    TRANSITION_RAMP_UP_SHORT_SHARP2_P0_TO_P100 = 93
    TRANSITION_RAMP_DOWN_LONG_SMOOTH1_P50_TO_P0 = 94
    TRANSITION_RAMP_DOWN_LONG_SMOOTH2_P50_TO_P0 = 95
    TRANSITION_RAMP_DOWN_MEDIUM_SMOOTH1_P50_TO_P0 = 96
    TRANSITION_RAMP_DOWN_MEDIUM_SMOOTH2_P50_TO_P0 = 97
    TRANSITION_RAMP_DOWN_SHORT_SMOOTH1_P50_TO_P0 = 98
    TRANSITION_RAMP_DOWN_SHORT_SMOOTH2_P50_TO_P0 = 99
    TRANSITION_RAMP_DOWN_LONG_SHARP1_P50_TO_P0 = 100
    TRANSITION_RAMP_DOWN_LONG_SHARP2_P50_TO_P0 = 101
    TRANSITION_RAMP_DOWN_MEDIUM_SHARP1_P50_TO_P0 = 102
    TRANSITION_RAMP_DOWN_MEDIUM_SHARP2_P50_TO_P0 = 103
    TRANSITION_RAMP_DOWN_SHORT_SHARP1_P50_TO_P0 = 104
    TRANSITION_RAMP_DOWN_SHORT_SHARP2_P50_TO_P0 = 105
    TRANSITION_RAMP_UP_LONG_SMOOTH1_P0_TO_P50 = 106
    TRANSITION_RAMP_UP_LONG_SMOOTH2_P0_TO_P50 = 107
    TRANSITION_RAMP_UP_MEDIUM_SMOOTH1_P0_TO_P50 = 108
    TRANSITION_RAMP_UP_MEDIUM_SMOOTH2_P0_TO_P50 = 109
    TRANSITION_RAMP_UP_SHORT_SMOOTH1_P0_TO_P50 = 110
    TRANSITION_RAMP_UP_SHORT_SMOOTH2_P0_TO_P50 = 111
    TRANSITION_RAMP_UP_LONG_SHARP1_P0_TO_P50 = 112
    TRANSITION_RAMP_UP_LONG_SHARP2_P0_TO_P50 = 113
    TRANSITION_RAMP_UP_MEDIUM_SHARP1_P0_TO_P50 = 114
    TRANSITION_RAMP_UP_MEDIUM_SHARP2_P0_TO_P50 = 115
    TRANSITION_RAMP_UP_SHORT_SHARP1_P0_TO_P50 = 116
    TRANSITION_RAMP_UP_SHORT_SHARP2_P0_TO_P50 = 117
    LONG_BUZZ_FOR_PROGRAMMATIC_STOPPING_P100 = 118
    SMOOTH_HUM_P50 = 119
    SMOOTH_HUM_P40 = 120
    SMOOTH_HUM_P30 = 121
    SMOOTH_HUM_P20 = 122
    SMOOTH_HUM_P10 = 123

    def Rest(seconds):
        if seconds > 1.270 or seconds < 0.0:
            raise ValueError("Seconds must be between 0.0 and 1.270")

        return 0x80 + int(round(seconds * 100.0))

class LedMode(IntEnum):
    OFF = 0
    GLOBAL_MANUAL = 1
    INDIVIDUAL_MANUAL = 2
    TRACK_THERMAL = 3
    BREATHE = 4

class VibrationMode(IntEnum):
    OFF = 0
    MANUAL = 1
    LIBRARY = 2
    SWEEP_FREQUENCY = 3
    SWEEP_INTENSITY = 4

class ThermalMode(IntEnum):
    OFF = 0
    MANUAL = 1
    TEMPERATURE_TARGET = 2


class Dot:
    class V63Registers():
        # RO
        DEVICE_NAME = 0
        HARDWARE_ID = 32
        FIRMWARE_ID = 64
        SERIAL_NUMBER = 96

        # RO
        SKIN_TEMP = 1000
        SINK_TEMP = 1002
        MCU_TEMP = 1004
        GATE_DRIVER_TEMP = 1006
        THERMAL_POWER = 1008

        # RW
        LED_MODE = 1010
        GLOBAL_MANUAL = 1012
        LED_INDIVIDUAL_MANUAL_0 = 1014
        LED_INDIVIDUAL_MANUAL_1 = 1016
        LED_INDIVIDUAL_MANUAL_2 = 1018
        LED_INDIVIDUAL_MANUAL_3 = 1020
        LED_INDIVIDUAL_MANUAL_4 = 1022
        LED_INDIVIDUAL_MANUAL_5 = 1024
        LED_INDIVIDUAL_MANUAL_6 = 1026
        LED_INDIVIDUAL_MANUAL_7 = 1028

        THERMAL_MODE = 1030
        THERMAL_INTENSITY = 1032
        THERMAL_SKIN_TEMP_TARGET = 1034

        VIBRATION_MODE = 1036
        VIBRATION_FREQUENCY = 1038
        VIBRATION_INTENSITY = 1040
        VIBRATION_GO = 1042
        VIBRATION_SEQUENCE_0123 = 1044
        VIBRATION_SEQUENCE_4567 = 1046

        def __init__(self, port, id):
            self.dev = modbus.Instrument(port, id, modbus.MODE_RTU,close_port_after_each_call=False, debug=False)
            self.dev.clear_buffers_before_each_transaction = True
            self.dev.serial.baudrate = 115200
            self.dev.serial.bytesize = 8    
            self.dev.serial.parity = serial.PARITY_NONE
            self.dev.serial.stopbits = 1


        def get_skin_Temp_Quick(self) -> float :

            lsw, msw = self.dev.read_registers(registeraddress = self.SKIN_TEMP, number_of_registers=2)
            return _from_IEEE754((msw<< 16) + lsw)

        def set_all(self, ledLists: List[List[int]], therIntensity: float, vibFrequency: float, vibIntensity:float) -> float :
            vals = []
            for r, g, b in ledLists:
                val = (b << 16) | (r << 8) | g 
                vals.append(val & 0xFFFF)
                vals.append(val >> 16)

            lsw = int(int(ThermalMode.MANUAL) % 65535)
            msw = int(int(ThermalMode.MANUAL) / 65535)
            # lsw, msw = _to_Long_Int(int(ThermalMode.MANUAL),littleEndianSwap = True)
            # vals.append(0)
            # vals.append(0)
            vals.append(lsw)
            vals.append(msw)

            # print(therIntensity)
            ti = _to_IEEE754(therIntensity)
            vals.append(ti & 0xFFFF)
            vals.append(ti >> 16)
            
            tt = _to_IEEE754(26.5)
            vals.append(tt & 0xFFFF)
            vals.append(tt >> 16)
            

            lsw = int(int(VibrationMode.MANUAL) % 65535)
            msw = int(int(VibrationMode.MANUAL) / 65535)
            # lsw, msw = _to_Long_Int(int(ThermalMode.MANUAL),littleEndianSwap = True)
            # vals.append(0)
            # vals.append(0)
            vals.append(lsw)
            vals.append(msw)
            # vm = _to_IEEE754(int(VibrationMode.MANUAL))
            # vals.append(1)
            # vals.append(0)
            # vals.append(vm & 0xFFFF)
            # vals.append(vm >> 16)

            vf = _to_IEEE754(vibFrequency)
            vals.append(vf & 0xFFFF)
            vals.append(vf >> 16)
            
            vi = _to_IEEE754(vibIntensity)
            vals.append(vi & 0xFFFF)
            vals.append(vi >> 16)

            # print(vals)

            
            # self.dev.read_float(self.SINK_TEMP, 3, 2, modbus.BYTEORDER_LITTLE_SWAP)   

            self.dev.write_registers(registeraddress = self.LED_INDIVIDUAL_MANUAL_0, values=vals)
            lsw, msw = self.dev.read_registers(registeraddress = self.SKIN_TEMP, number_of_registers=2)
            return _from_IEEE754((msw<< 16) + lsw)

    def __init__(self, port, id):
        self.port = port
        self.id = id
        self.registers = self.V63Registers(port, id)
        self.device_name = _fix_string_endianness(self.registers.dev.read_string(self.V63Registers.DEVICE_NAME,32, 3))
        self.hardware_id = _fix_string_endianness(self.registers.dev.read_string(self.V63Registers.HARDWARE_ID, 32, 3))
        self.firmware_id = _fix_string_endianness(self.registers.dev.read_string(self.V63Registers.FIRMWARE_ID, 32, 3)) 
        self.serial_number = _fix_string_endianness(self.registers.dev.read_string(self.V63Registers.SERIAL_NUMBER, 32, 3))

    def __str__(self):
        return f"Dot {self.id} (Name = {self.device_name}, Hardware ID = {self.hardware_id}, Firmware ID = {self.firmware_id}, Serial Number = {self.serial_number})"


def discover_devices(maxAddress) -> List[Dot]:
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
                    dot = Dot(port.device, x)
                    devices.append(dot)
                except Exception as e:
                    print(f"No device at address {x}")
            break
    return devices
