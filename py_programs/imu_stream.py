"""This program is for recording IMU data, through HC05 bluetooth module"""

import serial
import struct
import keyboard
import csv
from sys import stdout
from visualizer import rs_time
from PyQt6.QtCore import *


class SerialPort(object):
    # Contains functions that enable communication between the docking station and the IMU watches

    def __init__(self, serialport, serialrate=9600, csv_path="", csv_enable=False, single_file_protocol=False, csv_name=None, callback=False):
        # Initialise serial payload
        self.count = 0
        self.plSz = 0
        self.payload = bytearray()

        self.serialport = serialport
        self.ser_port = serial.Serial(serialport, serialrate, timeout=0.5)
        self.callback = callback

        self.csv_enabled = csv_enable
        if csv_enable:
            if csv_name is None:
                self.csv_file = open(csv_path+ "//imu01.csv", "w")
            else:
                self.csv_file = open(csv_path+ "//" + str(csv_name) + ".csv", "w")
                
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(['sys_time', 'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1', 'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2', 'sync'])
        self.triggered = True
        self.connected = False    

        stdout.write("Initializing imu program\n")


    def serial_read(self):
        """returns bool for valid read, also returns the data read"""
        if (self.ser_port.read() == b'\xff') and (self.ser_port.read() == b'\xff'):
            self.connected = True
            chksum = 255 + 255
            self.plSz = self.ser_port.read()[0]
            chksum += self.plSz
            self.payload = self.ser_port.read(self.plSz - 1)
            chksum += sum(self.payload)
            chksum = bytes([chksum % 256])
            _chksum = self.ser_port.read()

            return _chksum == chksum
        return False

    def disconnect(self):
        stdout.write("disconnected\n")

    def run_program(self, progress_imu=None):
        while self.ser_port.is_open:
            if self.serial_read():
                
                val = struct.unpack("13h", self.payload)    # two imu values + sync
                print(val[-1])
                rs = rs_time()
                if self.csv_enabled:
                    self.csv.writerow([rs, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9], val[10], val[11], val[12]])
                if keyboard.is_pressed("e"):
                    self.csv_file.close()
                    break
            if keyboard.is_pressed("q"):
                print("closing")
                break
            
            if not self.ser_port.is_open:
                print("port closed")
                break
            
            if self.callback:
                progress_imu.emit([1,2,3])
            
        print("program ended")


if __name__ == '__main__':

    _filepath = r"D:\CMC\visualizer\data_dumps"
    myport = SerialPort("COM9", 115200, csv_path=_filepath, csv_enable=True)
    myport.run_program()

