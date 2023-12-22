"""This program is for recording IMU data, through HC05 bluetooth module"""

import struct
import keyboard
import csv
from sys import stdout
import time
from visualizer import rs_time
import socket


class MobboCom(object):
    # Contains functions that enable communication between the docking station and the IMU watches

    def __init__(self, port = 23000, csv_path="", csv_enable=False, single_file_protocol=False, csv_name=None):
        # Initialise serial payload
        self.count = 0
        self.plSz = 0
        self.payload = bytearray()

        local_ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        local_ip.connect(("8.8.8.8", 80))

        # Get the local IPv4 address from the socket
        self.local_ip_address = local_ip.getsockname()[0]

        # Define the broadcast ip and port
        self.broadcast_ip = ''.join([ip + "." for ip in self.local_ip_address.split(".")[:-1]]) + "255" # getting broadcast ip address
        self.broadcast_port = port

        time.sleep(1)
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.udp_socket.settimeout(0.5)

        stdout.write("Initializing mobbo program\n")

    def jedi_read(self, _bytes):
        """returns bool for valid read, also returns the data read"""
        
        if (_bytes[0] == b'\xff') and (_bytes[1] == b'\xff'):
            chksum = 255 + 255
            self.plSz = _bytes[2]
            chksum += self.plSz
            self.payload = _bytes[3:3+self.plSz]
            chksum += sum(self.payload)
            chksum = bytes([chksum % 256])
            _chksum = _bytes[-1]            
            return _chksum == chksum

        return False

    def disconnect(self):
        stdout.write("disconnected\n")

    def run_program(self):

        message = "A"

        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        print(f"Multicast Sent: {message}")
        while True:
            return_message = self.udp_socket.recvfrom(23000)
            print(return_message[0])

            if self.jedi_read(return_message[0]):
                print(f"Multicast Received: {struct.unpack('4l', self.payload[1:])}")
                

if __name__ == '__main__':

    _filepath = r"D:\CMC\visualizer\data_dumps"
    myport = MobboCom(csv_path=_filepath, csv_enable=False)
    myport.run_program()

