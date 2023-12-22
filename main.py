import sys
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from PyQt6.QtWidgets import * 

from py_programs.gui import Ui_MainWindow
import numpy as np

import serial
import struct
import pyqtgraph as pg
import time
import socket
import os
import toml
import csv
from visualizer import rs_time
import polars as pl
from numba import njit

@njit
def roll(array):
    return np.roll(array, -1)

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(list)
    progress = pyqtSignal(list)

class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            pass
        else:
            pass
        finally:
            print('Thread completed')
            self.signals.finished.emit()  # Done
            
            
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    progress_callback = pyqtSignal(list)
    progress_imu = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setupUi(self)
        self.threadpool = QThreadPool()
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.record_trigger = False
        self.radio_trigger = 1
        self.terminate_trigger = False
        self.offset_trigger = [False, False, False, False, False, False]
        
        self.mobbo_offset = {'m1':[0, 0, 0, 0], 'm2':[0, 0, 0, 0], 'm3':[0, 0, 0, 0], 'm4':[0, 0, 0, 0], 'm5':[0, 0, 0, 0], 'm6':[0, 0, 0, 0]}
        
        # importing calibration excel sheet for loadcells
        self.calib_df = pl.read_csv(os.path.join(os.getcwd(), 'calibration','calib.csv'))
        self.calib_finder = {'m1':[1, 2, 3, 4], 'm2':[5, 6, 7, 8], 'm3':[9, 10, 11, 12], 'm4':[13, 14, 15, 16], 'm5':[17, 18, 19, 20], 'm6':[21, 22, 23, 24]}      
        
        # remove axis
        self.graphicsView.getAxis("bottom").setStyle(showValues=False)
        self.graphicsView.getAxis("left").setStyle(showValues=False)
        self.graphicsView.getAxis("top").setStyle(showValues=False)
        self.graphicsView.getAxis("right").setStyle(showValues=False)
        
        self.graphicsView_2.getAxis("bottom").setStyle(showValues=False)
        self.graphicsView_2.getAxis("left").setStyle(showValues=False)
        self.graphicsView_2.getAxis("top").setStyle(showValues=False)
        self.graphicsView_2.getAxis("right").setStyle(showValues=False)
        
        self.graphicsView_3.getAxis("bottom").setStyle(showValues=False)
        self.graphicsView_3.getAxis("left").setStyle(showValues=False)
        self.graphicsView_3.getAxis("top").setStyle(showValues=False)
        self.graphicsView_3.getAxis("right").setStyle(showValues=False)
        
        self.init_arrays()
        
        self.red_pen = pg.mkPen(color=(255, 0, 0), width=3)
        self.green_pen = pg.mkPen(color=(0, 255, 0), width=3)
        self.blue_pen = pg.mkPen(color=(0, 255, 255), width=3)
        self.orange_pen = pg.mkPen(color=(255, 165, 0), width=3)

        # IMU data plot IMU1
        self.data_line1_1 = self.graphicsView.plot(self.imu_x, self.imu1_ax, pen=self.red_pen, name='ax')
        self.data_line1_2 = self.graphicsView.plot(self.imu_x, self.imu1_ay, pen=self.green_pen, name='ay')
        self.data_line1_3 = self.graphicsView.plot(self.imu_x, self.imu1_az, pen=self.blue_pen, name='az')

        # IMU data plot IMU2
        self.data_line2_1 = self.graphicsView_2.plot(self.imu_x, self.imu2_ax, pen=self.red_pen)
        self.data_line2_2 = self.graphicsView_2.plot(self.imu_x, self.imu2_ay, pen=self.green_pen)
        self.data_line2_3 = self.graphicsView_2.plot(self.imu_x, self.imu2_az, pen=self.blue_pen)
        
        # Mobbo data plot
        self.data_line3_1 = self.graphicsView_3.plot(self.imu_x, self.mobbo_l1, pen=self.red_pen)
        self.data_line3_2 = self.graphicsView_3.plot(self.imu_x, self.mobbo_l2, pen=self.green_pen)
        self.data_line3_3 = self.graphicsView_3.plot(self.imu_x, self.mobbo_l3, pen=self.blue_pen)
        self.data_line3_4 = self.graphicsView_3.plot(self.imu_x, self.mobbo_l4, pen=self.orange_pen)
                
        self.start_button.clicked.connect(self.start_video)
        self.record_button.clicked.connect(self.create_folders)
        
        self.reconnect.clicked.connect(self.resend_mobbo)
        
        # radio buttons
        # r_m1 is always checked
        self.r_m1.setChecked(True)        
        self.r_m1.clicked.connect(self.radio_check)
        self.r_m2.clicked.connect(self.radio_check)
        self.r_m3.clicked.connect(self.radio_check)
        self.r_m4.clicked.connect(self.radio_check)
        self.r_m5.clicked.connect(self.radio_check)
        self.r_m6.clicked.connect(self.radio_check)
        
        time.sleep(1)
        self.start_button.clicked.connect(self.start_threads)
        
        
        # Create a UDP socket
        self.local_ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.local_ip.connect(("8.8.8.8", 80))

        # Get the local IPv4 address from the socket
        self.local_ip_address = self.local_ip.getsockname()[0]

        # Define the broadcast ip and port
        self.broadcast_ip = ''.join([ip + "." for ip in self.local_ip_address.split(".")[:-1]]) + "255" # getting broadcast ip address
        self.broadcast_port = 23000

        time.sleep(1)
        ttl = 1
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, ttl)
        
        self.data_pth_file = os.path.join(os.getcwd(), 'data_pth.toml')
        # read the toml file
        self.data_pth = toml.load(self.data_pth_file)['files']['pth']
        
        self.mobbo_csv_trigger = [False, False, False, False, False, False]
        self.mobbo_csv_writer = [None, None, None, None, None, None]
        
    def radio_check(self):
        if self.r_m1.isChecked():
            self.radio_trigger = 1
        elif self.r_m2.isChecked():
            self.radio_trigger = 2    
        elif self.r_m3.isChecked():
            self.radio_trigger = 3
        elif self.r_m4.isChecked():
            self.radio_trigger = 4
        elif self.r_m5.isChecked():
            self.radio_trigger = 5
        elif self.r_m6.isChecked():
            self.radio_trigger = 6
            
        self.mobbo_l1 = np.zeros(1000)
        self.mobbo_l2 = np.zeros(1000)
        self.mobbo_l3 = np.zeros(1000)
        self.mobbo_l4 = np.zeros(1000)   
        
         
    def create_folders(self):
        dir_list = os.listdir(self.data_pth)
        if len(dir_list) == 0:
            dir_name = '1'
        else:
            dir_name = str(len(dir_list) + 1)
        
        self.data_pth = os.path.join(self.data_pth, dir_name)
        os.mkdir(self.data_pth)
        
        # open video file
        self.video_file = cv2.VideoWriter(os.path.join(self.data_pth, 'video.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))   
        
        # create csv files for IMU and Mobbo
        self.imu_file = open(os.path.join(self.data_pth, 'imu.csv'), 'w')
        self.imu_writer = csv.writer(self.imu_file, lineterminator='\n')
        self.imu_writer.writerow(['timestamp','imu1_ax', 'imu1_ay', 'imu1_az', 'imu1_gx', 'imu1_gy', 'imu1_gz', 'imu2_ax', 'imu2_ay', 'imu2_az', 'imu2_gx', 'imu2_gy', 'imu2_gz', 'sync'])
        self.record_trigger = True
        
        # video timestamp
        self.video_timestamp_file = open(os.path.join(self.data_pth, 'video_timestamp.csv'), 'w')
        self.video_time_writer = csv.writer(self.video_timestamp_file, lineterminator='\n')
        self.video_time_writer.writerow(['timestamp'])
        
                
    def camera(self, progress_callback):

        while self.cap:
            ret, frame = self.cap.read()
            # print(frame)
            if ret:
                # print(frame.shape)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data.tobytes(), w, h, bytesPerLine, QImage.Format.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                progress_callback.emit([p])
                
                if self.record_trigger:
                    frame = cv2.resize(frame, (640, 480))
                    self.video_file.write(frame)
                    self.video_time_writer.writerow([rs_time()])
                if self.terminate_trigger:
                    self.video_file.release()
                    self.video_timestamp_file.close()
                    break
                    
                
    def init_arrays(self):
        self.imu_x = np.arange(1000)
        self.imu1_ax = np.zeros(1000)
        self.imu1_ay = np.zeros(1000)
        self.imu1_az = np.zeros(1000)
        self.imu2_ax = np.zeros(1000)
        self.imu2_ay = np.zeros(1000)
        self.imu2_az = np.zeros(1000)
        
        self.mobbo_l1 = np.zeros(1000)
        self.mobbo_l2 = np.zeros(1000)
        self.mobbo_l3 = np.zeros(1000)
        self.mobbo_l4 = np.zeros(1000)
        
    
    def mobbo_start(self, progress_callback):
        message = "A"

        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        # time.sleep(0.1)
        # self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        # time.sleep(0.5)
        # self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        # time.sleep(0.5)
        print(f"Multicast Sent: {message}")
        try:
            while True:
                # Send the multicast message
                if self.terminate_trigger:
                    break
                return_message = self.udp_socket.recvfrom(23000) 
                progress_callback.emit([return_message[0]])
        except KeyboardInterrupt:
            print("Multicast sender stopped by user.")
        finally:
            # Close the socket when done
            self.udp_socket.close()
        
    def closeEvent(self, a0: QCloseEvent) -> None:
        print('closing')
        self.terminate_trigger = True
        return super().closeEvent(a0)
    
    
    def init_serial(self):
        
        self.plSz = 0
        self.payload = bytearray()
        self.serialport = 'COM9'
        self.serialrate = 115200
        self.ser_port = serial.Serial(self.serialport, self.serialrate, timeout=0.5)
         
    
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
    
    def imu_start(self, progress_callback):
        
        self.init_serial()

        while self.ser_port.is_open:
            
            if self.terminate_trigger:
                self.ser_port.close()
                break
            
            if self.serial_read():
                val = struct.unpack("13h", self.payload)    # two imu values + sync
                progress_callback.emit(val)
                    
        print("program ended")
                
        
    def start_threads(self):
        self.imu_thread = Worker(self.imu_start)
        self.imu_thread.signals.progress.connect(self.imu_update)
        self.imu_thread.signals.finished.connect(self.thread_complete)
        self.imu_thread.signals.result.connect(self.thread_complete)
        self.threadpool.start(self.imu_thread)
        
        self.mobbo_thread = Worker(self.mobbo_start)
        self.mobbo_thread.signals.progress.connect(self.mobbo_update)
        self.mobbo_thread.signals.finished.connect(self.thread_complete)
        self.mobbo_thread.signals.result.connect(self.thread_complete)
        self.threadpool.start(self.mobbo_thread)
                
    def start_video(self):
        self.video_thread = Worker(self.camera)
        self.video_thread.signals.progress.connect(self.video_update)
        self.video_thread.signals.finished.connect(self.thread_complete)
        self.video_thread.signals.result.connect(self.thread_complete)
        self.threadpool.start(self.video_thread)
    
    
    def resend_mobbo(self):
        message = "A"

        self.udp_socket.sendto(message.encode(), (self.broadcast_ip, self.broadcast_port))
        print(f"Multicast Sent: {message}")
    
    @pyqtSlot(list)
    def mobbo_update(self, data):
        # clear the plot window       
        
        self.active_mobbo = data[0][3]
        self.mobbo_select = self.calib_finder[f'm{self.active_mobbo}']
        # print(self.active_mobbo)

        
        adc1 = struct.unpack('l', data[0][4:8])[0]
        adc2 = struct.unpack('l', data[0][8:12])[0]
        adc3 = struct.unpack('l', data[0][12:16])[0]
        adc4 = struct.unpack('l', data[0][16:20])[0]
        # print(adc1, adc2, adc3, adc4)
        
        if not self.offset_trigger[self.active_mobbo - 1]:
            self.mobbo_offset['m'+str(self.active_mobbo)][0] = (adc1 - self.calib_df['lc_offset'][self.mobbo_select[0] -1])/self.calib_df['lc_scalar'][self.mobbo_select[0]-1]
            self.mobbo_offset['m'+str(self.active_mobbo)][1] = (adc2 - self.calib_df['lc_offset'][self.mobbo_select[1] -1])/self.calib_df['lc_scalar'][self.mobbo_select[1]-1]
            self.mobbo_offset['m'+str(self.active_mobbo)][2] = (adc3 - self.calib_df['lc_offset'][self.mobbo_select[2] -1])/self.calib_df['lc_scalar'][self.mobbo_select[2]-1]
            self.mobbo_offset['m'+str(self.active_mobbo)][3] = (adc4 - self.calib_df['lc_offset'][self.mobbo_select[3] -1])/self.calib_df['lc_scalar'][self.mobbo_select[3]-1]
            self.offset_trigger[self.active_mobbo - 1] = True
            
        l1 = (adc1 - self.calib_df['lc_offset'][self.mobbo_select[0] -1])/self.calib_df['lc_scalar'][self.mobbo_select[0] -1] - self.mobbo_offset['m'+str(self.active_mobbo)][0]
        l2 = (adc2 - self.calib_df['lc_offset'][self.mobbo_select[1] -1])/self.calib_df['lc_scalar'][self.mobbo_select[1] -1] - self.mobbo_offset['m'+str(self.active_mobbo)][1]
        l3 = (adc3 - self.calib_df['lc_offset'][self.mobbo_select[2] -1])/self.calib_df['lc_scalar'][self.mobbo_select[2] -1] - self.mobbo_offset['m'+str(self.active_mobbo)][2]
        l4 = (adc4 - self.calib_df['lc_offset'][self.mobbo_select[3] -1])/self.calib_df['lc_scalar'][self.mobbo_select[3] -1] - self.mobbo_offset['m'+str(self.active_mobbo)][3]
        
        if self.active_mobbo == self.radio_trigger:
                
            self.mobbo_l1 = roll(self.mobbo_l1)
            self.mobbo_l1[-1] = l1
            self.mobbo_l2 = roll(self.mobbo_l2)
            self.mobbo_l2[-1] = l2
            self.mobbo_l3 = roll(self.mobbo_l3)
            self.mobbo_l3[-1] = l3
            self.mobbo_l4 = roll(self.mobbo_l4)
            self.mobbo_l4[-1] = l4
            self.data_line3_1.setData(self.imu_x, self.mobbo_l1, pen=self.red_pen)
            self.data_line3_2.setData(self.imu_x, self.mobbo_l2, pen=self.green_pen)
            self.data_line3_3.setData(self.imu_x, self.mobbo_l3, pen=self.blue_pen)
            self.data_line3_4.setData(self.imu_x, self.mobbo_l4, pen=self.orange_pen)
                    
        if self.record_trigger:
            if self.mobbo_csv_trigger[self.active_mobbo - 1] == False:
                self.mobbo_csv_trigger[self.active_mobbo - 1] = True
                self.mobbo_file = open(os.path.join(self.data_pth, f'mobbo{self.active_mobbo}.csv'), 'w')
                self.mobbo_writer = csv.writer(self.mobbo_file, lineterminator='\n')
                self.mobbo_writer.writerow(['timestamp', 'f1', 'f2', 'f3', 'f4', 'sync'])
                self.mobbo_csv_writer[self.active_mobbo - 1] = self.mobbo_writer           
            
            self.mobbo_csv_writer[self.active_mobbo - 1].writerow([rs_time(), l1, l2, l3, l4, data[0][20]])

    @pyqtSlot(list)
    def video_update(self, img):
        self.video_view.setPixmap(QPixmap.fromImage(img[0]))
        
    @pyqtSlot(list)
    def imu_update(self, data):
        # clear the plot window
        self.imu1_ax = roll(self.imu1_ax)
        self.imu1_ax[-1] = data[0] / 16384
        self.imu1_ay = roll(self.imu1_ay)
        self.imu1_ay[-1] = data[1] / 16384
        self.imu1_az = roll(self.imu1_az)
        self.imu1_az[-1] = data[2] / 16384
        self.imu2_ax = roll(self.imu2_ax)
        self.imu2_ax[-1] = data[6] / 16384
        self.imu2_ay = roll(self.imu2_ay)
        self.imu2_ay[-1] = data[7] / 16384
        self.imu2_az = roll(self.imu2_az)
        self.imu2_az[-1] = data[8] / 16384
    
        self.data_line1_1.setData(self.imu_x, self.imu1_ax, pen=self.red_pen, name='ax')
        self.data_line1_2.setData(self.imu_x, self.imu1_ay, pen=self.green_pen, name='ay')
        self.data_line1_3.setData(self.imu_x, self.imu1_az, pen=self.blue_pen, name='az')
        self.data_line2_1.setData(self.imu_x, self.imu2_ax)
        self.data_line2_2.setData(self.imu_x, self.imu2_ay)
        self.data_line2_3.setData(self.imu_x, self.imu2_az)
        
        acc1 = np.array(data[0:3]) / 16384
        acc2 = np.array(data[6:9]) / 16384
        gyro1 = np.array(data[3:6]) / 65.5
        gyro2 = np.array(data[9:12]) / 65.5
                
        data_write = [rs_time(), acc1[0], acc1[1], acc1[2], gyro1[0], gyro1[1], 
                      gyro1[2], acc2[0], acc2[1], acc2[2], gyro2[0], gyro2[1], 
                      gyro2[2], data[12]]
        
        if self.record_trigger:
            self.imu_writer.writerow(data_write)

                
    def thread_complete(self):
        print("THREAD COMPLETED!")      
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())