import sys
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from PyQt6.QtWidgets import * 

from gui import Ui_MainWindow
import numpy as np
from scipy.signal import savgol_filter

from imu_stream import SerialPort

import pyqtgraph as pg

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
            pass
            
            
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    progress_callback = pyqtSignal(list)
    progress_imu = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setupUi(self)
        self.threadpool = QThreadPool()
        self.cap = cv2.VideoCapture(0)
        # remove axis
        self.graphicsView.getAxis("bottom").setStyle(showValues=False)
        self.graphicsView.getAxis("left").setStyle(showValues=False)
        self.graphicsView.getAxis("top").setStyle(showValues=False)
        self.graphicsView.getAxis("right").setStyle(showValues=False)
                
        self.start_button.clicked.connect(self.start_video)
        self.start_button.clicked.connect(self.imu_sensor)
        
    def camera(self, progress_callback):

        while self.cap:
            ret, frame = self.cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data.tobytes(), w, h, bytesPerLine, QImage.Format.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                progress_callback.emit([p])
                
    def imu_sensor(self):
        progress_imu = pyqtSignal(list)
        _filepath = r"D:\CMC\visualizer\data_dumps"
        ser = SerialPort("COM9", 115200, csv_enable=False, csv_path=_filepath, csv_name="imu01.csv", callback=True)
        worker1 = Worker(ser.run_program, progress_imu)
        worker1.signals.progress.connect(self.imu_update)
        worker1.signals.finished.connect(self.thread_complete)
        worker1.signals.result.connect(self.thread_complete)
        self.threadpool.start(worker1)
    
    def mobbo_sensor(self, progress_callback):
        pass
                
                
    def start_video(self):
        worker = Worker(self.camera)
        worker.signals.progress.connect(self.video_update)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.result.connect(self.thread_complete)
        self.threadpool.start(worker)        

        
    @pyqtSlot(list)
    def video_update(self, img):
        self.video_view.setPixmap(QPixmap.fromImage(img[0]))
        
    @pyqtSlot(list)
    def imu_update(self, data):
        # clear the plot window
        print(data)
                
    def thread_complete(self):
        print("THREAD COMPLETE!")      
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())