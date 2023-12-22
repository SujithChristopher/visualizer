import sys
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np



class WorkerThread(QThread):
    data_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_data = np.arange(100)
        self.y_data = np.zeros(100)

    def run(self):
        while True:
            # Simulate data update (replace this with your data retrieval logic)
            self.y_data = np.roll(self.y_data, -1)
            self.y_data[-1] = np.random.rand()

            self.data_signal.emit(self.y_data)
            self.msleep(10)  # Sleep for 1000 milliseconds (1 second)

class RealTimePlot(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Real-Time Plot with PyQt6 and pyqtgraph')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout(self)

        # Create a PlotWidget from pyqtgraph
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Set up the plot
        self.plot_item = self.plot_widget.plotItem
        self.plot_data_item = self.plot_item.plot(pen=pg.mkPen('b', width=2))
        self.plot_item.showGrid(True, True)

        # Initialize the worker thread
        self.worker_thread = WorkerThread()
        self.worker_thread.data_signal.connect(self.update_plot)

        # Set up a timer to start the worker thread
        self.start_timer = QTimer(self)
        self.start_timer.timeout.connect(self.worker_thread.start)

        # Start the timer to begin updating the plot
        self.start_timer.start(1000)  # Start the timer after 1 second

    def update_plot(self, data):
        self.plot_data_item.setData(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    real_time_plot = RealTimePlot()
    real_time_plot.show()
    sys.exit(app.exec())
