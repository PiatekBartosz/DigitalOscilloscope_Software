import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer

class Oscilloscope(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope with Controls")
        self.resize(900, 500)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Plot widget
        self.plotWidget = pg.PlotWidget()
        main_layout.addWidget(self.plotWidget)
        self.curve = self.plotWidget.plot(pen='y')

        # Trigger level line
        self.trigger_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=1.5))
        self.plotWidget.addItem(self.trigger_line)

        # Controls layout
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Gain control (slider)
        gain_layout = QVBoxLayout()
        controls_layout.addLayout(gain_layout)
        gain_label = QLabel("Gain")
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(1, 20)  # 0.1 to 2.0 gain scaled by 10
        self.gain_slider.setValue(10)      # default 1.0
        self.gain_slider.valueChanged.connect(self.on_gain_change)
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(self.gain_slider)

        # Offset control (slider)
        offset_layout = QVBoxLayout()
        controls_layout.addLayout(offset_layout)
        offset_label = QLabel("Offset")
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-1000, 1000)  # offset in sample units
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self.on_offset_change)
        offset_layout.addWidget(offset_label)
        offset_layout.addWidget(self.offset_slider)

        # AC/DC coupling checkbox
        coupling_layout = QVBoxLayout()
        controls_layout.addLayout(coupling_layout)
        coupling_label = QLabel("AC Coupling")
        self.ac_coupling_checkbox = QCheckBox()
        self.ac_coupling_checkbox.setChecked(False)
        coupling_layout.addWidget(coupling_label)
        coupling_layout.addWidget(self.ac_coupling_checkbox)

        # Trigger level slider
        trigger_layout = QVBoxLayout()
        controls_layout.addLayout(trigger_layout)
        trigger_label = QLabel("Trigger Level")
        self.trigger_slider = QSlider(Qt.Horizontal)
        self.trigger_slider.setRange(-1500, 1500)
        self.trigger_slider.setValue(0)
        self.trigger_slider.valueChanged.connect(self.on_trigger_change)
        trigger_layout.addWidget(trigger_label)
        trigger_layout.addWidget(self.trigger_slider)

        # Internal state
        self.gain = 1.0
        self.offset = 0
        self.trigger_level = 0
        self.ac_coupling = False
        self.ptr = 0
        self.data_len = 1024

        # Timer to update plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)  # 50 FPS approx

    def on_gain_change(self):
        self.gain = self.gain_slider.value() / 10.0

    def on_offset_change(self):
        self.offset = self.offset_slider.value()

    def on_trigger_change(self):
        self.trigger_level = self.trigger_slider.value()
        self.trigger_line.setValue(self.trigger_level)

    def update_plot(self):
        # Generate mock sine wave + noise
        t = np.linspace(0, 2*np.pi, self.data_len)
        freq = 5  # Hz frequency
        sine_wave = np.sin(freq * t + self.ptr / 10)
        noise = np.random.normal(0, 0.1, self.data_len)
        samples = sine_wave + noise

        # Apply gain and offset
        samples = samples * self.gain * 1000  # scale amplitude
        samples = samples + self.offset

        # Apply AC coupling (remove DC offset)
        if self.ac_coupling_checkbox.isChecked():
            samples = samples - np.mean(samples)

        self.curve.setData(samples)

        # Update trigger line position (already done in slider callback)
        self.trigger_line.setValue(self.trigger_level)

        self.ptr += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    osc = Oscilloscope()
    osc.show()
    sys.exit(app.exec_())
