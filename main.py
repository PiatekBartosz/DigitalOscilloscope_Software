import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QRadioButton, QButtonGroup, QFrame
)
from PyQt5.QtCore import Qt, QTimer

class Oscilloscope(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope")
        self.resize(1000, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #eeeeee;
                font-family: Consolas, monospace;
                font-size: 12pt;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 6px;
                background: #333;
                margin: 0px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ffff;
                border: 1px solid #00dddd;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                background-color: #00ffff;
                border: 1px solid #00dddd;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #aaa;
                background-color: #444;
            }
        """)

        # Main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Plot area
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.showGrid(x=True, y=True, alpha=0.5)
        self.curve = self.plotWidget.plot(pen='y')
        self.trigger_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=1.5))
        self.plotWidget.addItem(self.trigger_line)
        main_layout.addWidget(self.plotWidget, stretch=4)

        # Controls area
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QVBoxLayout()
        controls_frame.setLayout(controls_layout)
        controls_frame.setMaximumWidth(250)
        main_layout.addWidget(controls_frame, stretch=1)

        # Gain
        gain_label = QLabel("Gain")
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(1, 20)
        self.gain_slider.setValue(10)
        self.gain_slider.valueChanged.connect(self.on_gain_change)
        controls_layout.addWidget(gain_label)
        controls_layout.addWidget(self.gain_slider)

        # Offset
        offset_label = QLabel("Offset")
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-1000, 1000)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self.on_offset_change)
        controls_layout.addWidget(offset_label)
        controls_layout.addWidget(self.offset_slider)

        # Coupling mode
        coupling_label = QLabel("Coupling")
        self.dc_radio = QRadioButton("DC")
        self.ac_radio = QRadioButton("AC")
        self.dc_radio.setChecked(True)
        self.coupling_group = QButtonGroup()
        self.coupling_group.addButton(self.dc_radio)
        self.coupling_group.addButton(self.ac_radio)
        self.coupling_group.buttonClicked.connect(self.on_coupling_change)
        controls_layout.addWidget(coupling_label)
        controls_layout.addWidget(self.dc_radio)
        controls_layout.addWidget(self.ac_radio)

        # Trigger level
        trigger_label = QLabel("Trigger Level")
        self.trigger_slider = QSlider(Qt.Horizontal)
        self.trigger_slider.setRange(-1500, 1500)
        self.trigger_slider.setValue(0)
        self.trigger_slider.valueChanged.connect(self.on_trigger_change)
        controls_layout.addWidget(trigger_label)
        controls_layout.addWidget(self.trigger_slider)

        controls_layout.addStretch()

        # Internal state
        self.gain = 1.0
        self.offset = 0
        self.trigger_level = 0
        self.ac_coupling = False
        self.ptr = 0
        self.data_len = 1024

        # Timer to update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)

    def on_gain_change(self):
        self.gain = self.gain_slider.value() / 10.0

    def on_offset_change(self):
        self.offset = self.offset_slider.value()

    def on_trigger_change(self):
        self.trigger_level = self.trigger_slider.value()
        self.trigger_line.setValue(self.trigger_level)

    def on_coupling_change(self, button):
        self.ac_coupling = (button.text() == "AC")

    def update_plot(self):
        t = np.linspace(0, 2*np.pi, self.data_len)
        freq = 5
        sine_wave = np.sin(freq * t + self.ptr / 10)
        noise = np.random.normal(0, 0.1, self.data_len)
        samples = sine_wave + noise

        samples *= self.gain * 1000
        samples += self.offset

        if self.ac_coupling:
            samples -= np.mean(samples)

        self.curve.setData(samples)
        self.trigger_line.setValue(self.trigger_level)
        self.ptr += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    osc = Oscilloscope()
    osc.show()
    sys.exit(app.exec_())

