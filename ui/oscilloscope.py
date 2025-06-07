import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QRadioButton, QButtonGroup, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from utils.controls import create_dial_widget

class Oscilloscope(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope")
        self.resize(1000, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        with open("style/style.qss", "r") as f:
            self.setStyleSheet(f.read())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Plot area
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.showGrid(x=True, y=True, alpha=0.5)
        self.curve = self.plotWidget.plot(pen='y')
        self.trigger_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=1.5))
        self.plotWidget.addItem(self.trigger_line)
        main_layout.addWidget(self.plotWidget, stretch=4)

        # Controls area
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        controls_frame.setMaximumWidth(280)
        main_layout.addWidget(controls_frame, stretch=1)

        self.gain_dial, self.gain_edit = create_dial_widget("Gain", 1, 20, 10, controls_layout, self.on_gain_change)
        self.offset_dial, self.offset_edit = create_dial_widget("Offset", -1000, 1000, 0, controls_layout, self.on_offset_change)
        self.trigger_dial, self.trigger_edit = create_dial_widget("Trigger Level", -1500, 1500, 0, controls_layout, self.on_trigger_change)
        self.timebase_dial, self.timebase_edit = create_dial_widget("Timebase", 1, 50, 10, controls_layout, self.on_timebase_change)
        self.vpos_dial, self.vpos_edit = create_dial_widget("Vert Pos", -500, 500, 0, controls_layout, self.on_vpos_change)

        # Trigger mode
        trigger_mode_label = QLabel("Trigger Mode")
        self.trigger_mode_combo = QComboBox()
        self.trigger_mode_combo.addItems(["Auto", "Normal", "Single"])
        self.trigger_mode_combo.currentTextChanged.connect(self.on_trigger_mode_change)
        controls_layout.addWidget(trigger_mode_label)
        controls_layout.addWidget(self.trigger_mode_combo)

        # Coupling
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

        # Run/Stop
        self.run_btn = QPushButton("Stop")
        self.run_btn.setCheckable(True)
        self.run_btn.toggled.connect(self.toggle_run)
        controls_layout.addWidget(self.run_btn)

        controls_layout.addStretch()

        # Internal state
        self.gain = 1.0
        self.offset = 0
        self.trigger_level = 0
        self.ac_coupling = False
        self.ptr = 0
        self.data_len = 1024
        self.timebase = 10
        self.vpos = 0
        self.running = True
        self.trigger_mode = "Auto"

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)

    def on_gain_change(self): self.gain = self.gain_dial.value() / 10.0
    def on_offset_change(self): self.offset = self.offset_dial.value()
    def on_trigger_change(self):
        self.trigger_level = self.trigger_dial.value()
        self.trigger_line.setValue(self.trigger_level)
    def on_timebase_change(self): self.timebase = self.timebase_dial.value()
    def on_vpos_change(self): self.vpos = self.vpos_dial.value()
    def on_trigger_mode_change(self, mode): self.trigger_mode = mode
    def on_coupling_change(self, button): self.ac_coupling = (button.text() == "AC")
    def toggle_run(self, checked):
        if checked:
            self.timer.stop()
            self.run_btn.setText("Run")
        else:
            self.timer.start(20)
            self.run_btn.setText("Stop")

    def update_plot(self):
        if not self.running:
            return
        t = np.linspace(0, 2 * np.pi, self.data_len)
        freq = 5 / (self.timebase / 10.0)
        sine_wave = np.sin(freq * t + self.ptr / 10)
        noise = np.random.normal(0, 0.1, self.data_len)
        samples = sine_wave + noise

        samples *= self.gain * 1000
        samples += self.offset + self.vpos
        if self.ac_coupling:
            samples -= np.mean(samples)

        self.curve.setData(samples)
        self.trigger_line.setValue(self.trigger_level)
        self.ptr += 1

