import os
import queue
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QRadioButton, QButtonGroup, QPushButton, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer

from utils.controls import create_dial_widget
from ui.command_panel import CommandPanel

ADC_COUNTS  = 16384
ADC_VREF    = 5.0  


def _raw_to_volts(raw: int) -> float:
    """Convert 14-bit offset-binary ADC code to volts."""
    return (raw - ADC_COUNTS / 2) / (ADC_COUNTS / 2) * ADC_VREF


class Oscilloscope(QMainWindow):
    DISPLAY_SAMPLES = 1024

    def __init__(self, conn_mgr, sample_queue: queue.Queue):
        super().__init__()
        self._conn_mgr     = conn_mgr
        self._sample_queue = sample_queue

        self._ch1_buf = deque([0.0] * self.DISPLAY_SAMPLES,
                              maxlen=self.DISPLAY_SAMPLES)
        self._ch2_buf = deque([0.0] * self.DISPLAY_SAMPLES,
                              maxlen=self.DISPLAY_SAMPLES)

        self._build_ui()

        self.gain           = 1.0
        self.offset         = 0.0
        self.trigger_level  = 0.0
        self.timebase       = 10
        self.vpos           = 0.0
        self.running        = True
        self.trigger_mode   = "Auto"
        self.ac_coupling    = False   # CH1
        self.ac_coupling_ch2 = False  # CH2
        self.ch1_enabled    = True
        self.ch2_enabled    = True

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(20)

    def _build_ui(self):
        self.setWindowTitle("Oscilloscope")
        self.resize(1100, 820)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowStaysOnTopHint)

        _here = os.path.dirname(os.path.abspath(__file__))
        _qss  = os.path.join(_here, "..", "style", "style.qss")
        try:
            with open(_qss, "r") as f:
                self.setStyleSheet(f.read())
        except (FileNotFoundError, OSError):
            pass

        central      = QWidget()
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setCentralWidget(central)

        # ── top: plot + controls ──────────────────────────────────────────────
        top_widget  = QWidget()
        main_layout = QHBoxLayout(top_widget)

        self.plotWidget = pg.PlotWidget()
        self.plotWidget.showGrid(x=True, y=True, alpha=0.5)
        self.plotWidget.setLabel("left",  "Voltage", units="V")
        self.plotWidget.setLabel("bottom","Samples")
        self._curve_ch1   = self.plotWidget.plot(pen='y',  name="CH1")
        self._curve_ch2   = self.plotWidget.plot(pen='c',  name="CH2")
        self._trigger_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen('r', width=1.5))
        self.plotWidget.addItem(self._trigger_line)
        main_layout.addWidget(self.plotWidget, stretch=4)

        ctrl_frame  = QFrame()
        ctrl_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_frame.setMaximumWidth(300)
        main_layout.addWidget(ctrl_frame, stretch=1)

        self._gain_dial, _  = create_dial_widget(
            "Gain (×0.1)", 1, 100, 10, ctrl_layout, self._on_gain_change)
        self._offset_dial, _ = create_dial_widget(
            "Offset (mV)", -2000, 2000, 0, ctrl_layout, self._on_offset_change)
        self._trigger_dial, _ = create_dial_widget(
            "Trigger Level (mV)", -2000, 2000, 0, ctrl_layout,
            self._on_trigger_change)
        self._timebase_dial, _ = create_dial_widget(
            "Timebase", 1, 50, 10, ctrl_layout, self._on_timebase_change)
        self._vpos_dial, _ = create_dial_widget(
            "Vert Pos (mV)", -500, 500, 0, ctrl_layout, self._on_vpos_change)

        ctrl_layout.addWidget(QLabel("Trigger Mode"))
        self._trig_mode_combo = QComboBox()
        self._trig_mode_combo.addItems(["Auto", "Normal", "Single"])
        self._trig_mode_combo.currentTextChanged.connect(
            self._on_trigger_mode_change)
        ctrl_layout.addWidget(self._trig_mode_combo)

        ctrl_layout.addWidget(QLabel("CH1 Coupling"))
        self._dc_radio = QRadioButton("DC")
        self._ac_radio = QRadioButton("AC")
        self._dc_radio.setChecked(True)
        self._coupling_group = QButtonGroup()
        self._coupling_group.addButton(self._dc_radio)
        self._coupling_group.addButton(self._ac_radio)
        self._coupling_group.buttonClicked.connect(self._on_coupling_change)
        ctrl_layout.addWidget(self._dc_radio)
        ctrl_layout.addWidget(self._ac_radio)

        ctrl_layout.addWidget(QLabel("CH1 Attenuation"))
        self._atten_1_radio   = QRadioButton("1:1")
        self._atten_100_radio = QRadioButton("1:100")
        self._atten_1_radio.setChecked(True)
        self._atten_group = QButtonGroup()
        self._atten_group.addButton(self._atten_1_radio)
        self._atten_group.addButton(self._atten_100_radio)
        self._atten_group.buttonClicked.connect(self._on_attenuation_change)
        ctrl_layout.addWidget(self._atten_1_radio)
        ctrl_layout.addWidget(self._atten_100_radio)

        ctrl_layout.addWidget(QLabel("CH2 Coupling"))
        self._ch2_dc_radio = QRadioButton("DC")
        self._ch2_ac_radio = QRadioButton("AC")
        self._ch2_dc_radio.setChecked(True)
        self._ch2_coupling_group = QButtonGroup()
        self._ch2_coupling_group.addButton(self._ch2_dc_radio)
        self._ch2_coupling_group.addButton(self._ch2_ac_radio)
        self._ch2_coupling_group.buttonClicked.connect(self._on_ch2_coupling_change)
        ctrl_layout.addWidget(self._ch2_dc_radio)
        ctrl_layout.addWidget(self._ch2_ac_radio)

        ctrl_layout.addWidget(QLabel("CH2 Attenuation"))
        self._ch2_atten_1_radio   = QRadioButton("1:1")
        self._ch2_atten_100_radio = QRadioButton("1:100")
        self._ch2_atten_1_radio.setChecked(True)
        self._ch2_atten_group = QButtonGroup()
        self._ch2_atten_group.addButton(self._ch2_atten_1_radio)
        self._ch2_atten_group.addButton(self._ch2_atten_100_radio)
        self._ch2_atten_group.buttonClicked.connect(self._on_ch2_attenuation_change)
        ctrl_layout.addWidget(self._ch2_atten_1_radio)
        ctrl_layout.addWidget(self._ch2_atten_100_radio)

        ctrl_layout.addWidget(QLabel("Trigger Coupling"))
        self._trig_dc_radio = QRadioButton("DC")
        self._trig_ac_radio = QRadioButton("AC")
        self._trig_dc_radio.setChecked(True)
        self._trig_coup_group = QButtonGroup()
        self._trig_coup_group.addButton(self._trig_dc_radio)
        self._trig_coup_group.addButton(self._trig_ac_radio)
        self._trig_coup_group.buttonClicked.connect(
            self._on_trigger_coupling_change)
        ctrl_layout.addWidget(self._trig_dc_radio)
        ctrl_layout.addWidget(self._trig_ac_radio)

        self._interleaved_btn = QPushButton("Interleaved: OFF")
        self._interleaved_btn.setCheckable(True)
        self._interleaved_btn.toggled.connect(self._on_interleaved_change)
        ctrl_layout.addWidget(self._interleaved_btn)

        self._status_label = QLabel("Connecting…")
        ctrl_layout.addWidget(self._status_label)

        if self._conn_mgr:
            self._conn_mgr.connected.connect(
                lambda: self._status_label.setText("Connected"))
            self._conn_mgr.disconnected.connect(
                lambda: self._status_label.setText("Disconnected"))
            self._conn_mgr.connecting.connect(
                lambda: self._status_label.setText("Connecting…"))
            self._conn_mgr.device_found.connect(
                lambda addr: self._status_label.setText(f"Found: {addr}"))

        self._run_btn = QPushButton("Stop")
        self._run_btn.setCheckable(True)
        self._run_btn.toggled.connect(self._toggle_run)
        ctrl_layout.addWidget(self._run_btn)

        ch_row = QHBoxLayout()
        self._ch1_btn = QPushButton("CH1: ON")
        self._ch1_btn.setCheckable(True)
        self._ch1_btn.setChecked(True)
        self._ch1_btn.toggled.connect(self._on_ch1_toggle)
        ch_row.addWidget(self._ch1_btn)
        self._ch2_btn = QPushButton("CH2: ON")
        self._ch2_btn.setCheckable(True)
        self._ch2_btn.setChecked(True)
        self._ch2_btn.toggled.connect(self._on_ch2_toggle)
        ch_row.addWidget(self._ch2_btn)
        ctrl_layout.addLayout(ch_row)

        ctrl_layout.addStretch()

        # ── bottom: command console ───────────────────────────────────────────
        self._cmd_panel = CommandPanel()
        self._cmd_panel.command_submitted.connect(self._send)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(self._cmd_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 220])
        outer_layout.addWidget(splitter)

        # wire connection state + firmware replies → console log
        if self._conn_mgr:
            self._conn_mgr.connected.connect(
                lambda: self._cmd_panel.log_ok("Connected"))
            self._conn_mgr.disconnected.connect(
                lambda: self._cmd_panel.log_error("Disconnected"))
            self._conn_mgr.connecting.connect(
                lambda: self._cmd_panel.log_info("Connecting…"))
            self._conn_mgr.device_found.connect(
                lambda addr: self._cmd_panel.log_ok(f"Device found: {addr}"))
            self._conn_mgr.response_received.connect(self._on_firmware_response)

    def _send(self, cmd: str):
        if self._conn_mgr:
            self._conn_mgr.send_command(cmd)

    def _on_firmware_response(self, line: str):
        if line.startswith("ERR"):
            self._cmd_panel.log_error(line)
        else:
            self._cmd_panel.log_ok(line)

    def _on_gain_change(self):
        self.gain = self._gain_dial.value() / 10.0

    def _on_offset_change(self):
        self.offset = self._offset_dial.value() / 1000.0

    def _on_trigger_change(self):
        self.trigger_level = self._trigger_dial.value() / 1000.0
        self._trigger_line.setValue(self.trigger_level)

    def _on_timebase_change(self):
        self.timebase = self._timebase_dial.value()

    def _on_vpos_change(self):
        self.vpos = self._vpos_dial.value() / 1000.0

    def _on_trigger_mode_change(self, mode: str):
        self.trigger_mode = mode

    def _on_coupling_change(self, button):
        coupling = "ac" if button.text() == "AC" else "dc"
        self.ac_coupling = (coupling == "ac")
        self._send(f"afe coupling 1 {coupling}")

    def _on_attenuation_change(self, button):
        atten = "100" if button.text() == "1:100" else "1"
        self._send(f"afe atten 1 {atten}")

    def _on_ch2_coupling_change(self, button):
        coupling = "ac" if button.text() == "AC" else "dc"
        self.ac_coupling_ch2 = (coupling == "ac")
        self._send(f"afe coupling 2 {coupling}")

    def _on_ch2_attenuation_change(self, button):
        atten = "100" if button.text() == "1:100" else "1"
        self._send(f"afe atten 2 {atten}")

    def _on_trigger_coupling_change(self, button):
        coupling = "ac" if button.text() == "AC" else "dc"
        self._send(f"afe trigger {coupling}")

    def _on_ch1_toggle(self, checked: bool):
        self.ch1_enabled = checked
        self._ch1_btn.setText(f"CH1: {'ON' if checked else 'OFF'}")
        self._curve_ch1.setVisible(checked)

    def _on_ch2_toggle(self, checked: bool):
        self.ch2_enabled = checked
        self._ch2_btn.setText(f"CH2: {'ON' if checked else 'OFF'}")
        self._curve_ch2.setVisible(checked)

    def _on_interleaved_change(self, checked: bool):
        self._interleaved_btn.setText(
            f"Interleaved: {'ON' if checked else 'OFF'}")
        self._send(f"afe interleaved {1 if checked else 0}")

    def _toggle_run(self, checked: bool):
        self.running = not checked
        if checked:
            self._timer.stop()
            self._run_btn.setText("Run")
        else:
            self._timer.start(20)
            self._run_btn.setText("Stop")

    def _update_plot(self):
        if not self.running:
            return

        drained = 0
        while drained < 512:
            try:
                ch1_raw, ch2_raw = self._sample_queue.get_nowait()
                self._ch1_buf.append(_raw_to_volts(ch1_raw))
                self._ch2_buf.append(_raw_to_volts(ch2_raw))
                drained += 1
            except queue.Empty:
                break

        ch1 = np.array(self._ch1_buf, dtype=np.float32)
        ch2 = np.array(self._ch2_buf, dtype=np.float32)

        ch1 = ch1 * self.gain + self.offset + self.vpos
        ch2 = ch2 * self.gain + self.offset + self.vpos

        if self.ac_coupling:
            ch1 -= np.mean(ch1)
        if self.ac_coupling_ch2:
            ch2 -= np.mean(ch2)

        if self.ch1_enabled:
            self._curve_ch1.setData(ch1)
        if self.ch2_enabled:
            self._curve_ch2.setData(ch2)
        self._trigger_line.setValue(self.trigger_level)
