import datetime
import logging
import os
import queue

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QComboBox, QPushButton, QSplitter,
    QFileDialog, QDialog, QDialogButtonBox, QPlainTextEdit, QToolButton,
    QScrollArea, QDial, QToolBar, QWidgetAction,
)
from PyQt6.QtCore import QEvent, Qt, QTimer

from utils.controls import create_dial_widget, create_float_dial_widget
from utils.settings import load_settings, save_settings
from ui.command_panel import CommandPanel
from analysis.capture_io import save_capture_csv

logger = logging.getLogger(__name__)

ADC_COUNTS  = 16384
ADC_SAMPLE_RATE_HZ = 80_000_000.0
DISPLAY_VERTICAL_DIVISIONS = 8
DISPLAY_HORIZONTAL_DIVISIONS = 10

AFE_ATTEN_1_TO_1 = 953_000.0 / (49_900.0 + 953_000.0)
AFE_ATTEN_1_TO_100 = 10_000.0 / (1_000_000.0 + 10_000.0)
AFE_VGA_MAX_DB = 24.0
AFE_DIFF_AMP_GAIN = 1.0
ADC_BITS = 14

TIMEBASES = [
    ("10.24 µs/div",   1),
    ("20.48 µs/div",   2),
    ("51.2 µs/div",    5),
    ("102.4 µs/div",  10),
    ("204.8 µs/div",  20),
    ("512 µs/div",    50),
    ("1.024 ms/div", 100),
]
CAPTURE_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


class Oscilloscope(QMainWindow):
    DISPLAY_SAMPLES = 8192

    def __init__(self, conn_mgr, frame_queue: queue.Queue):
        super().__init__()
        self._conn_mgr   = conn_mgr
        self._frame_queue = frame_queue

        self._ch1_data = np.zeros(self.DISPLAY_SAMPLES, dtype=np.float32)
        self._ch2_data = np.zeros(self.DISPLAY_SAMPLES, dtype=np.float32)
        self._ch1_raw = np.zeros(self.DISPLAY_SAMPLES, dtype=np.uint16)
        self._ch2_raw = np.zeros(self.DISPLAY_SAMPLES, dtype=np.uint16)
        self._have_frame = False
        self._afe_state: dict[str, str] = {}
        self._afe_info_text = "AFE state: waiting for firmware status…"
        self._settings = load_settings()
        self._cursor_mode: str | None = None
        self._cursor_points = {"horizontal": [], "vertical": []}

        self.capture_size    = self.DISPLAY_SAMPLES
        self.pretrigger_size = 0
        self.timebase        = TIMEBASES[0][1]
        self._build_ui()

        self._adc_format     = "Offset Binary"

        self.ch1_enabled     = True
        self.ch2_enabled     = True

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

        self._build_cursor_toolbar()

        menu_bar = self.menuBar()
        afe_state_action = menu_bar.addAction("AFE State")
        afe_state_action.triggered.connect(self._show_afe_state)
        menu_bar.addSeparator()
        help_menu = menu_bar.addMenu("&Help")
        afe_calculations_action = help_menu.addAction("AFE configuration calculations")
        afe_calculations_action.triggered.connect(self._show_afe_calculations)
        menu_bar.addSeparator()
        self._scale_label = QLabel()
        self._scale_label.setStyleSheet("padding: 0 8px;")
        scale_action = QWidgetAction(self)
        scale_action.setDefaultWidget(self._scale_label)
        menu_bar.addAction(scale_action)

        self._cmd_panel = CommandPanel()
        self._cmd_panel.command_submitted.connect(self._send)

        top_widget  = QWidget()
        main_layout = QHBoxLayout(top_widget)

        self.plotWidget = pg.PlotWidget()
        self.plotWidget.showGrid(x=True, y=True, alpha=0.4)
        self.plotWidget.setLabel("bottom","Samples")
        self.plotWidget.getAxis("left").setStyle(showValues=False)

        vb = self.plotWidget.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.disableAutoRange()
        self.plotWidget.setMenuEnabled(False)
        self.plotWidget.hideButtons()
        self.plotWidget.setXRange(0, self.DISPLAY_SAMPLES, padding=0)
        self.plotWidget.setYRange(-ADC_COUNTS // 2, ADC_COUNTS // 2, padding=0)
        vb.setLimits(xMin=0, xMax=self.DISPLAY_SAMPLES,
                     yMin=-ADC_COUNTS // 2, yMax=ADC_COUNTS // 2)

        x_step = self.DISPLAY_SAMPLES / 10
        x_ticks = [(x_step * i, str(int(x_step * i))) for i in range(11)]
        y_step = ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
        y_ticks = [(-ADC_COUNTS / 2 + y_step * i, "")
                   for i in range(DISPLAY_VERTICAL_DIVISIONS + 1)]
        self.plotWidget.getAxis('bottom').setTicks([x_ticks])
        self.plotWidget.getAxis('left').setTicks([y_ticks])

        self._curve_ch1   = self.plotWidget.plot(pen='y',  name="CH1")
        self._curve_ch2   = self.plotWidget.plot(pen='c',  name="CH2")
        self._trigger_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("r", width=2),
            movable=False,
        )
        self._trigger_line.setVisible(False)
        self.plotWidget.addItem(self._trigger_line)
        self._cursor_lines = {
            "horizontal": [
                pg.InfiniteLine(angle=0, pen=pg.mkPen("#44ff88", width=2), movable=False),
                pg.InfiniteLine(angle=0, pen=pg.mkPen("#44ff88", width=2), movable=False),
            ],
            "vertical": [
                pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff66ff", width=2), movable=False),
                pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff66ff", width=2), movable=False),
            ],
        }
        for lines in self._cursor_lines.values():
            for line in lines:
                line.setVisible(False)
                line.setZValue(100)
                self.plotWidget.addItem(line)
        self.plotWidget.scene().sigMouseClicked.connect(self._on_plot_clicked)
        main_layout.addWidget(self.plotWidget, stretch=4)

        ctrl_frame  = QFrame()
        ctrl_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ctrl_frame.setMinimumWidth(300)
        ctrl_frame.setMaximumWidth(300)
        ctrl_root_layout = QVBoxLayout(ctrl_frame)
        ctrl_root_layout.setContentsMargins(0, 0, 0, 0)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll = ctrl_scroll
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_root_layout.addWidget(ctrl_scroll)
        main_layout.addWidget(ctrl_frame, stretch=1)

        self._gain_dial, self._gain_value = create_float_dial_widget(
            "CH1 AFE Gain (%)", 0, 100, 50, ctrl_layout, self._on_gain_change)
        self._offset_dial, self._offset_value = create_float_dial_widget(
            "CH1 AFE Offset (%)", 0, 100, 50, ctrl_layout, self._on_offset_change)
        self._ch1_range_combo = self._add_adc_range_control(
            "CH1 ADC Range (differential)", "ch1", ctrl_layout)

        self._gain2_dial, self._gain2_value = create_float_dial_widget(
            "CH2 AFE Gain (%)", 0, 100, 50, ctrl_layout, self._on_gain2_change)
        self._offset2_dial, self._offset2_value = create_float_dial_widget(
            "CH2 AFE Offset (%)", 0, 100, 50, ctrl_layout, self._on_offset2_change)
        self._ch2_range_combo = self._add_adc_range_control(
            "CH2 ADC Range (differential)", "ch2", ctrl_layout)

        trigger_layout = QHBoxLayout()
        trigger_layout.addWidget(QLabel("Trigger Level (AFE %)"))
        trigger_layout.addWidget(self._make_help_button(
            "Trigger level",
            "The trigger percentage is applied to the currently selected "
            "trigger-source channel.\n\n"
            "It is not a percentage of the DAC's full output range. The "
            "firmware converts it to the positive leg of that channel's "
            "differential ADC range, centred at 1.5 V.\n\n"
            "For a 1 Vpp range, 0–100% maps to 1.25–1.75 V. For a 2 Vpp "
            "range, it maps to 1.00–2.00 V. Changing the ADC range therefore "
            "also changes the voltage represented by the trigger level."))
        trigger_layout.addStretch()
        ctrl_layout.addLayout(trigger_layout)
        self._trigger_dial, self._trigger_value = create_float_dial_widget(
            "", 0, 100, 50, ctrl_layout,
            self._on_trigger_change)

        ctrl_layout.addWidget(QLabel("Capture Depth"))
        self._sample_size_combo = QComboBox()
        for count in CAPTURE_SIZES:
            self._sample_size_combo.addItem(f"{count} samples", count)
        self._sample_size_combo.setCurrentIndex(
            self._sample_size_combo.findData(self.capture_size))
        self._sample_size_combo.currentIndexChanged.connect(self._on_sample_size_change)
        ctrl_layout.addWidget(self._sample_size_combo)

        ctrl_layout.addWidget(QLabel("Pretrigger Samples"))
        self._pretrigger_combo = QComboBox()
        self._populate_pretrigger_options()
        self._pretrigger_combo.currentIndexChanged.connect(self._on_pretrigger_change)
        ctrl_layout.addWidget(self._pretrigger_combo)

        timebase_layout = QHBoxLayout()
        timebase_layout.addWidget(QLabel("Timebase (H. Scale)"))
        timebase_layout.addWidget(self._make_help_button(
            "Timebase and frequency",
            self._timebase_help_text()))
        timebase_layout.addStretch()
        ctrl_layout.addLayout(timebase_layout)
        self._timebase_combo = QComboBox()
        for label, factor in TIMEBASES:
            self._timebase_combo.addItem(label, factor)
        self._timebase_combo.currentIndexChanged.connect(self._on_timebase_change)
        ctrl_layout.addWidget(self._timebase_combo)

        ctrl_layout.addWidget(QLabel("ADC Format"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["Offset Binary", "2's Complement"])
        self._format_combo.currentTextChanged.connect(self._on_format_change)
        ctrl_layout.addWidget(self._format_combo)

        ctrl_layout.addWidget(QLabel("CH1 Coupling"))
        self._coupling_combo = QComboBox()
        self._coupling_combo.addItems(["DC", "AC"])
        self._coupling_combo.currentTextChanged.connect(self._on_coupling_change)
        ctrl_layout.addWidget(self._coupling_combo)

        ctrl_layout.addWidget(QLabel("CH1 Attenuation"))
        self._atten_combo = QComboBox()
        self._atten_combo.addItems(["1:1", "1:100"])
        self._atten_combo.currentTextChanged.connect(self._on_attenuation_change)
        ctrl_layout.addWidget(self._atten_combo)

        ctrl_layout.addWidget(QLabel("CH2 Coupling"))
        self._ch2_coupling_combo = QComboBox()
        self._ch2_coupling_combo.addItems(["DC", "AC"])
        self._ch2_coupling_combo.currentTextChanged.connect(self._on_ch2_coupling_change)
        ctrl_layout.addWidget(self._ch2_coupling_combo)

        ctrl_layout.addWidget(QLabel("CH2 Attenuation"))
        self._ch2_atten_combo = QComboBox()
        self._ch2_atten_combo.addItems(["1:1", "1:100"])
        self._ch2_atten_combo.currentTextChanged.connect(self._on_ch2_attenuation_change)
        ctrl_layout.addWidget(self._ch2_atten_combo)

        ctrl_layout.addWidget(QLabel("Trigger Source"))
        self._trigger_source_combo = QComboBox()
        self._trigger_source_combo.addItems(["CH1", "CH2"])
        self._trigger_source_combo.currentTextChanged.connect(self._on_trigger_source_change)
        ctrl_layout.addWidget(self._trigger_source_combo)

        ctrl_layout.addWidget(QLabel("Trigger Mode"))
        self._trigger_mode_combo = QComboBox()
        self._trigger_mode_combo.addItems(["Off", "Normal"])
        self._trigger_mode_combo.currentTextChanged.connect(self._on_trigger_mode_change)
        ctrl_layout.addWidget(self._trigger_mode_combo)

        self._interleaved_btn = QPushButton("Interleaved: OFF")
        self._interleaved_btn.setCheckable(True)
        self._interleaved_btn.toggled.connect(self._on_interleaved_change)
        ctrl_layout.addWidget(self._interleaved_btn)

        self._status_label = QLabel("Connecting…")
        ctrl_layout.addWidget(self._status_label)

        self._refresh_afe_btn = QPushButton("Refresh AFE state")
        self._refresh_afe_btn.clicked.connect(lambda: self._send("status"))
        ctrl_layout.addWidget(self._refresh_afe_btn)

        if self._conn_mgr:
            self._conn_mgr.connected.connect(self._on_connected)
            self._conn_mgr.disconnected.connect(self._on_disconnected)
            self._conn_mgr.connecting.connect(
                lambda: self._status_label.setText("Connecting…"))
            self._conn_mgr.device_found.connect(
                lambda addr: self._status_label.setText(f"Found: {addr}"))
            self._conn_mgr.acquisition_done.connect(self._on_acquisition_done)

        acq_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setCheckable(True)
        self._run_btn.toggled.connect(self._toggle_run)
        acq_row.addWidget(self._run_btn)

        self._single_btn = QPushButton("Single")
        self._single_btn.clicked.connect(self._single_acquire)
        acq_row.addWidget(self._single_btn)
        ctrl_layout.addLayout(acq_row)

        self._save_btn = QPushButton("Save frame")
        self._save_btn.clicked.connect(self._save_frame)
        ctrl_layout.addWidget(self._save_btn)

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

        self._sidebar_wheel_controls = (
            ctrl_widget.findChildren(QComboBox) + ctrl_widget.findChildren(QDial))
        for control in self._sidebar_wheel_controls:
            control.installEventFilter(self)

        self._hardware_controls = [
            self._gain_dial, self._gain_value, self._offset_dial, self._offset_value,
            self._ch1_range_combo, self._gain2_dial, self._gain2_value,
            self._offset2_dial, self._offset2_value, self._ch2_range_combo,
            self._trigger_dial, self._trigger_value, self._sample_size_combo,
            self._pretrigger_combo, self._timebase_combo, self._coupling_combo,
            self._atten_combo, self._ch2_coupling_combo, self._ch2_atten_combo,
            self._trigger_source_combo, self._trigger_mode_combo, self._interleaved_btn,
        ]
        self._set_hardware_controls_enabled(False)

        ctrl_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(self._cmd_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([680, 140])
        outer_layout.addWidget(splitter)
        self._update_scale_display()

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
            self._conn_mgr.acquisition_done.connect(
                lambda: self._cmd_panel.log_info("Single acquisition complete"))

    def _build_cursor_toolbar(self):
        toolbar = QToolBar("Cursors", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._horizontal_cursor_button = QToolButton()
        self._horizontal_cursor_button.setText("Horizontal cursors")
        self._horizontal_cursor_button.setCheckable(True)
        self._horizontal_cursor_button.toggled.connect(
            lambda checked: self._set_cursor_mode("horizontal", checked))
        toolbar.addWidget(self._horizontal_cursor_button)

        self._vertical_cursor_button = QToolButton()
        self._vertical_cursor_button.setText("Vertical cursors")
        self._vertical_cursor_button.setCheckable(True)
        self._vertical_cursor_button.toggled.connect(
            lambda checked: self._set_cursor_mode("vertical", checked))
        toolbar.addWidget(self._vertical_cursor_button)

        clear_button = QToolButton()
        clear_button.setText("Clear cursors")
        clear_button.clicked.connect(self._clear_cursors)
        toolbar.addWidget(clear_button)

        self._cursor_readout = QLabel("Cursors: select a mode, then click two points on the plot")
        toolbar.addWidget(self._cursor_readout)

    def _set_cursor_mode(self, mode: str, enabled: bool):
        if not enabled:
            if self._cursor_mode == mode:
                self._cursor_mode = None
            return

        other_button = (self._vertical_cursor_button if mode == "horizontal"
                        else self._horizontal_cursor_button)
        other_button.blockSignals(True)
        other_button.setChecked(False)
        other_button.blockSignals(False)
        self._cursor_mode = mode
        self._cursor_points[mode] = []
        for line in self._cursor_lines[mode]:
            line.setVisible(False)
        self._cursor_readout.setText(
            f"{mode.capitalize()} cursors: click the first point, then the second point")

    def _clear_cursors(self):
        self._cursor_mode = None
        for button in (self._horizontal_cursor_button, self._vertical_cursor_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        for mode, lines in self._cursor_lines.items():
            self._cursor_points[mode] = []
            for line in lines:
                line.setVisible(False)
        self._cursor_readout.setText("Cursors: select a mode, then click two points on the plot")

    def _on_plot_clicked(self, event):
        if self._cursor_mode is None or event.button() != Qt.MouseButton.LeftButton:
            return
        view_box = self.plotWidget.getViewBox()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        mode = self._cursor_mode
        value = point.y() if mode == "horizontal" else point.x()
        points = self._cursor_points[mode]
        points.append(value)
        line = self._cursor_lines[mode][len(points) - 1]
        line.setPos(value)
        line.setVisible(True)

        if len(points) == 1:
            self._cursor_readout.setText(
                f"{mode.capitalize()} cursors: click the second point")
            return

        self._cursor_mode = None
        button = (self._horizontal_cursor_button if mode == "horizontal"
                  else self._vertical_cursor_button)
        button.blockSignals(True)
        button.setChecked(False)
        button.blockSignals(False)
        self._update_cursor_readout(mode, points)

    @staticmethod
    def _format_voltage(value: float) -> str:
        if value >= 1.0:
            return f"{value:.4g} V"
        if value >= 1e-3:
            return f"{value * 1e3:.4g} mV"
        return f"{value * 1e6:.4g} µV"

    def _update_cursor_readout(self, mode: str, points: list[float]):
        delta = abs(points[1] - points[0])
        if mode == "vertical":
            seconds = delta * self.timebase / ADC_SAMPLE_RATE_HZ
            self._cursor_readout.setText(
                f"Vertical cursors: Δsamples {delta:.2f}; Δt {self._format_time_per_div(seconds).replace('/div', '')}")
            return

        ch1_volts = delta * self._input_volts_per_div(1) * DISPLAY_VERTICAL_DIVISIONS / ADC_COUNTS
        ch2_volts = delta * self._input_volts_per_div(2) * DISPLAY_VERTICAL_DIVISIONS / ADC_COUNTS
        self._cursor_readout.setText(
            f"Horizontal cursors: Δcodes {delta:.1f}; "
            f"ΔV CH1 {self._format_voltage(ch1_volts)}; "
            f"CH2 {self._format_voltage(ch2_volts)} (nominal)")

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.Wheel
                and watched in getattr(self, "_sidebar_wheel_controls", ())):
            pixel_delta = event.pixelDelta().y()
            if pixel_delta:
                scroll_amount = pixel_delta
            else:
                wheel_steps = event.angleDelta().y() / 120.0
                scroll_amount = round(wheel_steps *
                                      self._sidebar_scroll.verticalScrollBar().singleStep() * 3)
            if scroll_amount:
                scrollbar = self._sidebar_scroll.verticalScrollBar()
                scrollbar.setValue(scrollbar.value() - scroll_amount)
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _on_connected(self):
        self._ch1_data[:] = 0
        self._ch2_data[:] = 0
        self._status_label.setText("Connected")
        self._run_btn.setEnabled(True)
        self._single_btn.setEnabled(True)
        self._set_hardware_controls_enabled(False)
        self._send("status")

    def _on_disconnected(self):
        self._status_label.setText("Disconnected")
        self._run_btn.setChecked(False)
        self._run_btn.setText("Run")
        self._single_btn.setEnabled(True)
        self._set_hardware_controls_enabled(False)

    def _send(self, cmd: str):
        logger.info("CMD: %s", cmd)
        if self._conn_mgr:
            self._conn_mgr.send_command(cmd)

    def _send_control(self, cmd: str):
        self._cmd_panel.log_cmd(cmd)
        self._send(cmd)
        self._send("status")

    def _on_firmware_response(self, line: str):
        if line.startswith("ERR"):
            self._cmd_panel.log_error(line)
        else:
            self._cmd_panel.log_ok(line)
            if line.startswith("STATUS "):
                self._apply_firmware_status(line)

    @staticmethod
    def _status_fields(line: str) -> dict[str, str]:
        """Parse the firmware's space-delimited ``key=value`` STATUS reply."""
        fields: dict[str, str] = {}
        for token in line.removeprefix("STATUS ").split():
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
        return fields

    @staticmethod
    def _set_dial_from_status(dial, value: str | None) -> None:
        if value is None:
            return
        try:
            requested = round(float(value))
        except ValueError:
            return
        requested = max(dial.minimum(), min(dial.maximum(), requested))
        dial.blockSignals(True)
        dial.setValue(requested)
        dial.blockSignals(False)

    @staticmethod
    def _set_float_dial_from_status(dial, value_widget, value: str | None) -> None:
        if value is None:
            return
        try:
            requested = float(value)
        except ValueError:
            return
        requested = max(value_widget.minimum(), min(value_widget.maximum(), requested))
        raw_value = round(requested * 100)
        dial.blockSignals(True)
        value_widget.blockSignals(True)
        dial.setValue(raw_value)
        value_widget.setValue(requested)
        value_widget.blockSignals(False)
        dial.blockSignals(False)

    @staticmethod
    def _set_combo_text_from_status(combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _set_combo_data_from_status(combo: QComboBox, value: int) -> None:
        index = combo.findData(value)
        if index < 0:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _set_hardware_controls_enabled(self, enabled: bool) -> None:
        for control in getattr(self, "_hardware_controls", ()):
            control.setEnabled(enabled)

    def _apply_firmware_status(self, line: str) -> None:
        fields = self._status_fields(line)
        self._afe_state.update(fields)
        self._status_label.setText("Connected")

        self._set_float_dial_from_status(
            self._gain_dial, self._gain_value, fields.get("afe_ch1_gain_pct"))
        self._set_float_dial_from_status(
            self._offset_dial, self._offset_value, fields.get("afe_ch1_offset_pct"))
        self._set_float_dial_from_status(
            self._gain2_dial, self._gain2_value, fields.get("afe_ch2_gain_pct"))
        self._set_float_dial_from_status(
            self._offset2_dial, self._offset2_value, fields.get("afe_ch2_offset_pct"))
        self._set_float_dial_from_status(
            self._trigger_dial, self._trigger_value, fields.get("afe_trigger_level_pct"))
        self._set_range_combo_from_status(
            self._ch1_range_combo, "ch1", fields.get("afe_ch1_range_vpp"))
        self._set_range_combo_from_status(
            self._ch2_range_combo, "ch2", fields.get("afe_ch2_range_vpp"))

        self._set_combo_text_from_status(
            self._atten_combo, fields.get("afe_ch1_atten", "1:1"))
        self._set_combo_text_from_status(
            self._coupling_combo, fields.get("afe_ch1_coupling", "dc").upper())
        self._set_combo_text_from_status(
            self._ch2_atten_combo, fields.get("afe_ch2_atten", "1:1"))
        self._set_combo_text_from_status(
            self._ch2_coupling_combo, fields.get("afe_ch2_coupling", "dc").upper())
        self._set_combo_text_from_status(
            self._trigger_source_combo, f"CH{fields.get('afe_trigger_source', '1')}")
        self._set_combo_text_from_status(
            self._trigger_mode_combo,
            "Normal" if fields.get("trigger") == "normal" else "Off")
        self._update_trigger_line()

        try:
            capture_depth = int(fields["depth"])
            pretrigger = int(fields["pretrigger"])
            decimation = int(fields["decim"])
        except (KeyError, ValueError):
            pass
        else:
            self.capture_size = capture_depth
            self.pretrigger_size = pretrigger
            self._update_sample_axis()
            self._set_combo_data_from_status(self._sample_size_combo, capture_depth)
            self._populate_pretrigger_options()
            self._set_combo_data_from_status(self._pretrigger_combo, pretrigger)
            self._set_combo_data_from_status(self._timebase_combo, decimation)
            self.timebase = decimation

        is_interleaved = fields.get("interleaved") == "1"
        self._interleaved_btn.blockSignals(True)
        self._interleaved_btn.setChecked(is_interleaved)
        self._interleaved_btn.setText(f"Interleaved: {'ON' if is_interleaved else 'OFF'}")
        self._interleaved_btn.blockSignals(False)

        self._set_hardware_controls_enabled(True)
        self._update_scale_display()

        self._afe_info_text = (
            "AFE state (reported by firmware)\n"
            f"CH1: gain {fields.get('afe_ch1_gain_pct', '?')} %, "
            f"offset {fields.get('afe_ch1_offset_pct', '?')} %, "
            f"atten {fields.get('afe_ch1_atten', '?')}, "
            f"{fields.get('afe_ch1_coupling', '?').upper()}, "
            f"range {fields.get('afe_ch1_range_vpp', '?')} Vpp diff\n"
            f"CH2: gain {fields.get('afe_ch2_gain_pct', '?')} %, "
            f"offset {fields.get('afe_ch2_offset_pct', '?')} %, "
            f"atten {fields.get('afe_ch2_atten', '?')}, "
            f"{fields.get('afe_ch2_coupling', '?').upper()}, "
            f"range {fields.get('afe_ch2_range_vpp', '?')} Vpp diff\n"
            f"Trigger: CH{fields.get('afe_trigger_source', '?')}, "
            f"{fields.get('afe_trigger_level_pct', '?')} % "
            f"({fields.get('afe_trigger_level_mv', '?')} mV); "
            f"interleaved: {'ON' if fields.get('interleaved') == '1' else 'OFF'}\n"
            f"Display: raw signed ADC codes, "
            f"{ADC_COUNTS // DISPLAY_VERTICAL_DIVISIONS} codes/div. "
            "Input V/div requires calibrated AFE transfer data."
        )

    def _on_gain_change(self):
        self._update_scale_display()
        self._send_control(f"afe gain 1 {self._gain_value.value():.2f}")

    def _on_offset_change(self):
        self._send_control(f"afe offset 1 {self._offset_value.value():.2f}")

    def _add_adc_range_control(self, label: str, channel: str, layout) -> QComboBox:
        label_layout = QHBoxLayout()
        label_widget = QLabel(label.replace(" (", "\n("))
        label_widget.setWordWrap(True)
        label_layout.addWidget(label_widget)
        label_layout.addWidget(self._make_help_button(
            "ADC range and trigger level",
            "Selects the differential full-scale range expected at this "
            "channel's ADC input. It does not rescale the plotted samples, "
            "which remain raw ADC codes.\n\n"
            "This setting also defines the trigger threshold voltage when "
            "this channel is selected as the trigger source. The threshold "
            "is centred at 1.5 V on the positive input leg:\n"
            "• 1 Vpp differential: 1.25 V to 1.75 V\n"
            "• 2 Vpp differential: 1.00 V to 2.00 V\n\n"
            "The Trigger Level control spans the applicable range from 0% "
            "to 100%."))
        label_layout.addStretch()
        layout.addLayout(label_layout)
        combo = QComboBox()
        combo.addItem("1 Vpp", 1)
        combo.addItem("2 Vpp", 2)
        combo.setCurrentIndex(combo.findData(self._range_for_channel(channel)))
        combo.currentIndexChanged.connect(
            lambda _index, ch=channel, box=combo: self._on_adc_range_change(ch, box))
        layout.addWidget(combo)
        return combo

    def _make_help_button(self, title: str, text: str) -> QToolButton:
        """Create a compact, keyboard-accessible contextual help button."""
        button = QToolButton()
        button.setText("?")
        button.setToolTip(f"More information: {title}")
        button.setAccessibleName(f"Help: {title}")
        button.setFixedSize(22, 22)
        button.clicked.connect(lambda: self._show_help_dialog(title, text))
        return button

    def _show_help_dialog(self, title: str, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(430)
        dialog_layout = QVBoxLayout(dialog)
        message = QLabel(text)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dialog_layout.addWidget(message)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        dialog_layout.addWidget(buttons)
        dialog.exec()

    def _show_afe_state(self) -> None:
        """Show the latest firmware-reported AFE state outside the sidebar."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AFE State")
        dialog.setMinimumSize(510, 260)
        dialog_layout = QVBoxLayout(dialog)
        state_view = QPlainTextEdit()
        state_view.setReadOnly(True)
        state_view.setPlainText(self._afe_info_text)
        dialog_layout.addWidget(state_view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        dialog.exec()

    def _timebase_help_text(self) -> str:
        lines = [
            "Timebase sets the FPGA decimation factor. The ADC runs at 80 MHz, "
            "and the displayed time/div also depends on capture depth.\n",
            "Timebase       FPGA decimation",
        ]
        for _label, factor in TIMEBASES:
            time_per_div_s = self._time_per_div(factor)
            lines.append(
                f"{self._format_time_per_div(time_per_div_s):<14} {factor}")
        return "\n".join(lines)

    def _time_per_div(self, decimation: int) -> float:
        return (self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS
                * decimation / ADC_SAMPLE_RATE_HZ)

    @staticmethod
    def _format_time_per_div(seconds: float) -> str:
        if seconds >= 1.0:
            return f"{seconds:.4g} s/div"
        if seconds >= 1e-3:
            return f"{seconds * 1e3:.4g} ms/div"
        if seconds >= 1e-6:
            return f"{seconds * 1e6:.4g} µs/div"
        return f"{seconds * 1e9:.4g} ns/div"

    def _input_volts_per_div(self, channel: int) -> float:
        if channel == 1:
            gain_percent = self._gain_value.value()
            attenuation_name = self._atten_combo.currentText()
            adc_range_vpp = float(self._ch1_range_combo.currentData())
        else:
            gain_percent = self._gain2_value.value()
            attenuation_name = self._ch2_atten_combo.currentText()
            adc_range_vpp = float(self._ch2_range_combo.currentData())
        attenuation = (AFE_ATTEN_1_TO_100 if attenuation_name == "1:100"
                       else AFE_ATTEN_1_TO_1)
        vga_linear = 10.0 ** (AFE_VGA_MAX_DB * gain_percent / 2000.0)
        return adc_range_vpp / (attenuation * vga_linear * AFE_DIFF_AMP_GAIN
                                * DISPLAY_VERTICAL_DIVISIONS)

    def _update_scale_display(self) -> None:
        time_per_div = self._time_per_div(self.timebase)
        self._scale_label.setText(
            "Scale (nominal): "
            f"CH1 {self._input_volts_per_div(1):.4g} V/div; "
            f"CH2 {self._input_volts_per_div(2):.4g} V/div; "
            f"{self._format_time_per_div(time_per_div)} "
            f"({self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS:g} samples/div)"
        )
        self._timebase_combo.blockSignals(True)
        for index, (_label, factor) in enumerate(TIMEBASES):
            self._timebase_combo.setItemText(
                index, self._format_time_per_div(self._time_per_div(factor)))
        self._timebase_combo.blockSignals(False)

    def _range_for_channel(self, channel: str) -> int:
        value = self._settings.get("adc_range_vpp", {}).get(channel, 2)
        return value if value in (1, 2) else 2

    def _set_range_combo_from_status(self, combo: QComboBox, channel: str,
                                     value: str | None) -> None:
        try:
            vpp = int(value) if value is not None else None
        except ValueError:
            return
        if vpp not in (1, 2):
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(combo.findData(vpp))
        combo.blockSignals(False)
        self._settings.setdefault("adc_range_vpp", {})[channel] = vpp
        save_settings(self._settings)

    def _on_adc_range_change(self, channel: str, combo: QComboBox):
        vpp = int(combo.currentData())
        self._settings.setdefault("adc_range_vpp", {})[channel] = vpp
        save_settings(self._settings)
        self._update_scale_display()
        self._send_control(f"afe range {1 if channel == 'ch1' else 2} {vpp}")

    def _on_gain2_change(self):
        self._update_scale_display()
        self._send_control(f"afe gain 2 {self._gain2_value.value():.2f}")

    def _on_offset2_change(self):
        self._send_control(f"afe offset 2 {self._offset2_value.value():.2f}")

    def _populate_pretrigger_options(self):
        """Offer only encodable pretrigger depths smaller than capture depth."""
        selected = self.pretrigger_size
        self._pretrigger_combo.blockSignals(True)
        self._pretrigger_combo.clear()
        self._pretrigger_combo.addItem("Off", 0)
        count = 1
        while count < self.capture_size and count <= 4096:
            self._pretrigger_combo.addItem(str(count), count)
            count *= 2
        index = self._pretrigger_combo.findData(selected)
        self._pretrigger_combo.setCurrentIndex(index if index >= 0 else 0)
        self._pretrigger_combo.blockSignals(False)

    def _on_trigger_change(self):
        self._update_trigger_line()
        self._send_control(f"afe trigger_level {self._trigger_value.value():.2f}")

    def _on_trigger_source_change(self, source: str):
        ch = "2" if source == "CH2" else "1"
        self._send_control(f"afe trigger_ch {ch}")

    def _on_trigger_mode_change(self, mode_text: str):
        mode = "normal" if mode_text == "Normal" else "off"
        self._update_trigger_line()
        self._send_control(f"afe trigger_mode {mode}")

    def _update_trigger_line(self):
        trigger_percent = self._trigger_value.value()
        display_code = ADC_COUNTS * trigger_percent / 100.0 - ADC_COUNTS / 2
        self._trigger_line.setPos(display_code)
        self._trigger_line.setVisible(self._trigger_mode_combo.currentText() == "Normal")

    def _on_sample_size_change(self, _index):
        count = int(self._sample_size_combo.currentData())
        self.capture_size = count
        self._update_sample_axis()
        if self.pretrigger_size >= count:
            self.pretrigger_size = 0
            self._populate_pretrigger_options()
            self._send_control("afe pretrigger 0")
        else:
            self._populate_pretrigger_options()
        self._send_control(f"afe sample_size {count}")

    def _update_sample_axis(self):
        self.plotWidget.setXRange(0, self.capture_size, padding=0)
        self.plotWidget.getViewBox().setLimits(xMin=0, xMax=self.capture_size)
        step = self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS
        ticks = [(step * index, str(int(step * index)))
                 for index in range(DISPLAY_HORIZONTAL_DIVISIONS + 1)]
        self.plotWidget.getAxis("bottom").setTicks([ticks])
        self._update_scale_display()

    def _on_pretrigger_change(self, _index):
        self.pretrigger_size = int(self._pretrigger_combo.currentData())
        self._send_control(f"afe pretrigger {self.pretrigger_size}")

    def _on_timebase_change(self, index: int):
        _, factor = TIMEBASES[index]
        self.timebase = factor
        self._update_scale_display()
        self._send_control(f"afe decim {factor}")

    def _on_format_change(self, fmt: str):
        self._adc_format = fmt

    def _on_coupling_change(self, selection: str):
        coupling = "ac" if selection == "AC" else "dc"
        self._send_control(f"afe coupling 1 {coupling}")

    def _on_attenuation_change(self, selection: str):
        atten = "100" if selection == "1:100" else "1"
        self._update_scale_display()
        self._send_control(f"afe atten 1 {atten}")

    def _on_ch2_coupling_change(self, selection: str):
        coupling = "ac" if selection == "AC" else "dc"
        self._send_control(f"afe coupling 2 {coupling}")

    def _on_ch2_attenuation_change(self, selection: str):
        atten = "100" if selection == "1:100" else "1"
        self._update_scale_display()
        self._send_control(f"afe atten 2 {atten}")

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
        self._send_control(f"afe interleaved {1 if checked else 0}")

    def _channel_afe_calculation(self, channel: int) -> str:
        prefix = f"afe_ch{channel}_"
        gain_percent = float(self._afe_state.get(
            f"{prefix}gain_pct",
            self._gain_value.value() if channel == 1 else self._gain2_value.value()))
        attenuation_name = self._afe_state.get(
            f"{prefix}atten",
            self._atten_combo.currentText() if channel == 1
            else self._ch2_atten_combo.currentText())
        range_vpp = float(self._afe_state.get(
            f"{prefix}range_vpp", self._range_for_channel(f"ch{channel}")))

        if attenuation_name == "1:100":
            attenuation = AFE_ATTEN_1_TO_100
            attenuation_formula = "10 kΩ / (1 MΩ + 10 kΩ)"
        else:
            attenuation = AFE_ATTEN_1_TO_1
            attenuation_formula = "953 kΩ / (49.9 kΩ + 953 kΩ)"

        vga_db = AFE_VGA_MAX_DB * gain_percent / 100.0
        vga_linear = 10.0 ** (vga_db / 20.0)
        total_gain = attenuation * vga_linear * AFE_DIFF_AMP_GAIN
        input_full_scale_vpp = range_vpp / total_gain
        input_lsb_v = input_full_scale_vpp / (2 ** ADC_BITS)

        return (
            f"CH{channel}\n"
            f"  Input attenuator: {attenuation_name}\n"
            f"    α = {attenuation_formula} = {attenuation:.7f} "
            f"({20.0 * np.log10(attenuation):.3f} dB)\n"
            f"  VGA control: {gain_percent:.2f} %\n"
            f"    G_VGA = 24 dB × {gain_percent:.2f}/100 = {vga_db:.3f} dB\n"
            f"    g_VGA = 10^({vga_db:.3f}/20) = {vga_linear:.6f} V/V\n"
            f"  Differential driver: {AFE_DIFF_AMP_GAIN:.1f} V/V (0 dB)\n"
            f"  Total transfer\n"
            f"    V_ADC,diff / V_input = α × g_VGA × 1 = {total_gain:.7f} V/V\n"
            f"  Selected ADC range: {range_vpp:.1f} Vpp differential\n"
            f"    Input full scale = {range_vpp:.1f} / {total_gain:.7f} "
            f"= {input_full_scale_vpp:.4f} Vpp\n"
            f"    Input scale = {input_full_scale_vpp / DISPLAY_VERTICAL_DIVISIONS:.4f} V/div\n"
            f"    Nominal input LSB = {input_lsb_v * 1e6:.3f} µV/LSB\n"
        )

    def _show_afe_calculations(self):
        trigger_channel = int(self._afe_state.get("afe_trigger_source", "1"))
        trigger_channel = trigger_channel if trigger_channel in (1, 2) else 1
        trigger_percent = float(self._afe_state.get(
            "afe_trigger_level_pct", self._trigger_value.value()))
        trigger_range = float(self._afe_state.get(
            f"afe_ch{trigger_channel}_range_vpp",
            self._range_for_channel(f"ch{trigger_channel}")))
        trigger_min = 1.5 - trigger_range / 4.0
        trigger_max = 1.5 + trigger_range / 4.0
        trigger_voltage = trigger_min + trigger_percent / 100.0 * (trigger_max - trigger_min)

        text = (
            "AFE configuration calculations (nominal)\n"
            "=" * 45 + "\n\n"
            + self._channel_afe_calculation(1) + "\n"
            + self._channel_afe_calculation(2) + "\n"
            "Trigger\n"
            f"  Source: CH{trigger_channel}; range: {trigger_range:.1f} Vpp differential\n"
            f"  Positive-leg limits: 1.5 V ± {trigger_range / 4.0:.3f} V "
            f"= {trigger_min:.3f} V … {trigger_max:.3f} V\n"
            f"  Threshold: {trigger_percent:.2f} % → {trigger_voltage:.4f} V\n\n"
            "The attenuator and VGA figures are nominal design values. "
            "They are for configuration and planning; use measured calibration "
            "data before treating them as absolute-voltage accuracy."
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("AFE configuration calculations")
        dialog.resize(760, 690)
        layout = QVBoxLayout(dialog)
        calculation_view = QPlainTextEdit(text)
        calculation_view.setReadOnly(True)
        calculation_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(calculation_view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _toggle_run(self, checked: bool):
        if checked:
            self._run_btn.setText("Stop")
            self._single_btn.setEnabled(False)
            if self._conn_mgr:
                self._conn_mgr.start_acquisition("continuous")
        else:
            self._run_btn.setText("Run")
            self._single_btn.setEnabled(True)
            if self._conn_mgr:
                self._conn_mgr.stop_acquisition()

    def _single_acquire(self):
        if self._conn_mgr:
            self._single_btn.setEnabled(False)
            self._status_label.setText("Acquiring…")
            self._conn_mgr.start_acquisition("single")

    def _on_acquisition_done(self):
        self._single_btn.setEnabled(True)
        self._status_label.setText("Connected")

    def _save_frame(self):
        if not self._have_frame:
            self._cmd_panel.log_error("No frame to save")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"capture_{ts}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save frame", default_name, "CSV file (*.csv)")
        if not path:
            return

        save_capture_csv(
            path,
            self._ch1_raw,
            self._ch2_raw,
            fs_hz=ADC_SAMPLE_RATE_HZ / self.timebase,
            n_bits=14,
            metadata={
                "capture_name": os.path.basename(path),
                "decim_factor": int(self.timebase),
                "capture_depth": len(self._ch1_raw),
                "pretrigger_samples": int(self.pretrigger_size),
                "trigger_mode": self._trigger_mode_combo.currentText().lower(),
                "trigger_source": 2 if self._trigger_source_combo.currentText() == "CH2" else 1,
                "adc_format": self._adc_format,
                **{f"firmware_{key}": value for key, value in self._afe_state.items()
                   if key.startswith("afe_") or key == "interleaved"},
            },
        )
        self._cmd_panel.log_ok(f"Saved: {os.path.basename(path)}")

    def _raw_to_display_codes(self, raw: np.ndarray) -> np.ndarray:
        """Return signed ADC codes without applying gain, offset, or filtering."""
        half = ADC_COUNTS // 2
        if self._adc_format == "Offset Binary":
            return raw.astype(np.int32) - half
        else:
            return np.where(raw >= half,
                            raw.astype(np.int32) - ADC_COUNTS,
                            raw.astype(np.int32))

    def _update_plot(self):
        latest = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            ch1_raw, ch2_raw = latest
            self._ch1_raw = np.asarray(ch1_raw, dtype=np.uint16).copy()
            self._ch2_raw = np.asarray(ch2_raw, dtype=np.uint16).copy()
            self._ch1_data = self._raw_to_display_codes(ch1_raw)
            self._ch2_data = self._raw_to_display_codes(ch2_raw)
            self._have_frame = True

        if self.ch1_enabled:
            self._curve_ch1.setData(self._ch1_data)
        if self.ch2_enabled:
            self._curve_ch2.setData(self._ch2_data)
