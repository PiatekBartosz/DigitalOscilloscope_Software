import datetime
import logging
import os
import pathlib
import queue
import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow,
    QGroupBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QComboBox,
    QPushButton,
    QSplitter,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QToolButton,
    QScrollArea,
    QDial,
    QSpinBox,
    QDoubleSpinBox,
    QToolBar,
    QWidgetAction,
)
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QActionGroup

from utils.controls import create_dial_widget, create_float_dial_widget
from utils.settings import load_settings, save_settings
from ui.command_panel import CommandPanel
from analysis.capture_io import save_capture_csv

logger = logging.getLogger(__name__)

ADC_COUNTS = 16384
ADC_SAMPLE_RATE_HZ = 80_000_000.0
DISPLAY_VERTICAL_DIVISIONS = 8
DISPLAY_VERTICAL_MARGIN_DIVISIONS = 0.5
DISPLAY_HORIZONTAL_DIVISIONS = 10
# Rendering more samples than horizontal pixels does not add detail and can
# dominate the GUI thread on large captures.  Capture and save paths keep the
# complete frame; this limit applies exclusively to plot data.
MAX_DISPLAY_POINTS = 3000
DISPLAY_INTERVAL_MS = math.ceil(1000 / 30)
FFT_WINDOWS = (
    ("Rectangular", "rect"),
    ("Hann", "hann"),
    ("Hamming", "hamming"),
    ("Blackman", "blackman"),
)

AFE_ATTEN_1_TO_1 = 953_000.0 / (49_900.0 + 953_000.0)
AFE_ATTEN_1_TO_100 = 10_000.0 / (1_000_000.0 + 10_000.0)
AFE_VGA_MAX_DB = 24.0
AFE_DIFF_AMP_GAIN = 1.0
ADC_BITS = 14

# Standard requested horizontal scales.  The FPGA uses an integer decimation,
# so the displayed actual scale may differ slightly from the selected target.
TIMEBASES = (
    ("200 ns/div", 200e-9),
    ("500 ns/div", 500e-9),
    ("1 µs/div", 1e-6),
    ("2 µs/div", 2e-6),
    ("5 µs/div", 5e-6),
    ("10 µs/div", 10e-6),
    ("20 µs/div", 20e-6),
    ("50 µs/div", 50e-6),
    ("100 µs/div", 100e-6),
    ("200 µs/div", 200e-6),
    ("500 µs/div", 500e-6),
    ("1 ms/div", 1e-3),
    ("2 ms/div", 2e-3),
    ("5 ms/div", 5e-3),
    ("10 ms/div", 10e-3),
)
CAPTURE_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)
MIN_DECIMATION = 1
MAX_DECIMATION = 1023
CAPTURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "results" / "captures"


class Oscilloscope(QMainWindow):
    DISPLAY_SAMPLES = 8192

    def __init__(self, conn_mgr, frame_queue: queue.Queue):
        super().__init__()
        self._conn_mgr = conn_mgr
        self._frame_queue = frame_queue

        self._ch1_data = np.zeros(self.DISPLAY_SAMPLES, dtype=np.float32)
        self._ch2_data = np.zeros(self.DISPLAY_SAMPLES, dtype=np.float32)
        self._plot_x = np.arange(self.DISPLAY_SAMPLES, dtype=np.int32)
        self._ch1_raw = np.zeros(self.DISPLAY_SAMPLES, dtype=np.uint16)
        self._ch2_raw = np.zeros(self.DISPLAY_SAMPLES, dtype=np.uint16)
        self._have_frame = False
        self._plot_mode = "timeseries"
        self._show_sample_points = False
        self._fft_x_max_hz = ADC_SAMPLE_RATE_HZ / 2
        self._fft_window = "hann"
        self._afe_state: dict[str, str] = {}
        self._afe_info_text = "AFE state: waiting for firmware status…"
        self._settings = load_settings()
        self._cursor_mode: str | None = None
        self._cursor_points = {"horizontal": [], "vertical": []}

        self.capture_size = self.DISPLAY_SAMPLES
        self.pretrigger_size = 0
        self.timebase_seconds = TIMEBASES[0][1]
        self._timebase_target: float | None = TIMEBASES[0][1]
        self.decimation = MIN_DECIMATION
        self._build_ui()

        self._adc_format = "Offset Binary"

        self.ch1_enabled = True
        self.ch2_enabled = True

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(DISPLAY_INTERVAL_MS)

    def _build_ui(self):
        self.setWindowTitle("Oscilloscope")
        self.resize(1100, 820)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        _here = os.path.dirname(os.path.abspath(__file__))
        _qss = os.path.join(_here, "..", "style", "style.qss")
        try:
            with open(_qss, "r") as f:
                self.setStyleSheet(f.read())
        except (FileNotFoundError, OSError):
            pass

        central = QWidget()
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setCentralWidget(central)

        self._build_cursor_toolbar()

        menu_bar = self.menuBar()
        afe_state_action = menu_bar.addAction("AFE State")
        afe_state_action.triggered.connect(self._show_afe_state)
        menu_bar.addSeparator()
        self._time_series_menu_action = menu_bar.addAction("Time Series")
        self._time_series_menu_action.setCheckable(True)
        self._time_series_menu_action.setChecked(True)
        self._time_series_menu_action.triggered.connect(
            lambda: self._set_plot_mode("timeseries")
        )
        self._fft_menu_action = menu_bar.addAction("FFT")
        self._fft_menu_action.setCheckable(True)
        self._fft_menu_action.triggered.connect(
            lambda: self._set_plot_mode("fft")
        )
        self._plot_mode_actions = QActionGroup(self)
        self._plot_mode_actions.setExclusive(True)
        self._plot_mode_actions.addAction(self._time_series_menu_action)
        self._plot_mode_actions.addAction(self._fft_menu_action)
        menu_bar.addSeparator()
        self._scale_label = QLabel()
        self._scale_label.setStyleSheet("padding: 0 8px;")
        scale_action = QWidgetAction(self)
        scale_action.setDefaultWidget(self._scale_label)
        menu_bar.addAction(scale_action)

        self._cmd_panel = CommandPanel()
        self._cmd_panel.command_submitted.connect(self._send)

        top_widget = QWidget()
        main_layout = QHBoxLayout(top_widget)

        self.plotWidget = pg.PlotWidget()
        self.plotWidget.showGrid(x=True, y=True, alpha=0.4)
        self.plotWidget.setLabel("bottom", "Samples")
        self.plotWidget.getAxis("left").setStyle(showValues=False)

        vb = self.plotWidget.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.disableAutoRange()
        self.plotWidget.setMenuEnabled(False)
        self.plotWidget.hideButtons()
        self.plotWidget.setXRange(0, self.DISPLAY_SAMPLES, padding=0)
        # Keep the signal area at eight divisions, with half a division of
        # uncluttered headroom above and below it.
        y_signal_min = -ADC_COUNTS // 2
        y_signal_max = ADC_COUNTS // 2
        y_margin = (
            ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS * DISPLAY_VERTICAL_MARGIN_DIVISIONS
        )
        self.plotWidget.setYRange(
            y_signal_min - y_margin, y_signal_max + y_margin, padding=0
        )
        vb.setLimits(
            xMin=0,
            xMax=self.DISPLAY_SAMPLES,
            yMin=y_signal_min - y_margin,
            yMax=y_signal_max + y_margin,
        )

        x_step = self.DISPLAY_SAMPLES / 10
        x_ticks = [(x_step * i, str(int(x_step * i))) for i in range(11)]
        y_step = ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
        y_ticks = [
            (-ADC_COUNTS / 2 + y_step * i, "")
            for i in range(DISPLAY_VERTICAL_DIVISIONS + 1)
        ]
        self.plotWidget.getAxis("bottom").setTicks([x_ticks])
        self.plotWidget.getAxis("left").setTicks([y_ticks])

        self._curve_ch1 = self.plotWidget.plot(pen="y", name="CH1")
        self._curve_ch2 = self.plotWidget.plot(pen="c", name="CH2")
        self._trigger_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("r", width=2),
            movable=False,
        )
        self._trigger_line.setVisible(False)
        self.plotWidget.addItem(self._trigger_line)
        self._cursor_lines = {
            "horizontal": [
                pg.InfiniteLine(
                    angle=0, pen=pg.mkPen("#44ff88", width=2), movable=False
                ),
                pg.InfiniteLine(
                    angle=0, pen=pg.mkPen("#44ff88", width=2), movable=False
                ),
            ],
            "vertical": [
                pg.InfiniteLine(
                    angle=90, pen=pg.mkPen("#ff66ff", width=2), movable=False
                ),
                pg.InfiniteLine(
                    angle=90, pen=pg.mkPen("#ff66ff", width=2), movable=False
                ),
            ],
        }
        for lines in self._cursor_lines.values():
            for line in lines:
                line.setVisible(False)
                line.setZValue(100)
                self.plotWidget.addItem(line)
        self.plotWidget.scene().sigMouseClicked.connect(self._on_plot_clicked)
        vb.sigRangeChanged.connect(self._on_plot_range_changed)
        main_layout.addWidget(self.plotWidget, stretch=4)

        ctrl_frame = QFrame()
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

        def add_control_group(title: str) -> QVBoxLayout:
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            ctrl_layout.addWidget(group)
            return layout

        acquisition_layout = add_control_group("Acquisition")
        trigger_controls_layout = add_control_group("Trigger")
        ch1_controls_layout = add_control_group("Channel 1 AFE")
        ch2_controls_layout = add_control_group("Channel 2 AFE")
        display_controls_layout = add_control_group("Display")
        connection_layout = add_control_group("Connection")

        self._gain_dial, self._gain_value = create_float_dial_widget(
            "Gain (%)", 0, 100, 50, ch1_controls_layout, self._on_gain_change
        )
        self._offset_dial, self._offset_value = create_float_dial_widget(
            "Offset (%)", 0, 100, 50, ch1_controls_layout, self._on_offset_change
        )
        self._ch1_range_combo = self._add_adc_range_control(
            "ADC Range (differential)", "ch1", ch1_controls_layout
        )

        self._gain2_dial, self._gain2_value = create_float_dial_widget(
            "Gain (%)", 0, 100, 50, ch2_controls_layout, self._on_gain2_change
        )
        self._offset2_dial, self._offset2_value = create_float_dial_widget(
            "Offset (%)", 0, 100, 50, ch2_controls_layout, self._on_offset2_change
        )
        self._ch2_range_combo = self._add_adc_range_control(
            "ADC Range (differential)", "ch2", ch2_controls_layout
        )

        trigger_layout = QHBoxLayout()
        trigger_layout.addWidget(QLabel("Trigger Level (AFE %)"))
        trigger_layout.addWidget(
            self._make_help_button(
                "Trigger level",
                "The trigger percentage is applied to the currently selected "
                "trigger-source channel.\n\n"
                "It is not a percentage of the DAC's full output range. The "
                "firmware converts it to the positive leg of that channel's "
                "differential ADC range, centred at 1.5 V.\n\n"
                "For a 1 Vpp range, 0–100% maps to 1.25–1.75 V. For a 2 Vpp "
                "range, it maps to 1.00–2.00 V. Changing the ADC range therefore "
                "also changes the voltage represented by the trigger level.",
            )
        )
        trigger_layout.addStretch()
        trigger_controls_layout.addLayout(trigger_layout)
        self._trigger_dial, self._trigger_value = create_float_dial_widget(
            "", 0, 100, 50, trigger_controls_layout, self._on_trigger_change
        )

        acquisition_layout.addWidget(QLabel("Timebase (H. Scale)"))
        self._timebase_combo = QComboBox()
        for label, seconds in TIMEBASES:
            self._timebase_combo.addItem(label, seconds)
        self._timebase_custom_index = self._timebase_combo.count()
        self._timebase_combo.addItem("Custom", None)
        self._timebase_combo.setItemData(
            self._timebase_custom_index,
            "Manual capture depth or decimation; actual timebase is shown above.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._timebase_combo.currentIndexChanged.connect(self._on_timebase_change)
        acquisition_layout.addWidget(self._timebase_combo)

        acquisition_layout.addWidget(QLabel("Capture Depth"))
        self._sample_size_combo = QComboBox()
        for count in CAPTURE_SIZES:
            self._sample_size_combo.addItem(f"{count} samples", count)
        self._sample_size_combo.setCurrentIndex(
            self._sample_size_combo.findData(self.capture_size)
        )
        self._sample_size_combo.currentIndexChanged.connect(self._on_sample_size_change)
        acquisition_layout.addWidget(self._sample_size_combo)

        acquisition_layout.addWidget(QLabel("Decimation"))
        self._decimation_spinbox = QSpinBox()
        self._decimation_spinbox.setRange(MIN_DECIMATION, MAX_DECIMATION)
        self._decimation_spinbox.setValue(self.decimation)
        self._decimation_spinbox.setKeyboardTracking(False)
        self._decimation_spinbox.valueChanged.connect(self._on_decimation_change)
        acquisition_layout.addWidget(self._decimation_spinbox)

        acquisition_layout.addWidget(QLabel("Pretrigger Samples"))
        self._pretrigger_combo = QComboBox()
        self._populate_pretrigger_options()
        self._pretrigger_combo.currentIndexChanged.connect(self._on_pretrigger_change)
        acquisition_layout.addWidget(self._pretrigger_combo)

        display_controls_layout.addWidget(QLabel("ADC Format"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["Offset Binary", "2's Complement"])
        self._format_combo.currentTextChanged.connect(self._on_format_change)
        display_controls_layout.addWidget(self._format_combo)

        ch1_controls_layout.addWidget(QLabel("Coupling"))
        self._coupling_combo = QComboBox()
        self._coupling_combo.addItems(["DC", "AC"])
        self._coupling_combo.currentTextChanged.connect(self._on_coupling_change)
        ch1_controls_layout.addWidget(self._coupling_combo)

        ch1_controls_layout.addWidget(QLabel("Attenuation"))
        self._atten_combo = QComboBox()
        self._atten_combo.addItems(["1:1", "1:100"])
        self._atten_combo.currentTextChanged.connect(self._on_attenuation_change)
        ch1_controls_layout.addWidget(self._atten_combo)

        ch2_controls_layout.addWidget(QLabel("Coupling"))
        self._ch2_coupling_combo = QComboBox()
        self._ch2_coupling_combo.addItems(["DC", "AC"])
        self._ch2_coupling_combo.currentTextChanged.connect(
            self._on_ch2_coupling_change
        )
        ch2_controls_layout.addWidget(self._ch2_coupling_combo)

        ch2_controls_layout.addWidget(QLabel("Attenuation"))
        self._ch2_atten_combo = QComboBox()
        self._ch2_atten_combo.addItems(["1:1", "1:100"])
        self._ch2_atten_combo.currentTextChanged.connect(
            self._on_ch2_attenuation_change
        )
        ch2_controls_layout.addWidget(self._ch2_atten_combo)

        trigger_controls_layout.addWidget(QLabel("Source"))
        self._trigger_source_combo = QComboBox()
        self._trigger_source_combo.addItems(["CH1", "CH2"])
        self._trigger_source_combo.currentTextChanged.connect(
            self._on_trigger_source_change
        )
        trigger_controls_layout.addWidget(self._trigger_source_combo)

        trigger_controls_layout.addWidget(QLabel("Mode"))
        self._trigger_mode_combo = QComboBox()
        self._trigger_mode_combo.addItems(["Off", "Normal"])
        self._trigger_mode_combo.currentTextChanged.connect(
            self._on_trigger_mode_change
        )
        trigger_controls_layout.addWidget(self._trigger_mode_combo)

        self._ch1_to_adc2_btn = QPushButton("CH1→ADC2: OFF")
        self._ch1_to_adc2_btn.setCheckable(True)
        self._ch1_to_adc2_btn.setToolTip("Route CH1 into ADC channel 2.")
        self._ch1_to_adc2_btn.toggled.connect(self._on_ch1_to_adc2_change)
        acquisition_layout.addWidget(self._ch1_to_adc2_btn)

        self._status_label = QLabel("Connecting…")
        connection_layout.addWidget(self._status_label)

        self._refresh_afe_btn = QPushButton("Refresh AFE state")
        self._refresh_afe_btn.clicked.connect(lambda: self._send("status"))
        connection_layout.addWidget(self._refresh_afe_btn)

        if self._conn_mgr:
            self._conn_mgr.connected.connect(self._on_connected)
            self._conn_mgr.disconnected.connect(self._on_disconnected)
            self._conn_mgr.connecting.connect(
                lambda: self._status_label.setText("Connecting…")
            )
            self._conn_mgr.device_found.connect(
                lambda addr: self._status_label.setText(f"Found: {addr}")
            )
            self._conn_mgr.acquisition_done.connect(self._on_acquisition_done)

        acq_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setCheckable(True)
        self._run_btn.toggled.connect(self._toggle_run)
        acq_row.addWidget(self._run_btn)

        self._single_btn = QPushButton("Single")
        self._single_btn.clicked.connect(self._single_acquire)
        acq_row.addWidget(self._single_btn)
        acquisition_layout.addLayout(acq_row)

        self._save_btn = QPushButton("Save frame")
        self._save_btn.clicked.connect(self._save_frame)
        acquisition_layout.addWidget(self._save_btn)

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
        display_controls_layout.addLayout(ch_row)

        self._sidebar_wheel_controls = (
            ctrl_widget.findChildren(QComboBox)
            + ctrl_widget.findChildren(QDial)
            + ctrl_widget.findChildren(QSpinBox)
        )
        for control in self._sidebar_wheel_controls:
            control.installEventFilter(self)

        self._hardware_controls = [
            self._gain_dial,
            self._gain_value,
            self._offset_dial,
            self._offset_value,
            self._ch1_range_combo,
            self._gain2_dial,
            self._gain2_value,
            self._offset2_dial,
            self._offset2_value,
            self._ch2_range_combo,
            self._trigger_dial,
            self._trigger_value,
            self._sample_size_combo,
            self._decimation_spinbox,
            self._pretrigger_combo,
            self._timebase_combo,
            self._coupling_combo,
            self._atten_combo,
            self._ch2_coupling_combo,
            self._ch2_atten_combo,
            self._trigger_source_combo,
            self._trigger_mode_combo,
            self._ch1_to_adc2_btn,
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
                lambda: self._cmd_panel.log_ok("Connected")
            )
            self._conn_mgr.disconnected.connect(
                lambda: self._cmd_panel.log_error("Disconnected")
            )
            self._conn_mgr.connecting.connect(
                lambda: self._cmd_panel.log_info("Connecting…")
            )
            self._conn_mgr.device_found.connect(
                lambda addr: self._cmd_panel.log_ok(f"Device found: {addr}")
            )
            self._conn_mgr.response_received.connect(self._on_firmware_response)
            self._conn_mgr.acquisition_done.connect(
                lambda: self._cmd_panel.log_info("Single acquisition complete")
            )

    def _build_cursor_toolbar(self):
        toolbar = QToolBar("Cursors", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._horizontal_cursor_button = QToolButton()
        self._horizontal_cursor_button.setText("Horizontal cursors")
        self._horizontal_cursor_button.setCheckable(True)
        self._horizontal_cursor_button.toggled.connect(
            lambda checked: self._set_cursor_mode("horizontal", checked)
        )
        horizontal_cursor_action = toolbar.addWidget(self._horizontal_cursor_button)

        self._vertical_cursor_button = QToolButton()
        self._vertical_cursor_button.setText("Vertical cursors")
        self._vertical_cursor_button.setCheckable(True)
        self._vertical_cursor_button.toggled.connect(
            lambda checked: self._set_cursor_mode("vertical", checked)
        )
        vertical_cursor_action = toolbar.addWidget(self._vertical_cursor_button)

        clear_button = QToolButton()
        clear_button.setText("Clear cursors")
        clear_button.clicked.connect(self._clear_cursors)
        clear_cursor_action = toolbar.addWidget(clear_button)

        self._cursor_readout = QLabel(
            "Cursors: select a mode, then click two points on the plot"
        )
        cursor_readout_action = toolbar.addWidget(self._cursor_readout)
        self._cursor_toolbar_actions = (
            horizontal_cursor_action,
            vertical_cursor_action,
            clear_cursor_action,
            cursor_readout_action,
        )

        self._cursor_toolbar_separator = toolbar.addSeparator()

        self._sample_points_button = QToolButton()
        self._sample_points_button.setText("Sample points")
        self._sample_points_button.setCheckable(True)
        self._sample_points_button.setToolTip(
            "Show individual time-domain samples without connecting lines"
        )
        self._sample_points_button.toggled.connect(self._on_sample_points_toggle)
        sample_points_action = toolbar.addWidget(self._sample_points_button)

        self._full_record_button = QToolButton()
        self._full_record_button.setText("Full record")
        self._full_record_button.setToolTip("Reset the time-domain view to the full capture")
        self._full_record_button.clicked.connect(self._reset_time_series_view)
        full_record_action = toolbar.addWidget(self._full_record_button)
        self._time_series_toolbar_actions = (sample_points_action, full_record_action)

        self._fft_toolbar_separator = toolbar.addSeparator()

        fft_x_label_action = toolbar.addWidget(QLabel("FFT X max:"))
        self._fft_x_max_spinbox = QDoubleSpinBox()
        self._fft_x_max_spinbox.setDecimals(4)
        self._fft_x_max_spinbox.setRange(0.0001, ADC_SAMPLE_RATE_HZ / 2e6)
        self._fft_x_max_spinbox.setSingleStep(0.1)
        self._fft_x_max_spinbox.setSuffix(" MHz")
        self._fft_x_max_spinbox.setValue(self._fft_x_max_hz / 1e6)
        self._fft_x_max_spinbox.valueChanged.connect(self._on_fft_x_max_change)
        self._fft_x_max_spinbox.setEnabled(False)
        fft_x_spinbox_action = toolbar.addWidget(self._fft_x_max_spinbox)

        fft_window_label_action = toolbar.addWidget(QLabel("FFT window:"))
        self._fft_window_combo = QComboBox()
        for label, window in FFT_WINDOWS:
            self._fft_window_combo.addItem(label, window)
        self._fft_window_combo.setCurrentIndex(self._fft_window_combo.findData("hann"))
        self._fft_window_combo.setToolTip(
            "Window applied to each live FFT before calculating its magnitude"
        )
        self._fft_window_combo.currentIndexChanged.connect(self._on_fft_window_change)
        self._fft_window_combo.setEnabled(False)
        fft_window_combo_action = toolbar.addWidget(self._fft_window_combo)
        self._fft_toolbar_actions = (
            fft_x_label_action,
            fft_x_spinbox_action,
            fft_window_label_action,
            fft_window_combo_action,
        )
        self._set_plot_toolbar_visibility(is_time_series=True)

    def _set_cursor_mode(self, mode: str, enabled: bool):
        if not enabled:
            if self._cursor_mode == mode:
                self._cursor_mode = None
            return

        other_button = (
            self._vertical_cursor_button
            if mode == "horizontal"
            else self._horizontal_cursor_button
        )
        other_button.blockSignals(True)
        other_button.setChecked(False)
        other_button.blockSignals(False)
        self._cursor_mode = mode
        self._cursor_points[mode] = []
        for line in self._cursor_lines[mode]:
            line.setVisible(False)
        self._cursor_readout.setText(
            f"{mode.capitalize()} cursors: click the first point, then the second point"
        )

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
        self._cursor_readout.setText(
            "Cursors: select a mode, then click two points on the plot"
        )

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
                f"{mode.capitalize()} cursors: click the second point"
            )
            return

        self._cursor_mode = None
        button = (
            self._horizontal_cursor_button
            if mode == "horizontal"
            else self._vertical_cursor_button
        )
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
            seconds = delta * self.decimation / ADC_SAMPLE_RATE_HZ
            self._cursor_readout.setText(
                f"Vertical cursors: Δsamples {delta:.2f}; Δt {self._format_time_per_div(seconds).replace('/div', '')}"
            )
            return

        ch1_volts = (
            delta
            * self._input_volts_per_div(1)
            * DISPLAY_VERTICAL_DIVISIONS
            / ADC_COUNTS
        )
        ch2_volts = (
            delta
            * self._input_volts_per_div(2)
            * DISPLAY_VERTICAL_DIVISIONS
            / ADC_COUNTS
        )
        self._cursor_readout.setText(
            f"Horizontal cursors: Δcodes {delta:.1f}; "
            f"ΔV CH1 {self._format_voltage(ch1_volts)}; "
            f"CH2 {self._format_voltage(ch2_volts)} (nominal)"
        )

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and watched in getattr(
            self, "_sidebar_wheel_controls", ()
        ):
            pixel_delta = event.pixelDelta().y()
            if pixel_delta:
                scroll_amount = pixel_delta
            else:
                wheel_steps = event.angleDelta().y() / 120.0
                scroll_amount = round(
                    wheel_steps
                    * self._sidebar_scroll.verticalScrollBar().singleStep()
                    * 3
                )
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
            self._gain_dial, self._gain_value, fields.get("afe_ch1_gain_pct")
        )
        self._set_float_dial_from_status(
            self._offset_dial, self._offset_value, fields.get("afe_ch1_offset_pct")
        )
        self._set_float_dial_from_status(
            self._gain2_dial, self._gain2_value, fields.get("afe_ch2_gain_pct")
        )
        self._set_float_dial_from_status(
            self._offset2_dial, self._offset2_value, fields.get("afe_ch2_offset_pct")
        )
        self._set_float_dial_from_status(
            self._trigger_dial, self._trigger_value, fields.get("afe_trigger_level_pct")
        )
        self._set_range_combo_from_status(
            self._ch1_range_combo, "ch1", fields.get("afe_ch1_range_vpp")
        )
        self._set_range_combo_from_status(
            self._ch2_range_combo, "ch2", fields.get("afe_ch2_range_vpp")
        )

        self._set_combo_text_from_status(
            self._atten_combo, fields.get("afe_ch1_atten", "1:1")
        )
        self._set_combo_text_from_status(
            self._coupling_combo, fields.get("afe_ch1_coupling", "dc").upper()
        )
        self._set_combo_text_from_status(
            self._ch2_atten_combo, fields.get("afe_ch2_atten", "1:1")
        )
        self._set_combo_text_from_status(
            self._ch2_coupling_combo, fields.get("afe_ch2_coupling", "dc").upper()
        )
        self._set_combo_text_from_status(
            self._trigger_source_combo, f"CH{fields.get('afe_trigger_source', '1')}"
        )
        self._set_combo_text_from_status(
            self._trigger_mode_combo,
            "Normal" if fields.get("trigger") == "normal" else "Off",
        )
        is_ch1_to_adc2 = fields.get("ch1_to_adc2") == "1"
        self._ch1_to_adc2_btn.blockSignals(True)
        self._ch1_to_adc2_btn.setChecked(is_ch1_to_adc2)
        self._ch1_to_adc2_btn.setText(
            f"CH1→ADC2: {'ON' if is_ch1_to_adc2 else 'OFF'}"
        )
        self._ch1_to_adc2_btn.blockSignals(False)
        self._update_trigger_line()

        try:
            capture_depth = int(fields["depth"])
            pretrigger = int(fields["pretrigger"])
            decimation = int(fields["decim"])
        except (KeyError, ValueError):
            pass
        else:
            retains_timebase_target = (
                self._timebase_target is not None
                and self.capture_size == capture_depth
                and self.decimation == decimation
            )
            self.capture_size = capture_depth
            self.pretrigger_size = pretrigger
            self.decimation = decimation
            self.timebase_seconds = self._time_per_div(self.decimation)
            if not retains_timebase_target:
                self._timebase_target = None
            self._update_sample_axis()
            self._set_combo_data_from_status(self._sample_size_combo, capture_depth)
            self._populate_pretrigger_options()
            self._set_combo_data_from_status(self._pretrigger_combo, pretrigger)
            self._set_timebase_combo(self._timebase_target)
            self._set_decimation_spinbox(decimation)

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
            f"({fields.get('afe_trigger_level_mv', '?')} mV)\n"
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
        label_layout.addWidget(
            self._make_help_button(
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
                "to 100%.",
            )
        )
        label_layout.addStretch()
        layout.addLayout(label_layout)
        combo = QComboBox()
        combo.addItem("1 Vpp", 1)
        combo.addItem("2 Vpp", 2)
        combo.setCurrentIndex(combo.findData(self._range_for_channel(channel)))
        combo.currentIndexChanged.connect(
            lambda _index, ch=channel, box=combo: self._on_adc_range_change(ch, box)
        )
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

    def _time_per_div(self, decimation: int) -> float:
        return (
            self.capture_size
            / DISPLAY_HORIZONTAL_DIVISIONS
            * decimation
            / ADC_SAMPLE_RATE_HZ
        )

    @staticmethod
    def _requested_decimation(time_per_div: float, capture_size: int) -> int:
        return round(
            time_per_div * DISPLAY_HORIZONTAL_DIVISIONS * ADC_SAMPLE_RATE_HZ / capture_size
        )

    @classmethod
    def _decimation_for(cls, time_per_div: float, capture_size: int) -> int:
        """Return the closest valid FPGA decimation for a requested timebase."""
        requested = cls._requested_decimation(time_per_div, capture_size)
        return max(MIN_DECIMATION, min(MAX_DECIMATION, requested))

    @staticmethod
    def _largest_compatible_capture_size(time_per_div: float) -> int:
        """Use the largest depth with a valid closest integer decimation."""
        for capture_size in reversed(CAPTURE_SIZES):
            requested = Oscilloscope._requested_decimation(
                time_per_div, capture_size
            )
            if MIN_DECIMATION <= requested <= MAX_DECIMATION:
                return capture_size
        return CAPTURE_SIZES[0]

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
        attenuation = (
            AFE_ATTEN_1_TO_100 if attenuation_name == "1:100" else AFE_ATTEN_1_TO_1
        )
        vga_linear = 10.0 ** (AFE_VGA_MAX_DB * gain_percent / 2000.0)
        return adc_range_vpp / (
            attenuation * vga_linear * AFE_DIFF_AMP_GAIN * DISPLAY_VERTICAL_DIVISIONS
        )

    def _update_scale_display(self) -> None:
        time_per_div = self._time_per_div(self.decimation)
        if self._timebase_target is None:
            timebase_text = f"Custom; actual {self._format_time_per_div(time_per_div)}"
        else:
            timebase_text = (
                f"target {self._format_time_per_div(self._timebase_target)}; "
                f"actual {self._format_time_per_div(time_per_div)}"
            )
        self._scale_label.setText(
            "Scale (nominal): "
            f"CH1 {self._input_volts_per_div(1):.4g} V/div; "
            f"CH2 {self._input_volts_per_div(2):.4g} V/div; "
            f"{timebase_text}; "
            f"({self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS:g} samples/div)"
        )

    def _set_timebase_combo(self, target: float | None) -> None:
        index = (
            self._timebase_custom_index
            if target is None
            else self._timebase_combo.findData(target)
        )
        if index < 0:
            index = self._timebase_custom_index
        self._timebase_combo.blockSignals(True)
        self._timebase_combo.setCurrentIndex(index)
        self._timebase_combo.blockSignals(False)

    def _set_decimation_spinbox(self, decimation: int) -> None:
        self._decimation_spinbox.blockSignals(True)
        self._decimation_spinbox.setValue(decimation)
        self._decimation_spinbox.blockSignals(False)

    def _range_for_channel(self, channel: str) -> int:
        value = self._settings.get("adc_range_vpp", {}).get(channel, 2)
        return value if value in (1, 2) else 2

    def _set_range_combo_from_status(
        self, combo: QComboBox, channel: str, value: str | None
    ) -> None:
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
        self._trigger_line.setVisible(
            self._trigger_mode_combo.currentText() == "Normal"
        )

    def _on_sample_size_change(self, _index):
        count = int(self._sample_size_combo.currentData())
        self.capture_size = count
        self._timebase_target = None
        self.timebase_seconds = self._time_per_div(self.decimation)
        self._set_timebase_combo(None)
        self._update_sample_axis()
        if self.pretrigger_size >= count:
            self.pretrigger_size = 0
            self._populate_pretrigger_options()
            self._send_control("afe pretrigger 0")
        else:
            self._populate_pretrigger_options()
        self._send_acquisition_configuration(count)

    def _on_decimation_change(self, decimation: int) -> None:
        self.decimation = decimation
        self._timebase_target = None
        self.timebase_seconds = self._time_per_div(decimation)
        self._set_timebase_combo(None)
        self._update_sample_axis()
        self._send_acquisition_configuration(self.capture_size)

    def _update_sample_axis(self):
        if self._plot_mode == "fft":
            self._configure_fft_axes()
            self._update_scale_display()
            return

        y_margin = (
            ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS * DISPLAY_VERTICAL_MARGIN_DIVISIONS
        )
        # Update the limits before the visible range.  The FFT view has much
        # narrower limits; applying the time-domain range first would clamp it
        # to the old FFT limits and leave the time series vertically distorted.
        self.plotWidget.getViewBox().setLimits(
            xMin=0,
            xMax=self.capture_size,
            yMin=-ADC_COUNTS // 2 - y_margin,
            yMax=ADC_COUNTS // 2 + y_margin,
        )
        self.plotWidget.setXRange(0, self.capture_size, padding=0)
        self.plotWidget.setYRange(
            -ADC_COUNTS // 2 - y_margin, ADC_COUNTS // 2 + y_margin, padding=0
        )
        self.plotWidget.setLabel("bottom", "Samples")
        self.plotWidget.setLabel("left", None)
        self.plotWidget.getAxis("left").setStyle(showValues=False)
        step = self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS
        ticks = [
            (step * index, str(int(step * index)))
            for index in range(DISPLAY_HORIZONTAL_DIVISIONS + 1)
        ]
        self.plotWidget.getAxis("bottom").setTicks([ticks])
        self._update_scale_display()

    def _configure_fft_axes(self) -> None:
        sample_rate_hz = ADC_SAMPLE_RATE_HZ / self.decimation
        nyquist_hz = sample_rate_hz / 2
        x_max_hz = min(self._fft_x_max_hz, nyquist_hz)
        if x_max_hz != self._fft_x_max_hz:
            self._fft_x_max_hz = x_max_hz
            self._fft_x_max_spinbox.blockSignals(True)
            self._fft_x_max_spinbox.setValue(x_max_hz / 1e6)
            self._fft_x_max_spinbox.blockSignals(False)
        # See _update_sample_axis: change constraints first so a previous
        # time-domain view cannot clamp the FFT view (or vice versa).
        self.plotWidget.getViewBox().setLimits(
            xMin=0, xMax=x_max_hz, yMin=-140, yMax=5
        )
        self.plotWidget.setXRange(0, x_max_hz, padding=0)
        self.plotWidget.setYRange(-140, 5, padding=0)
        self.plotWidget.setLabel("bottom", "Frequency", units="Hz")
        self.plotWidget.setLabel("left", "Magnitude", units="dBFS")
        self.plotWidget.getAxis("left").setStyle(showValues=True)
        self.plotWidget.getAxis("bottom").setTicks(None)

    def _set_plot_mode(self, plot_mode: str) -> None:
        """Switch the single top-bar mode tab to the requested display."""
        self._plot_mode = plot_mode
        is_time_series = self._plot_mode == "timeseries"
        self._time_series_menu_action.setChecked(is_time_series)
        self._fft_menu_action.setChecked(not is_time_series)
        self._clear_cursors()
        self._set_plot_toolbar_visibility(is_time_series)
        if is_time_series:
            self._update_sample_axis()
            self._update_trigger_line()
        else:
            self._configure_fft_axes()
            self._trigger_line.setVisible(False)
            self._cursor_readout.setText("FFT display")
        if self._have_frame:
            self._render_current_frame()

    def _set_plot_toolbar_visibility(self, is_time_series: bool) -> None:
        for action in self._cursor_toolbar_actions:
            action.setVisible(is_time_series)
        self._cursor_toolbar_separator.setVisible(is_time_series)
        for action in self._time_series_toolbar_actions:
            action.setVisible(is_time_series)
        self._fft_toolbar_separator.setVisible(not is_time_series)
        for action in self._fft_toolbar_actions:
            action.setVisible(not is_time_series)
        self._fft_x_max_spinbox.setEnabled(not is_time_series)
        self._fft_window_combo.setEnabled(not is_time_series)

    def _on_sample_points_toggle(self, checked: bool) -> None:
        self._show_sample_points = checked
        if self._plot_mode == "timeseries" and self._have_frame:
            self._render_time_series()

    def _reset_time_series_view(self) -> None:
        if self._plot_mode == "timeseries":
            self._update_sample_axis()

    def _on_plot_range_changed(self, *_args) -> None:
        if self._plot_mode == "timeseries" and self._have_frame:
            self._render_time_series()

    def _on_fft_x_max_change(self, value_mhz: float) -> None:
        self._fft_x_max_hz = value_mhz * 1e6
        if self._plot_mode == "fft":
            self._configure_fft_axes()

    def _on_fft_window_change(self, index: int) -> None:
        self._fft_window = self._fft_window_combo.itemData(index)
        if self._plot_mode == "fft" and self._have_frame:
            self._render_fft()

    def _on_pretrigger_change(self, _index):
        self.pretrigger_size = int(self._pretrigger_combo.currentData())
        self._send_control(f"afe pretrigger {self.pretrigger_size}")

    def _on_timebase_change(self, index: int):
        target = self._timebase_combo.itemData(index)
        if target is None:
            return
        self._timebase_target = float(target)
        count = self._largest_compatible_capture_size(self._timebase_target)
        self.capture_size = count
        self.decimation = self._decimation_for(self._timebase_target, count)
        self.timebase_seconds = self._time_per_div(self.decimation)
        self._set_combo_data_from_status(self._sample_size_combo, count)
        self._set_decimation_spinbox(self.decimation)
        if self.pretrigger_size >= count:
            self.pretrigger_size = 0
            self._populate_pretrigger_options()
            self._send("afe pretrigger 0")
        else:
            self._populate_pretrigger_options()
        self._update_sample_axis()
        self._send_acquisition_configuration(count)

    def _send_acquisition_configuration(self, capture_size: int) -> None:
        """Update depth and calculated decimation, then request one status reply."""
        sample_size_cmd = f"afe sample_size {capture_size}"
        decimation_cmd = f"afe decim {self.decimation}"
        self._cmd_panel.log_cmd(sample_size_cmd)
        self._cmd_panel.log_cmd(decimation_cmd)
        self._send(sample_size_cmd)
        self._send(decimation_cmd)
        self._send("status")

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

    def _on_ch1_to_adc2_change(self, checked: bool):
        self._ch1_to_adc2_btn.setText(f"CH1→ADC2: {'ON' if checked else 'OFF'}")
        self._send_control(f"afe ch1_to_adc2 {1 if checked else 0}")

    def _channel_afe_calculation(self, channel: int) -> str:
        prefix = f"afe_ch{channel}_"
        gain_percent = float(
            self._afe_state.get(
                f"{prefix}gain_pct",
                self._gain_value.value() if channel == 1 else self._gain2_value.value(),
            )
        )
        attenuation_name = self._afe_state.get(
            f"{prefix}atten",
            self._atten_combo.currentText()
            if channel == 1
            else self._ch2_atten_combo.currentText(),
        )
        range_vpp = float(
            self._afe_state.get(
                f"{prefix}range_vpp", self._range_for_channel(f"ch{channel}")
            )
        )

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
        input_lsb_v = input_full_scale_vpp / (2**ADC_BITS)

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
        trigger_percent = float(
            self._afe_state.get("afe_trigger_level_pct", self._trigger_value.value())
        )
        trigger_range = float(
            self._afe_state.get(
                f"afe_ch{trigger_channel}_range_vpp",
                self._range_for_channel(f"ch{trigger_channel}"),
            )
        )
        trigger_min = 1.5 - trigger_range / 4.0
        trigger_max = 1.5 + trigger_range / 4.0
        trigger_voltage = trigger_min + trigger_percent / 100.0 * (
            trigger_max - trigger_min
        )

        text = (
            "AFE configuration calculations (nominal)\n"
            "="
            * 45
            + "\n\n"
            + self._channel_afe_calculation(1)
            + "\n"
            + self._channel_afe_calculation(2)
            + "\n"
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
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        default_path = CAPTURES_DIR / f"capture_{ts}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save frame", str(default_path), "CSV file (*.csv)"
        )
        if not path:
            return

        save_capture_csv(
            path,
            self._ch1_raw,
            self._ch2_raw,
            fs_hz=ADC_SAMPLE_RATE_HZ / self.decimation,
            n_bits=14,
            metadata={
                "capture_name": os.path.basename(path),
                "decim_factor": int(self.decimation),
                "capture_depth": len(self._ch1_raw),
                "pretrigger_samples": int(self.pretrigger_size),
                "trigger_mode": self._trigger_mode_combo.currentText().lower(),
                "trigger_source": 2
                if self._trigger_source_combo.currentText() == "CH2"
                else 1,
                "adc_format": self._adc_format,
                **{
                    f"firmware_{key}": value
                    for key, value in self._afe_state.items()
                    if key.startswith("afe_") or key == "ch1_to_adc2"
                },
            },
        )
        self._cmd_panel.log_ok(f"Saved: {os.path.basename(path)}")

    def _raw_to_display_codes(self, raw: np.ndarray) -> np.ndarray:
        """Return signed ADC codes without applying gain, offset, or filtering."""
        half = ADC_COUNTS // 2
        if self._adc_format == "Offset Binary":
            return raw.astype(np.int32) - half
        else:
            return np.where(
                raw >= half, raw.astype(np.int32) - ADC_COUNTS, raw.astype(np.int32)
            )

    @staticmethod
    def _display_indices(sample_count: int) -> np.ndarray:
        """Limit FFT display points without changing the computed spectrum."""
        if sample_count <= MAX_DISPLAY_POINTS:
            return np.arange(sample_count, dtype=np.int32)
        return np.linspace(0, sample_count - 1, MAX_DISPLAY_POINTS, dtype=np.int32)

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
            self._have_frame = True
            self._render_current_frame()

    def _render_current_frame(self) -> None:
        if self._plot_mode == "fft":
            self._render_fft()
        else:
            self._render_time_series()

    def _render_time_series(self) -> None:
        start, stop = self._visible_sample_range(len(self._ch1_raw))
        sample_count = stop - start
        pixel_width = max(1, int(self.plotWidget.getViewBox().width()))
        use_envelope = not self._show_sample_points and sample_count > 2 * pixel_width

        if use_envelope:
            self._plot_x, self._ch1_data = self._min_max_envelope(
                self._ch1_raw, start, stop, pixel_width
            )
            _, self._ch2_data = self._min_max_envelope(
                self._ch2_raw, start, stop, pixel_width
            )
        else:
            self._plot_x = np.arange(start, stop, dtype=np.int32)
            self._ch1_data = self._raw_to_display_codes(self._ch1_raw[start:stop])
            self._ch2_data = self._raw_to_display_codes(self._ch2_raw[start:stop])
        if self.ch1_enabled:
            self._set_time_series_curve(
                self._curve_ch1, self._plot_x, self._ch1_data, "y"
            )
        if self.ch2_enabled:
            self._set_time_series_curve(
                self._curve_ch2, self._plot_x, self._ch2_data, "c"
            )

    def _visible_sample_range(self, sample_count: int) -> tuple[int, int]:
        x_min, x_max = self.plotWidget.getViewBox().viewRange()[0]
        start = min(sample_count - 1, max(0, int(np.floor(x_min))))
        stop = min(sample_count, int(np.ceil(x_max)) + 1)
        return start, max(start + 1, stop)

    def _min_max_envelope(
        self, raw: np.ndarray, start: int, stop: int, pixel_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Summarize each horizontal pixel column with its sample extrema."""
        codes = self._raw_to_display_codes(raw[start:stop])
        bucket_count = min(pixel_width, len(codes))
        edges = np.linspace(0, len(codes), bucket_count + 1, dtype=np.int32)
        bucket_starts = edges[:-1]
        bucket_ends = edges[1:]
        lows = np.minimum.reduceat(codes, bucket_starts)
        highs = np.maximum.reduceat(codes, bucket_starts)
        centers = start + (bucket_starts + bucket_ends - 1) / 2.0

        x = np.empty(bucket_count * 3, dtype=np.float64)
        y = np.empty(bucket_count * 3, dtype=np.float64)
        x[0::3] = centers
        x[1::3] = centers
        x[2::3] = np.nan
        y[0::3] = lows
        y[1::3] = highs
        y[2::3] = np.nan
        return x, y

    def _set_time_series_curve(
        self, curve, x: np.ndarray, y: np.ndarray, color: str
    ) -> None:
        if self._show_sample_points:
            curve.setData(
                x,
                y,
                pen=None,
                symbol="o",
                symbolSize=3,
                symbolPen=color,
                symbolBrush=color,
            )
        else:
            curve.setData(x, y, pen=color, symbol=None, connect="finite")

    def _render_fft(self) -> None:
        sample_count = len(self._ch1_raw)
        if sample_count < 2:
            return

        sample_rate_hz = ADC_SAMPLE_RATE_HZ / self.decimation
        frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
        window = {
            "rect": np.ones(sample_count),
            "hann": np.hanning(sample_count),
            "hamming": np.hamming(sample_count),
            "blackman": np.blackman(sample_count),
        }[self._fft_window]
        coherent_gain = window.sum() / 2.0

        def spectrum_dbfs(codes: np.ndarray) -> np.ndarray:
            codes = np.asarray(codes, dtype=np.float64)
            spectrum = np.fft.rfft(codes * window)
            magnitude = np.abs(spectrum) / coherent_gain
            magnitude[0] = np.abs(spectrum[0]) / window.sum()
            return 20.0 * np.log10(np.maximum(magnitude / (ADC_COUNTS / 2), 1e-12))

        indices = self._display_indices(len(frequencies))
        if self.ch1_enabled:
            self._curve_ch1.setData(
                frequencies[indices],
                spectrum_dbfs(self._raw_to_display_codes(self._ch1_raw))[indices],
                pen="y",
                symbol=None,
            )
        if self.ch2_enabled:
            self._curve_ch2.setData(
                frequencies[indices],
                spectrum_dbfs(self._raw_to_display_codes(self._ch2_raw))[indices],
                pen="c",
                symbol=None,
            )
