import datetime
import logging
import math
import os
import pathlib
import queue
import subprocess
from functools import lru_cache

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QActionGroup
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDial,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from analysis.capture_io import save_capture_csv
from ui.command_panel import CommandPanel
from utils.calibration import (
    ADC_FORMATS,
    clone_document,
    configuration_for_volts_per_div,
    interpolate_profile,
    load_calibration,
    profile_key,
    save_calibration,
)
from utils.controls import create_float_dial_widget
from utils.settings import load_settings, save_settings
from utils.vertical_scales import requested_volts_per_div_values

logger = logging.getLogger(__name__)

ADC_BITS = 14
ADC_COUNTS = 1 << ADC_BITS
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
    ("Hann", "hann"),
    ("Blackman", "blackman"),
)
FFT_Y_MIN_DBFS = -140
FFT_Y_MAX_DBFS = 5
FFT_Y_DIVISION_DB = 10
MAX_SERIES_CAPTURES = 1000
MAX_INVALID_SERIES_FRAMES = 3
DEFAULT_TIMEBASE_SECONDS = 10e-6
DECIMATION_STUDY_POINTS = (
    ("5 MHz, D = 1", 5_000_000.0, 1, "5MHz_d001"),
    ("1 MHz, D = 5", 1_000_000.0, 5, "1MHz_d005"),
    ("500 kHz, D = 10", 500_000.0, 10, "500kHz_d010"),
    ("100 kHz, D = 50", 100_000.0, 50, "100kHz_d050"),
    ("50 kHz, D = 100", 50_000.0, 100, "50kHz_d100"),
)

AFE_ATTEN_1_TO_1 = 953_000.0 / (49_900.0 + 953_000.0)
AFE_ATTEN_1_TO_100 = 10_000.0 / (1_000_000.0 + 10_000.0)
AFE_VGA_MAX_DB = 24.0
AFE_DIFF_AMP_GAIN = 1.0

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
MAX_CAPTURE_SAMPLES = 8192
MAX_PRETRIGGER_SAMPLES = MAX_CAPTURE_SAMPLES // 2
CAPTURE_SIZES = tuple(1 << exponent for exponent in range(7, 14))
MIN_DECIMATION = 1
MAX_DECIMATION = 1023
CAPTURES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "results" / "captures"
)


@lru_cache(maxsize=1)
def software_revision() -> tuple[str, str]:
    """Return the source revision and whether the software tree is modified."""
    project_dir = pathlib.Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown"
    return commit or "unknown", str(bool(dirty.strip())).lower()


def time_series_y_ticks() -> list[tuple[float, str]]:
    """Return the eight fixed vertical divisions used by the time-domain view."""
    step = ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
    return [
        (-ADC_COUNTS / 2 + step * index, "")
        for index in range(DISPLAY_VERTICAL_DIVISIONS + 1)
    ]


class Oscilloscope(QMainWindow):
    DISPLAY_SAMPLES = MAX_CAPTURE_SAMPLES

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
        self._frame_sequence: int | None = None
        self._batch_active = False
        self._batch_state = "idle"
        self._batch_base_path: pathlib.Path | None = None
        self._batch_metadata: dict[str, object] = {}
        self._batch_total = 0
        self._batch_saved = 0
        self._batch_invalid_frames = 0
        self._batch_previous_sequence: int | None = None
        self._status_generation = 0
        self._batch_required_status_generation = 0
        self._last_status_timestamp = ""
        self._is_connected = False
        self._plot_mode = "timeseries"
        self._show_sample_points = False
        self._fft_x_max_hz = ADC_SAMPLE_RATE_HZ / 2
        self._fft_window = "hann"
        self._afe_state: dict[str, str] = {}
        self._afe_info_text = "AFE state: waiting for firmware status…"
        self._settings = load_settings()
        self._calibration = load_calibration()
        self._normal_volts_per_div = dict(self._settings["normal_volts_per_div"])
        self._ui_mode = self._settings.get("ui_mode", "normal")
        if self._ui_mode not in ("normal", "advanced"):
            self._ui_mode = "normal"
        self._cursor_mode: str | None = None
        self._cursor_points = {"horizontal": [], "vertical": []}

        self.capture_size = self.DISPLAY_SAMPLES
        self.pretrigger_size = 0
        self.timebase_seconds = DEFAULT_TIMEBASE_SECONDS
        self._timebase_target: float | None = DEFAULT_TIMEBASE_SECONDS
        self.decimation = MIN_DECIMATION
        self._adc_format = "Offset Binary"
        self.ch1_enabled = True
        self.ch2_enabled = True
        self._build_ui()

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
        mode_menu = menu_bar.addMenu("Interface")
        self._normal_mode_action = mode_menu.addAction("Normal mode")
        self._normal_mode_action.setCheckable(True)
        self._normal_mode_action.triggered.connect(lambda: self._set_ui_mode("normal"))
        self._advanced_mode_action = mode_menu.addAction("Advanced mode")
        self._advanced_mode_action.setCheckable(True)
        self._advanced_mode_action.triggered.connect(
            lambda: self._set_ui_mode("advanced")
        )
        self._ui_mode_actions = QActionGroup(self)
        self._ui_mode_actions.setExclusive(True)
        self._ui_mode_actions.addAction(self._normal_mode_action)
        self._ui_mode_actions.addAction(self._advanced_mode_action)
        mode_menu.addSeparator()
        self._calibration_action = mode_menu.addAction("Calibration profiles…")
        self._calibration_action.triggered.connect(self._show_calibration_dialog)
        menu_bar.addSeparator()
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
        self._fft_menu_action.triggered.connect(lambda: self._set_plot_mode("fft"))
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
        self.plotWidget.getAxis("bottom").setTicks([x_ticks])
        self.plotWidget.getAxis("left").setTicks([time_series_y_ticks()])

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
        ctrl_frame.setMinimumWidth(320)
        ctrl_frame.setMaximumWidth(320)
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

        def add_control_group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            ctrl_layout.addWidget(group)
            return group, layout

        self._normal_controls_group, normal_controls_layout = add_control_group(
            "Oscilloscope controls"
        )
        self._acquisition_group, acquisition_layout = add_control_group("Acquisition")
        self._trigger_group, trigger_controls_layout = add_control_group("Trigger")
        self._ch1_afe_group, ch1_controls_layout = add_control_group("Channel 1 AFE")
        self._ch2_afe_group, ch2_controls_layout = add_control_group("Channel 2 AFE")
        self._display_group, display_controls_layout = add_control_group("Display")
        self._series_group, series_layout = add_control_group("Measurement series")
        self._connection_group, connection_layout = add_control_group("Connection")

        series_layout.addWidget(QLabel("Study"))
        self._series_study_combo = QComboBox()
        self._series_study_combo.addItem("Noise", "noise")
        self._series_study_combo.addItem("Harmonic distortion", "harmonics")
        self._series_study_combo.addItem("Gain influence", "gain")
        self._series_study_combo.addItem("Attenuation influence", "attenuation")
        self._series_study_combo.addItem("Frequency influence", "frequency")
        self._series_study_combo.addItem("Decimation influence", "decimation")
        self._series_study_combo.addItem("Channel averaging", "averaging")
        self._series_study_combo.addItem("Other", "other")
        self._series_study_combo.currentIndexChanged.connect(
            self._on_series_study_change
        )
        series_layout.addWidget(self._series_study_combo)

        self._series_decimation_point_label = QLabel("Decimation study point")
        series_layout.addWidget(self._series_decimation_point_label)
        self._series_decimation_point_combo = QComboBox()
        for label, frequency_hz, decimation, filename_token in DECIMATION_STUDY_POINTS:
            self._series_decimation_point_combo.addItem(
                label,
                {
                    "frequency_hz": frequency_hz,
                    "decimation": decimation,
                    "filename_token": filename_token,
                },
            )
        self._series_decimation_point_combo.currentIndexChanged.connect(
            self._on_series_decimation_point_change
        )
        series_layout.addWidget(self._series_decimation_point_combo)
        self._series_decimation_instruction = QLabel()
        self._series_decimation_instruction.setWordWrap(True)
        series_layout.addWidget(self._series_decimation_instruction)
        self._series_generator_frequency_confirmed = QCheckBox(
            "Generator frequency set to the selected point"
        )
        series_layout.addWidget(self._series_generator_frequency_confirmed)

        series_layout.addWidget(QLabel("Input condition"))
        self._series_input_combo = QComboBox()
        self._series_input_combo.addItem("Grounded inputs", "grounded")
        self._series_input_combo.addItem("BNC splitter", "generator_splitter")
        self._series_input_combo.addItem("Other", "other")
        series_layout.addWidget(self._series_input_combo)

        series_layout.addWidget(QLabel("Generator waveform"))
        self._series_waveform_combo = QComboBox()
        self._series_waveform_combo.addItem("Output disabled", "off")
        self._series_waveform_combo.addItem("Sine", "sine")
        series_layout.addWidget(self._series_waveform_combo)

        series_layout.addWidget(QLabel("Generator frequency (Hz)"))
        self._series_frequency = QDoubleSpinBox()
        self._series_frequency.setRange(0.0, 100_000_000.0)
        self._series_frequency.setDecimals(3)
        self._series_frequency.setSingleStep(1000.0)
        self._series_frequency.setValue(5_000_000.0)
        self._series_frequency.setKeyboardTracking(False)
        series_layout.addWidget(self._series_frequency)

        series_layout.addWidget(QLabel("Generator amplitude (Vpk)"))
        self._series_amplitude = QDoubleSpinBox()
        self._series_amplitude.setRange(0.0, 1000.0)
        self._series_amplitude.setDecimals(6)
        self._series_amplitude.setSingleStep(0.001)
        self._series_amplitude.setKeyboardTracking(False)
        series_layout.addWidget(self._series_amplitude)

        series_layout.addWidget(QLabel("Generator offset (V)"))
        self._series_generator_offset = QDoubleSpinBox()
        self._series_generator_offset.setRange(-1000.0, 1000.0)
        self._series_generator_offset.setDecimals(6)
        self._series_generator_offset.setSingleStep(0.001)
        self._series_generator_offset.setKeyboardTracking(False)
        series_layout.addWidget(self._series_generator_offset)

        series_layout.addWidget(QLabel("Generator load setting"))
        self._series_load_combo = QComboBox()
        self._series_load_combo.addItem("50 ohm", "50_ohm")
        self._series_load_combo.setToolTip(
            "The Hantek HDG6202 output load is fixed at 50 ohm for all studies."
        )
        series_layout.addWidget(self._series_load_combo)

        self._series_sense_confirmed = QCheckBox("SENSE jumpers verified")
        self._series_sense_confirmed.setToolTip(
            "Confirm that both physical SENSE jumpers match the ADC ranges "
            "reported by the device."
        )
        series_layout.addWidget(self._series_sense_confirmed)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Measurements"))
        self._series_count = QSpinBox()
        self._series_count.setRange(1, MAX_SERIES_CAPTURES)
        self._series_count.setValue(10)
        self._series_count.setKeyboardTracking(False)
        count_row.addWidget(self._series_count)
        series_layout.addLayout(count_row)

        self._series_btn = QPushButton("Capture series")
        self._series_btn.clicked.connect(self._toggle_measurement_series)
        self._series_btn.setEnabled(False)
        series_layout.addWidget(self._series_btn)
        self._series_progress = QLabel("Idle")
        self._series_progress.setWordWrap(True)
        series_layout.addWidget(self._series_progress)
        self._series_controls = (
            self._series_study_combo,
            self._series_decimation_point_combo,
            self._series_generator_frequency_confirmed,
            self._series_input_combo,
            self._series_waveform_combo,
            self._series_frequency,
            self._series_amplitude,
            self._series_generator_offset,
            self._series_load_combo,
            self._series_sense_confirmed,
            self._series_count,
        )
        self._on_series_study_change()

        self._gain_dial, self._gain_value = create_float_dial_widget(
            "Gain (%)", 0, 100, 50, ch1_controls_layout, self._on_gain_change
        )
        self._offset_dial, self._offset_value = create_float_dial_widget(
            "Offset (%)", 0, 100, 50, ch1_controls_layout, self._on_offset_change
        )
        self._ch1_range_combo = self._add_adc_range_control(
            "ADC range (SENSE jumper)", "ch1", ch1_controls_layout
        )

        self._gain2_dial, self._gain2_value = create_float_dial_widget(
            "Gain (%)", 0, 100, 50, ch2_controls_layout, self._on_gain2_change
        )
        self._offset2_dial, self._offset2_value = create_float_dial_widget(
            "Offset (%)", 0, 100, 50, ch2_controls_layout, self._on_offset2_change
        )
        self._ch2_range_combo = self._add_adc_range_control(
            "ADC range (SENSE jumper)", "ch2", ch2_controls_layout
        )

        trigger_layout = QHBoxLayout()
        trigger_layout.addWidget(QLabel("Trigger Level (AFE %)"))
        trigger_layout.addWidget(
            self._make_help_button(
                "Trigger level",
                "The trigger percentage is applied to the currently selected "
                "trigger-source channel.\n\n"
                "It is not a percentage of the DAC's full output range. The "
                "firmware converts it using the ADC range configured for that "
                "channel, centred at 1.5 V. That configuration must match the "
                "physical SENSE jumper.\n\n"
                "For a 1 Vpp range, 0–100% maps to 1.25–1.75 V. For a 2 Vpp "
                "range, it maps to 1.00–2.00 V. The configured range therefore "
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

        display_controls_layout.addWidget(QLabel("ADC code format"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(ADC_FORMATS)
        self._format_combo.setCurrentText(self._adc_format)
        self._format_combo.setToolTip(
            "Diagnostic interpretation of received ADC codes. "
            "Offset Binary is the default format used by the calibrated profiles."
        )
        self._format_combo.currentTextChanged.connect(self._on_format_change)
        display_controls_layout.addWidget(self._format_combo)
        self._advanced_calibrated_display = QCheckBox("Calibrated volts")
        self._advanced_calibrated_display.setChecked(False)
        self._advanced_calibrated_display.setToolTip(
            "Apply the saved calibration profile in Advanced mode. "
            "When cleared, display signed raw ADC codes."
        )
        self._advanced_calibrated_display.toggled.connect(self._update_sample_axis)
        display_controls_layout.addWidget(self._advanced_calibrated_display)

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
        acquisition_layout.addLayout(ch_row)
        acquisition_layout.addLayout(acq_row)

        # Normal mode uses dedicated proxy controls.  They remain synchronized
        # with the full set of service controls retained by Advanced mode.
        normal_controls_layout.addWidget(QLabel("Timebase"))
        self._normal_timebase_combo = QComboBox()
        for label, seconds in TIMEBASES:
            self._normal_timebase_combo.addItem(label, seconds)
        self._normal_timebase_combo.addItem("Custom", None)
        self._normal_timebase_combo.currentIndexChanged.connect(
            self._timebase_combo.setCurrentIndex
        )
        normal_controls_layout.addWidget(self._normal_timebase_combo)
        normal_controls_layout.addWidget(QLabel("CH1 V/div"))
        self._normal_ch1_scale_combo = QComboBox()
        self._normal_ch1_scale_combo.setToolTip(
            "Select a standard V/div scale. A verified calibration point with a "
            "sufficient input range is applied automatically."
        )
        self._normal_ch1_scale_combo.currentIndexChanged.connect(
            lambda _index: self._apply_normal_scale(1)
        )
        normal_controls_layout.addWidget(self._normal_ch1_scale_combo)
        normal_controls_layout.addWidget(QLabel("CH2 V/div"))
        self._normal_ch2_scale_combo = QComboBox()
        self._normal_ch2_scale_combo.setToolTip(
            "Select a standard V/div scale. A verified calibration point with a "
            "sufficient input range is applied automatically."
        )
        self._normal_ch2_scale_combo.currentIndexChanged.connect(
            lambda _index: self._apply_normal_scale(2)
        )
        normal_controls_layout.addWidget(self._normal_ch2_scale_combo)
        self._normal_offset_dial, self._normal_offset_value = create_float_dial_widget(
            "CH1 offset (%)", 0, 100, 50, normal_controls_layout,
            lambda: self._apply_normal_offset(1)
        )
        self._normal_offset2_dial, self._normal_offset2_value = create_float_dial_widget(
            "CH2 offset (%)", 0, 100, 50, normal_controls_layout,
            lambda: self._apply_normal_offset(2)
        )
        normal_controls_layout.addWidget(QLabel("CH1 coupling"))
        self._normal_ch1_coupling_combo = QComboBox()
        self._normal_ch1_coupling_combo.addItems(["DC", "AC"])
        self._normal_ch1_coupling_combo.currentTextChanged.connect(
            self._coupling_combo.setCurrentText
        )
        normal_controls_layout.addWidget(self._normal_ch1_coupling_combo)
        normal_controls_layout.addWidget(QLabel("CH2 coupling"))
        self._normal_ch2_coupling_combo = QComboBox()
        self._normal_ch2_coupling_combo.addItems(["DC", "AC"])
        self._normal_ch2_coupling_combo.currentTextChanged.connect(
            self._ch2_coupling_combo.setCurrentText
        )
        normal_controls_layout.addWidget(self._normal_ch2_coupling_combo)
        normal_controls_layout.addWidget(QLabel("Trigger source"))
        self._normal_trigger_source_combo = QComboBox()
        self._normal_trigger_source_combo.addItems(["CH1", "CH2"])
        self._normal_trigger_source_combo.currentTextChanged.connect(
            self._trigger_source_combo.setCurrentText
        )
        normal_controls_layout.addWidget(self._normal_trigger_source_combo)
        normal_controls_layout.addWidget(QLabel("Trigger mode"))
        self._normal_trigger_mode_combo = QComboBox()
        self._normal_trigger_mode_combo.addItems(["Off", "Normal"])
        self._normal_trigger_mode_combo.currentTextChanged.connect(
            self._trigger_mode_combo.setCurrentText
        )
        normal_controls_layout.addWidget(self._normal_trigger_mode_combo)
        self._normal_trigger_dial, self._normal_trigger_value = create_float_dial_widget(
            "Trigger level (%)", 0, 100, 50, normal_controls_layout,
            lambda: self._trigger_value.setValue(self._normal_trigger_value.value())
        )
        normal_controls_layout.addWidget(QLabel("Channels"))
        normal_ch_row = QHBoxLayout()
        self._normal_ch1_btn = QPushButton("CH1: ON")
        self._normal_ch1_btn.setCheckable(True)
        self._normal_ch1_btn.setChecked(True)
        self._normal_ch1_btn.toggled.connect(self._ch1_btn.setChecked)
        normal_ch_row.addWidget(self._normal_ch1_btn)
        self._normal_ch2_btn = QPushButton("CH2: ON")
        self._normal_ch2_btn.setCheckable(True)
        self._normal_ch2_btn.setChecked(True)
        self._normal_ch2_btn.toggled.connect(self._ch2_btn.setChecked)
        normal_ch_row.addWidget(self._normal_ch2_btn)
        normal_controls_layout.addLayout(normal_ch_row)

        normal_acq_row = QHBoxLayout()
        self._normal_run_btn = QPushButton("Run")
        self._normal_run_btn.setCheckable(True)
        self._normal_run_btn.toggled.connect(self._run_btn.setChecked)
        normal_acq_row.addWidget(self._normal_run_btn)
        self._normal_single_btn = QPushButton("Single")
        self._normal_single_btn.clicked.connect(self._single_acquire)
        normal_acq_row.addWidget(self._normal_single_btn)
        normal_controls_layout.addLayout(normal_acq_row)
        self._normal_save_btn = QPushButton("Save frame")
        self._normal_save_btn.clicked.connect(self._save_frame)
        normal_controls_layout.addWidget(self._normal_save_btn)
        self._normal_calibration_status = QLabel()
        self._normal_calibration_status.setWordWrap(True)
        self._normal_calibration_status.setMaximumWidth(248)
        normal_controls_layout.addWidget(self._normal_calibration_status)

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
            self._normal_ch1_scale_combo,
            self._normal_ch2_scale_combo,
            self._normal_ch1_coupling_combo,
            self._normal_ch2_coupling_combo,
            self._normal_timebase_combo,
            self._normal_trigger_source_combo,
            self._normal_trigger_mode_combo,
        ]
        self._set_hardware_controls_enabled(False)

        self._advanced_groups = (
            self._acquisition_group,
            self._trigger_group,
            self._ch1_afe_group,
            self._ch2_afe_group,
            self._display_group,
        )

        ctrl_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_widget)
        splitter.addWidget(self._cmd_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([680, 140])
        outer_layout.addWidget(splitter)
        self._refresh_normal_scale_controls()
        self._set_hardware_controls_enabled(False)
        self._set_ui_mode(self._ui_mode, persist=False)
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
        self._full_record_button.setToolTip(
            "Reset the time-domain view to the full capture"
        )
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
        magnitude = abs(value)
        if magnitude >= 1.0:
            return f"{value:.4g} V"
        if magnitude >= 1e-3:
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

        if self._can_display_calibrated_volts():
            if self._normal_dual_vertical_scale_active():
                self._cursor_readout.setText(
                    "Horizontal cursors: "
                    f"ΔCH1 {self._format_voltage(delta * self._normal_volts_per_div['ch1'])}; "
                    f"ΔCH2 {self._format_voltage(delta * self._normal_volts_per_div['ch2'])}"
                )
            else:
                self._cursor_readout.setText(
                    f"Horizontal cursors: ΔV {self._format_voltage(delta)}"
                )
        else:
            self._cursor_readout.setText(
                f"Horizontal cursors: Δcodes {delta:.1f} (uncalibrated)"
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
        self._is_connected = True
        self._series_sense_confirmed.setChecked(False)
        self._ch1_data[:] = 0
        self._ch2_data[:] = 0
        self._status_label.setText("Connected")
        self._run_btn.setEnabled(True)
        self._single_btn.setEnabled(True)
        self._normal_run_btn.setEnabled(True)
        self._normal_single_btn.setEnabled(True)
        self._series_btn.setEnabled(True)
        # Advanced controls may be used as soon as the TCP connection exists.
        # A subsequent status reply synchronizes their displayed values with
        # the device, but must not prevent selecting a custom acquisition setup.
        self._set_hardware_controls_enabled(True)
        self._send("status")

    def _on_disconnected(self):
        self._is_connected = False
        if self._batch_active:
            self._finish_measurement_series("Series stopped: device disconnected")
        self._status_label.setText("Disconnected")
        self._run_btn.setChecked(False)
        self._run_btn.setText("Run")
        self._single_btn.setEnabled(True)
        self._normal_run_btn.blockSignals(True)
        self._normal_run_btn.setChecked(False)
        self._normal_run_btn.blockSignals(False)
        self._normal_run_btn.setText("Run")
        self._normal_single_btn.setEnabled(True)
        self._series_btn.setEnabled(False)
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
        if self._batch_active and enabled:
            enabled = False
        for control in getattr(self, "_hardware_controls", ()):
            control.setEnabled(enabled)

    def _apply_firmware_status(self, line: str) -> None:
        fields = self._status_fields(line)
        self._status_generation += 1
        self._last_status_timestamp = datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        )
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
        self._ch1_to_adc2_btn.setText(f"CH1→ADC2: {'ON' if is_ch1_to_adc2 else 'OFF'}")
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
        self._refresh_normal_scale_controls()
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

        if (
            self._batch_active
            and self._status_generation >= self._batch_required_status_generation
        ):
            if self._batch_state == "waiting_initial_status":
                self._acquire_next_series_frame()
            elif self._batch_state == "waiting_frame_status":
                self._complete_current_series_frame()

    def _on_gain_change(self):
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._update_scale_display()
        self._send_control(f"afe gain 1 {self._gain_value.value():.2f}")

    def _on_offset_change(self):
        if hasattr(self, "_normal_offset_value"):
            self._normal_offset_value.blockSignals(True)
            self._normal_offset_value.setValue(self._offset_value.value())
            self._normal_offset_value.blockSignals(False)
        self._send_control(f"afe offset 1 {self._offset_value.value():.2f}")

    def _apply_normal_offset(self, channel: int) -> None:
        """Apply the normal-mode offset directly to the analogue front end.

        The offset is a live positioning control, independent of vertical
        calibration.  Sending the command here avoids relying on a signal
        emitted by a hidden advanced-mode widget.
        """
        if channel == 1:
            value = self._normal_offset_value.value()
            dial = self._offset_dial
            field = self._offset_value
        else:
            value = self._normal_offset2_value.value()
            dial = self._offset2_dial
            field = self._offset2_value
        dial.blockSignals(True)
        field.blockSignals(True)
        dial.setValue(round(value * 100))
        field.setValue(value)
        field.blockSignals(False)
        dial.blockSignals(False)
        self._send_control(f"afe offset {channel} {value:.2f}")

    def _add_adc_range_control(self, label: str, channel: str, layout) -> QComboBox:
        label_layout = QHBoxLayout()
        label_widget = QLabel(label.replace(" (", "\n("))
        label_widget.setWordWrap(True)
        label_layout.addWidget(label_widget)
        label_layout.addWidget(
            self._make_help_button(
                "ADC range and trigger level",
                "This control does not switch the ADC hardware. Select the "
                "range that matches the physical SENSE jumper on this channel: "
                "SENSE connected to VDD selects 2 Vpp, while SENSE connected "
                "to VOCM selects 1 Vpp. It does not rescale the plotted samples, "
                "which remain raw ADC codes.\n\n"
                "This configuration also defines the trigger threshold voltage when "
                "this channel is selected as the trigger source. The displayed "
                "threshold is referenced to the positive input leg and is centred "
                "at 1.5 V:\n"
                "• 1 Vpp differential: 1.25 V to 1.75 V\n"
                "• 2 Vpp differential: 1.00 V to 2.00 V\n\n"
                "The Trigger Level control spans this signal range from 0% to "
                "100%. The comparator is connected to the negative leg, so its "
                "DAC threshold voltage decreases as the displayed level increases.",
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

    def _set_ui_mode(self, mode: str, *, persist: bool = True) -> None:
        """Switch between ordinary calibrated and service-oriented controls."""
        if mode not in ("normal", "advanced"):
            raise ValueError(f"unsupported UI mode: {mode}")
        self._ui_mode = mode
        is_advanced = mode == "advanced"
        self._normal_controls_group.setVisible(not is_advanced)
        for group in getattr(self, "_advanced_groups", ()):
            group.setVisible(is_advanced)
        self._normal_mode_action.setChecked(mode == "normal")
        self._advanced_mode_action.setChecked(is_advanced)
        self._calibration_action.setEnabled(is_advanced)
        self._calibration_action.setToolTip(
            "Available in Advanced mode"
            if not is_advanced
            else "Edit saved calibration profiles"
        )
        if persist:
            self._settings["ui_mode"] = mode
            save_settings(self._settings)
        self._refresh_normal_scale_controls()
        self._update_sample_axis()

    def _channel_gain_percent(self, channel: int) -> float:
        key = f"afe_ch{channel}_gain_pct"
        fallback = (
            self._gain_value.value() if channel == 1 else self._gain2_value.value()
        )
        try:
            return float(self._afe_state.get(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def _channel_afe_offset_percent(self, channel: int) -> float:
        """Return the reported AFE offset, or the current UI setting before status arrives."""
        key = f"afe_ch{channel}_offset_pct"
        fallback = (
            self._offset_value.value() if channel == 1 else self._offset2_value.value()
        )
        try:
            return float(self._afe_state.get(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    def _channel_attenuation(self, channel: int) -> str:
        key = f"afe_ch{channel}_atten"
        fallback = (
            self._atten_combo.currentText()
            if channel == 1
            else self._ch2_atten_combo.currentText()
        )
        attenuation = self._afe_state.get(key, fallback)
        return attenuation if attenuation in ("1:1", "1:100") else fallback

    def _calibration_profile_for_channel(self, channel: int) -> dict | None:
        attenuation = self._channel_attenuation(channel)
        sense_vpp = self._range_for_channel(f"ch{channel}")
        profile = self._calibration["profiles"].get(
            profile_key(channel, attenuation, sense_vpp)
        )
        if profile is None:
            return None
        if float(profile["sense_vpp"]) != float(
            self._range_for_channel(f"ch{channel}")
        ):
            return None
        if profile["adc_format"] != self._adc_format:
            return None
        return profile

    def _calibration_for_channel(self, channel: int) -> tuple[float, float] | None:
        profile = self._calibration_profile_for_channel(channel)
        if profile is None:
            return None
        calibration = interpolate_profile(profile, self._channel_gain_percent(channel))
        if calibration is None:
            return None
        # The AFE offset is a live percentage control.  It must shift the
        # displayed trace, rather than be cancelled by a stored voltage offset.
        return calibration, 0.0

    def _calibrated_volts_per_div(self, channel: int) -> float | None:
        calibration = self._calibration_for_channel(channel)
        if calibration is None:
            return None
        return abs(calibration[0]) * ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS

    def _can_display_calibrated_volts(self) -> bool:
        if self._ui_mode == "advanced" and not self._advanced_calibrated_display.isChecked():
            return False
        active_channels = []
        if self.ch1_enabled:
            active_channels.append(1)
        if self.ch2_enabled:
            active_channels.append(2)
        return bool(active_channels) and all(
            self._calibration_for_channel(channel) is not None
            for channel in active_channels
        )

    def _time_display_values(self, raw: np.ndarray, channel: int) -> np.ndarray:
        """Return values in displayed vertical divisions or raw ADC codes."""
        codes = self._raw_to_display_codes(raw)
        if not self._can_display_calibrated_volts():
            return codes
        slope, offset = self._calibration_for_channel(channel)
        volts = codes.astype(np.float64) * slope + offset
        if self._ui_mode == "normal":
            volts_per_div = self._normal_volts_per_div.get(f"ch{channel}")
            if isinstance(volts_per_div, (int, float)) and volts_per_div > 0:
                return volts / float(volts_per_div)
        return volts

    def _input_volts(self, raw: np.ndarray, channel: int) -> np.ndarray:
        """Return calibrated input voltage without display-scale normalization."""
        codes = self._raw_to_display_codes(raw)
        slope, offset = self._calibration_for_channel(channel)
        return codes.astype(np.float64) * slope + offset

    def _time_axis_limits(self) -> tuple[float, float, str, bool]:
        """Return vertical range, label and whether tick labels are meaningful."""
        if self._ui_mode == "advanced" and not self._advanced_calibrated_display.isChecked():
            margin = ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS * DISPLAY_VERTICAL_MARGIN_DIVISIONS
            return (
                -ADC_COUNTS // 2 - margin,
                ADC_COUNTS // 2 + margin,
                "ADC code",
                True,
            )
        if not self._can_display_calibrated_volts():
            margin = (
                ADC_COUNTS
                / DISPLAY_VERTICAL_DIVISIONS
                * DISPLAY_VERTICAL_MARGIN_DIVISIONS
            )
            return (
                -ADC_COUNTS // 2 - margin,
                ADC_COUNTS // 2 + margin,
                "",
                False,
            )
        if (
            self._ui_mode == "normal"
            and self._can_display_calibrated_volts()
            and self._shared_normal_volts_per_div() is not None
        ):
            divisions = DISPLAY_VERTICAL_DIVISIONS / 2
            return (
                -divisions - DISPLAY_VERTICAL_MARGIN_DIVISIONS,
                divisions + DISPLAY_VERTICAL_MARGIN_DIVISIONS,
                (
                    "CH1 input voltage"
                    if self._normal_dual_vertical_scale_active()
                    else "Input voltage"
                ),
                True,
            )
        selected_vdiv = self._shared_normal_volts_per_div()
        if selected_vdiv is not None:
            signal_limit = selected_vdiv * DISPLAY_VERTICAL_DIVISIONS / 2
            margin = selected_vdiv * DISPLAY_VERTICAL_MARGIN_DIVISIONS
            return (
                -signal_limit - margin,
                signal_limit + margin,
                "Input voltage",
                True,
            )

        limits = []
        for channel in (1, 2):
            if not (self.ch1_enabled if channel == 1 else self.ch2_enabled):
                continue
            slope, offset = self._calibration_for_channel(channel)
            limits.extend(
                (
                    offset - abs(slope) * ADC_COUNTS / 2,
                    offset + abs(slope) * ADC_COUNTS / 2,
                )
            )
        lower, upper = min(limits), max(limits)
        span = max(upper - lower, 1e-9)
        margin = span * DISPLAY_VERTICAL_MARGIN_DIVISIONS / DISPLAY_VERTICAL_DIVISIONS
        return lower - margin, upper + margin, "Input voltage", True

    def _shared_normal_volts_per_div(self) -> float | None:
        """Return the common grid scale required by the currently active channels."""
        selected = []
        for channel in (1, 2):
            if not (self.ch1_enabled if channel == 1 else self.ch2_enabled):
                continue
            value = self._normal_volts_per_div.get(f"ch{channel}")
            if isinstance(value, (int, float)) and value > 0:
                selected.append(float(value))
        return max(selected) if selected else None

    def _normal_dual_vertical_scale_active(self) -> bool:
        """Return whether Normal mode can show independent CH1 and CH2 scales."""
        return (
            self._ui_mode == "normal"
            and self.ch1_enabled
            and self.ch2_enabled
            and self._can_display_calibrated_volts()
            and all(
                isinstance(self._normal_volts_per_div.get(f"ch{channel}"), (int, float))
                and self._normal_volts_per_div[f"ch{channel}"] > 0
                for channel in (1, 2)
            )
        )

    def _refresh_normal_scale_controls(self) -> None:
        """Offer requested V/div scales that the saved calibration can realize."""
        if not hasattr(self, "_normal_ch1_scale_combo"):
            return
        statuses = []
        for channel, combo in (
            (1, self._normal_ch1_scale_combo),
            (2, self._normal_ch2_scale_combo),
        ):
            combo.blockSignals(True)
            combo.clear()
            options: dict[float, tuple[str, float]] = {}
            sense_vpp = self._range_for_channel(f"ch{channel}")
            for attenuation in ("1:1", "1:100"):
                profile = self._calibration["profiles"][
                    profile_key(channel, attenuation, sense_vpp)
                ]
                if float(profile["sense_vpp"]) != float(
                    self._range_for_channel(f"ch{channel}")
                ):
                    continue
                if profile["adc_format"] != self._adc_format:
                    continue
                for standard_vdiv in requested_volts_per_div_values(attenuation):
                    configuration = configuration_for_volts_per_div(
                        profile,
                        standard_vdiv,
                        adc_counts=ADC_COUNTS,
                        divisions=DISPLAY_VERTICAL_DIVISIONS,
                    )
                    if configuration is None:
                        continue
                    gain_pct, _ = configuration
                    existing = options.get(standard_vdiv)
                    if existing is None or gain_pct > existing[1]:
                        options[standard_vdiv] = (
                            attenuation,
                            gain_pct,
                        )
            if options:
                for standard_vdiv in sorted(options, reverse=True):
                    attenuation, gain_pct = options[standard_vdiv]
                    combo.addItem(
                        f"{self._format_voltage(standard_vdiv)}/div",
                        {
                            "volts_per_div": standard_vdiv,
                            "attenuation": attenuation,
                            "gain_pct": gain_pct,
                        },
                    )
                saved_vdiv = self._normal_volts_per_div.get(f"ch{channel}")
                best_index = next(
                    (
                        index
                        for index in range(combo.count())
                        if combo.itemData(index)["volts_per_div"] == saved_vdiv
                    ),
                    None,
                )
                if best_index is None:
                    combo.insertItem(0, "Select V/div")
                    combo.setCurrentIndex(0)
                    statuses.append(f"CH{channel}: select V/div")
                else:
                    combo.setCurrentIndex(best_index)
                    statuses.append(f"CH{channel}: calibrated")
                combo.setEnabled(True)
            else:
                combo.addItem("Calibration required")
                combo.setEnabled(False)
                statuses.append(f"CH{channel}: no saved measured scale")
            combo.blockSignals(False)
        if not self._can_display_calibrated_volts():
            statuses.append("Voltage axis requires calibration of every active channel")
        self._normal_calibration_status.setText(" · ".join(statuses))

    def _apply_normal_scale(self, channel: int) -> None:
        combo = (
            self._normal_ch1_scale_combo
            if channel == 1
            else self._normal_ch2_scale_combo
        )
        selection = combo.currentData()
        if not isinstance(selection, dict):
            return
        self._normal_volts_per_div[f"ch{channel}"] = selection["volts_per_div"]
        self._settings["normal_volts_per_div"] = dict(self._normal_volts_per_div)
        save_settings(self._settings)
        attenuation_combo = self._atten_combo if channel == 1 else self._ch2_atten_combo
        gain_widget = self._gain_value if channel == 1 else self._gain2_value
        attenuation_combo.setCurrentText(selection["attenuation"])
        gain_widget.setValue(selection["gain_pct"])
        self._update_sample_axis()

    def _show_calibration_dialog(self) -> None:
        """Edit discrete V/div-to-gain calibration profiles."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibration profiles")
        dialog.resize(760, 460)
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Each row is one requested V/div sensitivity. Enter the measured "
                "VGA setting and V/code coefficient. The AFE offset is a separate "
                "live percentage control and is not stored in this profile."
            )
        )
        profile_combo = QComboBox()
        for channel in (1, 2):
            for sense_vpp in (1.0, 2.0):
                for attenuation in ("1:1", "1:100"):
                    profile_combo.addItem(
                        f"CH{channel}, {attenuation}, {sense_vpp:g} Vpp",
                        profile_key(channel, attenuation, sense_vpp),
                    )
        layout.addWidget(profile_combo)

        configuration_row = QHBoxLayout()
        configuration_row.addWidget(QLabel("SENSE range"))
        sense_combo = QComboBox()
        sense_combo.addItem("1 Vpp", 1.0)
        sense_combo.addItem("2 Vpp", 2.0)
        sense_combo.setEnabled(False)
        configuration_row.addWidget(sense_combo)
        configuration_row.addWidget(QLabel("ADC code format"))
        format_combo = QComboBox()
        format_combo.addItems(ADC_FORMATS)
        configuration_row.addWidget(format_combo)
        configuration_row.addStretch()
        layout.addLayout(configuration_row)

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["V/div", "VGA (%)", "V/code"])
        table.verticalHeader().setVisible(False)
        layout.addWidget(table)

        working = clone_document(self._calibration)

        active_key = profile_combo.currentData()

        def write_current_profile(key: str | None = None) -> bool:
            key = active_key if key is None else key
            profile = working["profiles"][key]
            try:
                for row, point in enumerate(profile["points"]):
                    gain_text = table.item(row, 1).text().strip()
                    slope_text = table.item(row, 2).text().strip()
                    point["gain_pct"] = None if not gain_text else float(gain_text)
                    point["volts_per_code"] = None if not slope_text else float(slope_text)
            except (AttributeError, ValueError):
                QMessageBox.warning(
                    dialog,
                    "Invalid calibration",
                    "Each row requires numeric VGA and V/code settings, or both "
                    "fields must remain blank.",
                )
                return False
            profile["adc_format"] = format_combo.currentText()
            return True

        def load_profile(_index: int | None = None) -> None:
            key = profile_combo.currentData()
            profile = working["profiles"][key]
            sense_combo.setCurrentIndex(sense_combo.findData(profile["sense_vpp"]))
            format_combo.setCurrentText(profile["adc_format"])
            table.setRowCount(len(profile["points"]))
            for row, point in enumerate(profile["points"]):
                values = (
                    f"{point['volts_per_div']:g}",
                    "" if point["gain_pct"] is None else f"{point['gain_pct']:g}",
                    ""
                    if point["volts_per_code"] is None
                    else f"{point['volts_per_code']:.12g}",
                )
                for column, value in enumerate(values):
                    table.setItem(row, column, QTableWidgetItem(value))

        def change_profile(index: int) -> None:
            nonlocal active_key
            if not write_current_profile(active_key):
                profile_combo.blockSignals(True)
                profile_combo.setCurrentIndex(profile_combo.findData(active_key))
                profile_combo.blockSignals(False)
                return
            active_key = profile_combo.currentData()
            load_profile()

        profile_combo.currentIndexChanged.connect(change_profile)
        load_profile()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )

        def save_and_close() -> None:
            if not write_current_profile():
                return
            try:
                save_calibration(working)
            except ValueError as exc:
                QMessageBox.warning(dialog, "Invalid calibration", str(exc))
                return
            self._calibration = load_calibration()
            self._refresh_normal_scale_controls()
            self._update_sample_axis()
            dialog.accept()

        buttons.accepted.connect(save_and_close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
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
            time_per_div
            * DISPLAY_HORIZONTAL_DIVISIONS
            * ADC_SAMPLE_RATE_HZ
            / capture_size
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
            requested = Oscilloscope._requested_decimation(time_per_div, capture_size)
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
        calibrated = self._calibrated_volts_per_div(channel)
        if calibrated is not None:
            return calibrated
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
        ch1_scale = self._normal_volts_per_div.get("ch1") or self._calibrated_volts_per_div(1)
        ch2_scale = self._normal_volts_per_div.get("ch2") or self._calibrated_volts_per_div(2)
        ch1_text = (
            "uncalibrated"
            if ch1_scale is None
            else f"{self._format_voltage(ch1_scale)}/div"
        )
        ch2_text = (
            "uncalibrated"
            if ch2_scale is None
            else f"{self._format_voltage(ch2_scale)}/div"
        )
        self._scale_label.setText(
            "Scale: "
            f"CH1 {ch1_text}; "
            f"CH2 {ch2_text}; "
            f"{timebase_text}; "
            f"({self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS:g} samples/div)"
        )
        self.plotWidget.setTitle("")

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
        if hasattr(self, "_normal_timebase_combo"):
            self._normal_timebase_combo.blockSignals(True)
            self._normal_timebase_combo.setCurrentIndex(index)
            self._normal_timebase_combo.blockSignals(False)

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
        if hasattr(self, "_series_sense_confirmed"):
            self._series_sense_confirmed.setChecked(False)
        self._settings.setdefault("adc_range_vpp", {})[channel] = vpp
        save_settings(self._settings)
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._send_control(f"afe range {1 if channel == 'ch1' else 2} {vpp}")

    def _on_gain2_change(self):
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._update_scale_display()
        self._send_control(f"afe gain 2 {self._gain2_value.value():.2f}")

    def _on_offset2_change(self):
        if hasattr(self, "_normal_offset2_value"):
            self._normal_offset2_value.blockSignals(True)
            self._normal_offset2_value.setValue(self._offset2_value.value())
            self._normal_offset2_value.blockSignals(False)
        self._send_control(f"afe offset 2 {self._offset2_value.value():.2f}")

    def _populate_pretrigger_options(self):
        """Offer only encodable pretrigger depths smaller than capture depth."""
        selected = self.pretrigger_size
        self._pretrigger_combo.blockSignals(True)
        self._pretrigger_combo.clear()
        self._pretrigger_combo.addItem("Off", 0)
        count = 1
        while count < self.capture_size and count <= MAX_PRETRIGGER_SAMPLES:
            self._pretrigger_combo.addItem(str(count), count)
            count *= 2
        index = self._pretrigger_combo.findData(selected)
        self._pretrigger_combo.setCurrentIndex(max(index, 0))
        self._pretrigger_combo.blockSignals(False)

    def _on_trigger_change(self):
        if hasattr(self, "_normal_trigger_value"):
            self._normal_trigger_value.blockSignals(True)
            self._normal_trigger_value.setValue(self._trigger_value.value())
            self._normal_trigger_value.blockSignals(False)
        self._update_trigger_line()
        self._send_control(f"afe trigger_level {self._trigger_value.value():.2f}")

    def _on_trigger_source_change(self, source: str):
        if hasattr(self, "_normal_trigger_source_combo"):
            self._normal_trigger_source_combo.blockSignals(True)
            self._normal_trigger_source_combo.setCurrentText(source)
            self._normal_trigger_source_combo.blockSignals(False)
        ch = "2" if source == "CH2" else "1"
        self._send_control(f"afe trigger_ch {ch}")

    def _on_trigger_mode_change(self, mode_text: str):
        if hasattr(self, "_normal_trigger_mode_combo"):
            self._normal_trigger_mode_combo.blockSignals(True)
            self._normal_trigger_mode_combo.setCurrentText(mode_text)
            self._normal_trigger_mode_combo.blockSignals(False)
        mode = "normal" if mode_text == "Normal" else "off"
        self._update_trigger_line()
        self._send_control(f"afe trigger_mode {mode}")

    def _update_trigger_line(self):
        trigger_percent = self._trigger_value.value()
        display_code = ADC_COUNTS * trigger_percent / 100.0 - ADC_COUNTS / 2
        self._trigger_line.setPos(display_code)
        self._trigger_line.setVisible(
            self._trigger_mode_combo.currentText() == "Normal"
            and not self._can_display_calibrated_volts()
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

        y_min, y_max, y_label, show_y_values = self._time_axis_limits()
        # Update the limits before the visible range.  The FFT view has much
        # narrower limits; applying the time-domain range first would clamp it
        # to the old FFT limits and leave the time series vertically distorted.
        self.plotWidget.getViewBox().setLimits(
            xMin=0,
            xMax=self.capture_size,
            yMin=y_min,
            yMax=y_max,
        )
        self.plotWidget.setXRange(0, self.capture_size, padding=0)
        self.plotWidget.setYRange(y_min, y_max, padding=0)
        self.plotWidget.setLabel("bottom", "Samples")
        self.plotWidget.setLabel("left", y_label, units="V" if y_label == "Input voltage" else None)
        self.plotWidget.getAxis("left").setStyle(showValues=show_y_values)
        dual_scale = self._normal_dual_vertical_scale_active()
        self.plotWidget.getPlotItem().showAxis("right", show=dual_scale)
        if dual_scale:
            ch1_vdiv = float(self._normal_volts_per_div["ch1"])
            ch2_vdiv = float(self._normal_volts_per_div["ch2"])
            divisions = range(
                -DISPLAY_VERTICAL_DIVISIONS // 2,
                DISPLAY_VERTICAL_DIVISIONS // 2 + 1,
            )
            self.plotWidget.setLabel("left", "CH1 input voltage", units="V")
            self.plotWidget.setLabel("right", "CH2 input voltage", units="V")
            self.plotWidget.getAxis("left").setTicks(
                [[(division, self._format_voltage(division * ch1_vdiv)) for division in divisions]]
            )
            self.plotWidget.getAxis("right").setStyle(showValues=True)
            self.plotWidget.getAxis("right").setTicks(
                [[(division, self._format_voltage(division * ch2_vdiv)) for division in divisions]]
            )
        else:
            self.plotWidget.getAxis("right").setTicks(None)
        if show_y_values:
            if dual_scale:
                self._update_scale_display()
                return
            if self._ui_mode == "advanced" and not self._advanced_calibrated_display.isChecked():
                self.plotWidget.getAxis("left").setTicks(
                    [[(value, str(value)) for value, _ in time_series_y_ticks()]]
                )
                self._update_scale_display()
                return
            selected_vdiv = self._shared_normal_volts_per_div()
            if selected_vdiv is None:
                self.plotWidget.getAxis("left").setTicks(None)
            else:
                self.plotWidget.getAxis("left").setTicks(
                    [
                        [
                            (
                                division,
                                self._format_voltage(division * selected_vdiv),
                            )
                            for division in range(
                                -DISPLAY_VERTICAL_DIVISIONS // 2,
                                DISPLAY_VERTICAL_DIVISIONS // 2 + 1,
                            )
                        ]
                    ]
                )
        else:
            # _configure_fft_axes installs dBFS ticks explicitly. Restore the raw
            # ADC-code divisions when returning to the time-domain view.
            self.plotWidget.getAxis("left").setTicks([time_series_y_ticks()])
        step = self.capture_size / DISPLAY_HORIZONTAL_DIVISIONS
        ticks = [
            (step * index, str(int(step * index)))
            for index in range(DISPLAY_HORIZONTAL_DIVISIONS + 1)
        ]
        self.plotWidget.getAxis("bottom").setTicks([ticks])
        self._update_scale_display()

    def _configure_fft_axes(self) -> None:
        # FFT uses one common dBFS axis.  The independent CH1/CH2 voltage
        # scales are meaningful only for the time-domain display.
        plot_item = self.plotWidget.getPlotItem()
        plot_item.showAxis("right", show=False)
        self.plotWidget.getAxis("right").setTicks(None)
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
            xMin=0, xMax=x_max_hz, yMin=FFT_Y_MIN_DBFS, yMax=FFT_Y_MAX_DBFS
        )
        self.plotWidget.setXRange(0, x_max_hz, padding=0)
        self.plotWidget.setYRange(FFT_Y_MIN_DBFS, FFT_Y_MAX_DBFS, padding=0)
        self.plotWidget.setLabel("bottom", "Frequency", units="Hz")
        self.plotWidget.setLabel("left", "Magnitude", units="dBFS")
        fft_y_ticks = [
            (value, str(value))
            for value in range(FFT_Y_MIN_DBFS, FFT_Y_MAX_DBFS, FFT_Y_DIVISION_DB)
        ]
        self.plotWidget.getAxis("left").setTicks([fft_y_ticks])
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
        """Select a diagnostic display interpretation without changing the device."""
        self._adc_format = fmt
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._update_scale_display()

    def _on_coupling_change(self, selection: str):
        coupling = "ac" if selection == "AC" else "dc"
        self._send_control(f"afe coupling 1 {coupling}")

    def _on_attenuation_change(self, selection: str):
        atten = "100" if selection == "1:100" else "1"
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._update_scale_display()
        self._send_control(f"afe atten 1 {atten}")

    def _on_ch2_coupling_change(self, selection: str):
        coupling = "ac" if selection == "AC" else "dc"
        self._send_control(f"afe coupling 2 {coupling}")

    def _on_ch2_attenuation_change(self, selection: str):
        atten = "100" if selection == "1:100" else "1"
        self._refresh_normal_scale_controls()
        self._update_sample_axis()
        self._update_scale_display()
        self._send_control(f"afe atten 2 {atten}")

    def _on_ch1_toggle(self, checked: bool):
        self.ch1_enabled = checked
        self._ch1_btn.setText(f"CH1: {'ON' if checked else 'OFF'}")
        if hasattr(self, "_normal_ch1_btn"):
            self._normal_ch1_btn.blockSignals(True)
            self._normal_ch1_btn.setChecked(checked)
            self._normal_ch1_btn.setText(f"CH1: {'ON' if checked else 'OFF'}")
            self._normal_ch1_btn.blockSignals(False)
        self._curve_ch1.setVisible(checked)
        self._refresh_normal_scale_controls()
        self._update_sample_axis()

    def _on_ch2_toggle(self, checked: bool):
        self.ch2_enabled = checked
        self._ch2_btn.setText(f"CH2: {'ON' if checked else 'OFF'}")
        if hasattr(self, "_normal_ch2_btn"):
            self._normal_ch2_btn.blockSignals(True)
            self._normal_ch2_btn.setChecked(checked)
            self._normal_ch2_btn.setText(f"CH2: {'ON' if checked else 'OFF'}")
            self._normal_ch2_btn.blockSignals(False)
        self._curve_ch2.setVisible(checked)
        self._refresh_normal_scale_controls()
        self._update_sample_axis()

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
        signal_threshold_voltage = trigger_min + trigger_percent / 100.0 * (
            trigger_max - trigger_min
        )
        comparator_threshold_voltage = trigger_max - trigger_percent / 100.0 * (
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
            f"  Positive-leg equivalent: {trigger_percent:.2f} % → "
            f"{signal_threshold_voltage:.4f} V\n"
            f"  DAC threshold at negative comparator input: "
            f"{comparator_threshold_voltage:.4f} V\n\n"
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
        self._normal_run_btn.blockSignals(True)
        self._normal_run_btn.setChecked(checked)
        self._normal_run_btn.blockSignals(False)
        if checked:
            self._run_btn.setText("Stop")
            self._normal_run_btn.setText("Stop")
            self._single_btn.setEnabled(False)
            self._normal_single_btn.setEnabled(False)
            if self._conn_mgr:
                self._conn_mgr.start_acquisition("continuous")
        else:
            self._run_btn.setText("Run")
            self._normal_run_btn.setText("Run")
            self._single_btn.setEnabled(True)
            self._normal_single_btn.setEnabled(True)
            if self._conn_mgr:
                self._conn_mgr.stop_acquisition()

    def _on_series_study_change(self) -> None:
        if not hasattr(self, "_series_study_combo"):
            return
        study = self._series_study_combo.currentData()
        is_noise = study == "noise"
        is_decimation = study == "decimation"
        self._series_decimation_point_label.setVisible(is_decimation)
        self._series_decimation_point_combo.setVisible(is_decimation)
        self._series_decimation_instruction.setVisible(is_decimation)
        self._series_generator_frequency_confirmed.setVisible(is_decimation)
        self._series_input_combo.setCurrentIndex(
            self._series_input_combo.findData(
                "grounded" if is_noise else "generator_splitter"
            )
        )
        self._series_waveform_combo.setCurrentIndex(
            self._series_waveform_combo.findData("off" if is_noise else "sine")
        )
        if is_noise:
            self._series_frequency.setValue(0.0)
            self._series_amplitude.setValue(0.0)
            self._series_generator_offset.setValue(0.0)
        elif self._series_frequency.value() == 0.0:
            self._series_frequency.setValue(5_000_000.0)
        if is_decimation and hasattr(self, "_decimation_spinbox"):
            self._on_series_decimation_point_change()

    def _on_series_decimation_point_change(self) -> None:
        if not hasattr(self, "_series_decimation_point_combo"):
            return
        point = self._series_decimation_point_combo.currentData()
        if not isinstance(point, dict):
            return
        frequency_hz = float(point["frequency_hz"])
        decimation = int(point["decimation"])
        effective_sample_rate_hz = ADC_SAMPLE_RATE_HZ / decimation
        self._series_frequency.setValue(frequency_hz)
        if hasattr(self, "_decimation_spinbox"):
            self._decimation_spinbox.setValue(decimation)
        if self._series_amplitude.value() == 0.0:
            self._series_amplitude.setValue(0.13)
        self._series_generator_frequency_confirmed.setChecked(False)
        self._series_decimation_instruction.setText(
            f"Set the generator manually to {frequency_hz:g} Hz. "
            f"The GUI applies D = {decimation}; effective Fs = "
            f"{effective_sample_rate_hz:g} Hz; expected 16 samples/period."
        )

    def _series_setup_error(self) -> str | None:
        if not self._series_sense_confirmed.isChecked():
            return (
                "Confirm that the physical SENSE jumpers match the selected "
                "ADC ranges before starting the series."
            )
        waveform = self._series_waveform_combo.currentData()
        if waveform != "off" and self._series_frequency.value() <= 0.0:
            return "Enter a generator frequency greater than 0 Hz."
        if waveform != "off" and self._series_amplitude.value() <= 0.0:
            return "Enter a generator amplitude greater than 0 Vpk."
        if (
            self._series_input_combo.currentData() == "grounded"
            and waveform != "off"
        ):
            return "Disable the generator output for grounded-input measurements."
        if self._series_study_combo.currentData() == "decimation":
            point = self._series_decimation_point_combo.currentData()
            if not isinstance(point, dict):
                return "Select a decimation study point."
            if self._decimation_spinbox.value() != int(point["decimation"]):
                return "The applied decimation does not match the selected study point."
            if not math.isclose(
                self._series_frequency.value(),
                float(point["frequency_hz"]),
                rel_tol=0.0,
                abs_tol=0.001,
            ):
                return "The generator frequency metadata does not match the selected study point."
            if not self._series_generator_frequency_confirmed.isChecked():
                return "Confirm that the generator frequency is set to the selected study point."
        return None

    def _series_metadata(self, base_path: pathlib.Path) -> dict[str, object]:
        waveform = self._series_waveform_combo.currentData()
        metadata = {
            "measurement_type": self._series_study_combo.currentData(),
            "series_id": base_path.stem,
            "input_condition": self._series_input_combo.currentData(),
            "generator_model": "Hantek HDG6202",
            "generator_waveform": waveform,
            "generator_frequency_hz": (
                f"{self._series_frequency.value():.12g}" if waveform != "off" else "0"
            ),
            "generator_amplitude_vpk": (
                f"{self._series_amplitude.value():.12g}" if waveform != "off" else "0"
            ),
            "generator_offset_v": (
                f"{self._series_generator_offset.value():.12g}"
                if waveform != "off"
                else "0"
            ),
            "generator_load": self._series_load_combo.currentData(),
            "sense_configuration_confirmed": "true",
            "sense_ch1_vpp": int(self._ch1_range_combo.currentData()),
            "sense_ch2_vpp": int(self._ch2_range_combo.currentData()),
        }
        if self._series_study_combo.currentData() == "decimation":
            point = self._series_decimation_point_combo.currentData()
            decimation = int(point["decimation"])
            effective_sample_rate_hz = ADC_SAMPLE_RATE_HZ / decimation
            metadata.update(
                {
                    "decimation_study_point": self._series_decimation_point_combo.currentText(),
                    "decimation_study_expected_factor": decimation,
                    "effective_sample_rate_hz": f"{effective_sample_rate_hz:.12g}",
                    "expected_samples_per_period": "16",
                }
            )
        return metadata

    @staticmethod
    def _series_output_path(
        base_path: pathlib.Path, repetition: int, total: int
    ) -> pathlib.Path:
        digits = max(2, len(str(total)))
        return base_path.with_name(
            f"{base_path.stem}_{repetition:0{digits}d}{base_path.suffix or '.csv'}"
        )

    def _toggle_measurement_series(self) -> None:
        if self._batch_active:
            if self._conn_mgr:
                self._conn_mgr.stop_acquisition()
            self._finish_measurement_series("Series stopped by user")
            return

        setup_error = self._series_setup_error()
        if setup_error:
            QMessageBox.warning(self, "Cannot start series", setup_error)
            return
        if not self._is_connected or not self._conn_mgr:
            QMessageBox.warning(self, "Cannot start series", "Device is not connected.")
            return
        if self._run_btn.isChecked():
            QMessageBox.warning(
                self,
                "Cannot start series",
                "Stop continuous acquisition before starting a measurement series.",
            )
            return

        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        study = str(self._series_study_combo.currentData())
        adc_range = int(self._ch1_range_combo.currentData())
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        if study == "decimation":
            point = self._series_decimation_point_combo.currentData()
            default_stem = (
                f"decimation_{adc_range}vpp_{point['filename_token']}_{timestamp}"
            )
        else:
            default_stem = f"{study}_{adc_range}vpp_{timestamp}"
        default_path = CAPTURES_DIR / f"{default_stem}.csv"
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save measurement series",
            str(default_path),
            "CSV file (*.csv)",
        )
        if not selected_path:
            return
        base_path = pathlib.Path(selected_path)
        if base_path.suffix.lower() != ".csv":
            base_path = base_path.with_suffix(".csv")

        total = self._series_count.value()
        existing = [
            self._series_output_path(base_path, index, total)
            for index in range(1, total + 1)
            if self._series_output_path(base_path, index, total).exists()
        ]
        if existing:
            QMessageBox.warning(
                self,
                "Series files already exist",
                f"The series would overwrite {len(existing)} existing file(s). "
                "Choose a different base name.",
            )
            return

        self._batch_active = True
        self._batch_state = "waiting_initial_status"
        self._batch_base_path = base_path
        self._batch_metadata = self._series_metadata(base_path)
        self._batch_total = total
        self._batch_saved = 0
        self._batch_invalid_frames = 0
        self._batch_previous_sequence = None
        self._batch_required_status_generation = self._status_generation + 1
        self._set_series_running(True)
        self._series_progress.setText("Reading applied hardware configuration…")
        self._send("status")
        required_generation = self._batch_required_status_generation
        QTimer.singleShot(
            3000,
            lambda generation=required_generation: self._check_series_status_timeout(
                generation
            ),
        )

    def _set_series_running(self, running: bool) -> None:
        self._series_btn.setText("Stop series" if running else "Capture series")
        for control in self._series_controls:
            control.setEnabled(not running)
        self._normal_controls_group.setEnabled(not running)
        for group in self._advanced_groups:
            group.setEnabled(not running)
        self._series_btn.setEnabled(self._is_connected or running)
        if not running:
            self._set_hardware_controls_enabled(self._is_connected)

    def _check_series_status_timeout(self, required_generation: int) -> None:
        if (
            self._batch_active
            and self._batch_required_status_generation == required_generation
            and self._status_generation < required_generation
            and self._batch_state
            in {"waiting_initial_status", "waiting_frame_status"}
        ):
            self._finish_measurement_series(
                "Series stopped: no hardware status reply was received"
            )

    def _acquire_next_series_frame(self) -> None:
        if not self._batch_active or not self._conn_mgr:
            return
        required_status = {
            "depth",
            "pretrigger",
            "decim",
            "overflow",
            "build",
            "version",
            "afe_ch1_range_vpp",
            "afe_ch2_range_vpp",
        }
        missing = sorted(required_status.difference(self._afe_state))
        if missing:
            self._finish_measurement_series(
                "Series stopped: incomplete hardware status (missing "
                + ", ".join(missing)
                + ")"
            )
            return
        if self._batch_state == "waiting_initial_status":
            for channel in (1, 2):
                metadata_key = f"sense_ch{channel}_vpp"
                reported_key = f"afe_ch{channel}_range_vpp"
                if str(self._batch_metadata[metadata_key]) != self._afe_state[reported_key]:
                    self._finish_measurement_series(
                        "Series stopped: the reported ADC range changed after "
                        "SENSE confirmation. Verify the jumpers and confirm them again."
                    )
                    return
        self._batch_state = "acquiring"
        next_index = self._batch_saved + 1
        self._series_progress.setText(
            f"Acquiring {next_index}/{self._batch_total}…"
        )
        self._status_label.setText(
            f"Series {next_index}/{self._batch_total}: acquiring…"
        )
        self._conn_mgr.start_acquisition("single")

    def _on_series_frame_available(self) -> None:
        if not self._batch_active or self._batch_state != "acquiring":
            return
        self._batch_state = "waiting_frame_status"
        self._batch_required_status_generation = self._status_generation + 1
        self._series_progress.setText(
            f"Validating {self._batch_saved + 1}/{self._batch_total}…"
        )
        self._send("status")
        required_generation = self._batch_required_status_generation
        QTimer.singleShot(
            3000,
            lambda generation=required_generation: self._check_series_status_timeout(
                generation
            ),
        )

    def _current_series_frame_error(self) -> str | None:
        if self._frame_sequence is None:
            return "missing frame sequence number"
        if len(self._ch1_raw) != len(self._ch2_raw) or len(self._ch1_raw) == 0:
            return "channel arrays have different or zero lengths"
        try:
            expected_depth = int(self._afe_state["depth"])
        except (KeyError, ValueError):
            return "invalid capture depth in hardware status"
        if len(self._ch1_raw) != expected_depth:
            return f"received {len(self._ch1_raw)} samples instead of {expected_depth}"
        # The single-buffer FPGA deliberately stops accepting new samples while
        # the completed frame is drained. Its overflow flag therefore records
        # acquisition dead time after the frame, not corruption of that frame.
        # Keep the flag in CSV metadata, but do not reject the captured data.
        if self._batch_previous_sequence is not None:
            expected_sequence = (self._batch_previous_sequence + 1) & 0xFFFFFFFF
            if self._frame_sequence != expected_sequence:
                return (
                    f"frame sequence jumped from {self._batch_previous_sequence} "
                    f"to {self._frame_sequence}"
                )
        for channel, samples, enabled in (
            (1, self._ch1_raw, self.ch1_enabled),
            (2, self._ch2_raw, self.ch2_enabled),
        ):
            if enabled and (np.any(samples == 0) or np.any(samples == ADC_COUNTS - 1)):
                return f"channel {channel} contains clipped samples"
        return None

    def _complete_current_series_frame(self) -> None:
        if not self._batch_active or self._batch_state != "waiting_frame_status":
            return
        frame_error = self._current_series_frame_error()
        self._batch_previous_sequence = self._frame_sequence
        if frame_error:
            self._batch_invalid_frames += 1
            self._cmd_panel.log_error(f"Rejected series frame: {frame_error}")
            if self._batch_invalid_frames >= MAX_INVALID_SERIES_FRAMES:
                self._finish_measurement_series(
                    f"Series stopped after {self._batch_invalid_frames} invalid frames: "
                    f"{frame_error}"
                )
                return
            QTimer.singleShot(0, self._acquire_next_series_frame)
            return

        repetition = self._batch_saved + 1
        path = self._series_output_path(
            self._batch_base_path, repetition, self._batch_total
        )
        metadata = {
            **self._batch_metadata,
            "series_total": self._batch_total,
            "repetition_index": repetition,
            "invalid_frames_before_capture": self._batch_invalid_frames,
        }
        try:
            self._write_current_frame(path, metadata)
        except (OSError, ValueError) as exc:
            self._finish_measurement_series(f"Series stopped: {exc}")
            return

        self._batch_saved = repetition
        self._series_progress.setText(
            f"Saved {self._batch_saved}/{self._batch_total}: {path.name}"
        )
        if self._batch_saved >= self._batch_total:
            self._finish_measurement_series(
                f"Series complete: saved {self._batch_saved} frames"
            )
        else:
            QTimer.singleShot(0, self._acquire_next_series_frame)

    def _finish_measurement_series(self, message: str) -> None:
        was_active = self._batch_active
        self._batch_active = False
        self._batch_state = "idle"
        self._set_series_running(False)
        self._series_progress.setText(message)
        self._status_label.setText("Connected" if self._is_connected else "Disconnected")
        if was_active:
            if message.startswith("Series complete"):
                self._cmd_panel.log_ok(message)
            else:
                self._cmd_panel.log_error(message)

    def _single_acquire(self):
        if self._conn_mgr:
            self._single_btn.setEnabled(False)
            self._normal_single_btn.setEnabled(False)
            self._status_label.setText("Acquiring…")
            self._conn_mgr.start_acquisition("single")

    def _on_acquisition_done(self):
        if self._batch_active:
            return
        self._single_btn.setEnabled(True)
        self._normal_single_btn.setEnabled(True)
        self._status_label.setText("Connected")

    def _save_frame(self):
        if not self._have_frame:
            self._cmd_panel.log_error("No frame to save")
            return

        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        default_path = CAPTURES_DIR / f"capture_{ts}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save frame", str(default_path), "CSV file (*.csv)"
        )
        if not path:
            return

        self._write_current_frame(path, {"measurement_type": "manual"})

    def _write_current_frame(
        self, path: str | pathlib.Path, extra_metadata: dict[str, object] | None = None
    ) -> None:
        path = pathlib.Path(path)
        calibrated_display = self._can_display_calibrated_volts()
        try:
            applied_decimation = int(self._afe_state.get("decim", self.decimation))
        except (TypeError, ValueError):
            applied_decimation = int(self.decimation)
        if applied_decimation <= 0:
            applied_decimation = int(self.decimation)
        git_commit, git_dirty = software_revision()
        metadata = {
            "capture_name": os.path.basename(path),
            "decim_factor": applied_decimation,
            "requested_decim_factor": int(self.decimation),
            "capture_depth": len(self._ch1_raw),
            "pretrigger_samples": int(self.pretrigger_size),
            "actual_seconds_per_div": f"{self._time_per_div(applied_decimation):.12g}",
            "requested_seconds_per_div": (
                "custom"
                if self._timebase_target is None
                else f"{self._timebase_target:.12g}"
            ),
            "ui_mode": self._ui_mode,
            "voltage_display_mode": (
                "calibrated_volts" if calibrated_display else "raw_adc_codes"
            ),
            "trigger_mode": self._trigger_mode_combo.currentText().lower(),
            "trigger_source": 2
            if self._trigger_source_combo.currentText() == "CH2"
            else 1,
            "trigger_level_pct": f"{self._trigger_value.value():.4f}",
            "adc_format": self._adc_format,
            "calibration_schema_version": self._calibration["schema_version"],
            "frame_sequence": (
                "" if self._frame_sequence is None else self._frame_sequence
            ),
            "firmware_status_timestamp": self._last_status_timestamp,
            "software_git_commit": git_commit,
            "software_git_dirty": git_dirty,
            **{
                f"firmware_{key}": value
                for key, value in self._afe_state.items()
                if key not in {"build", "version"}
            },
        }
        if "build" in self._afe_state:
            metadata["fpga_build"] = self._afe_state["build"]
        if "version" in self._afe_state:
            metadata["fpga_version"] = self._afe_state["version"]

        for channel in (1, 2):
            attenuation = self._channel_attenuation(channel)
            coupling_combo = (
                self._coupling_combo if channel == 1 else self._ch2_coupling_combo
            )
            normal_vdiv = self._normal_volts_per_div.get(f"ch{channel}")
            calibration = self._calibration_for_channel(channel)
            metadata[f"channel_enabled_ch{channel}"] = str(
                self.ch1_enabled if channel == 1 else self.ch2_enabled
            ).lower()
            metadata[f"attenuation_ch{channel}"] = attenuation
            metadata[f"coupling_ch{channel}"] = coupling_combo.currentText().lower()
            metadata[f"gain_pct_ch{channel}"] = f"{self._channel_gain_percent(channel):.4f}"
            metadata[f"afe_offset_pct_ch{channel}"] = (
                f"{self._channel_afe_offset_percent(channel):.4f}"
            )
            metadata[f"calibration_profile_ch{channel}"] = profile_key(
                channel,
                attenuation,
                self._range_for_channel(f"ch{channel}"),
            )
            metadata[f"normal_requested_volts_per_div_ch{channel}"] = (
                "" if normal_vdiv is None else f"{float(normal_vdiv):.12g}"
            )
            if calibration is not None:
                metadata[f"calibration_volts_per_code_ch{channel}"] = (
                    f"{calibration[0]:.12g}"
                )
                metadata[f"calibrated_volts_per_div_ch{channel}"] = (
                    f"{self._calibrated_volts_per_div(channel):.12g}"
                )
            else:
                metadata[f"calibrated_volts_per_div_ch{channel}"] = ""

            if self._ui_mode == "normal":
                metadata[f"displayed_volts_per_div_ch{channel}"] = (
                    "" if normal_vdiv is None else f"{float(normal_vdiv):.12g}"
                )
            elif calibrated_display and calibration is not None:
                metadata[f"displayed_volts_per_div_ch{channel}"] = (
                    f"{self._calibrated_volts_per_div(channel):.12g}"
                )
            else:
                metadata[f"displayed_volts_per_div_ch{channel}"] = "raw_adc_codes"

        metadata.update(extra_metadata or {})

        voltage_columns = {}
        if calibrated_display:
            voltage_columns = {
                "ch1_volts": self._input_volts(self._ch1_raw, 1),
                "ch2_volts": self._input_volts(self._ch2_raw, 2),
            }

        save_capture_csv(
            path,
            self._ch1_raw,
            self._ch2_raw,
            fs_hz=ADC_SAMPLE_RATE_HZ / applied_decimation,
            n_bits=14,
            metadata=metadata,
            **voltage_columns,
        )
        self._cmd_panel.log_ok(f"Saved: {os.path.basename(path)}")

    def _raw_to_display_codes(self, raw: np.ndarray) -> np.ndarray:
        """Return signed ADC codes without applying gain, offset, or filtering."""
        half = ADC_COUNTS // 2
        if self._adc_format == "Offset Binary":
            return raw.astype(np.int32) - half
        return np.where(
            raw >= half, raw.astype(np.int32) - ADC_COUNTS, raw.astype(np.int32)
        )

    @staticmethod
    def _display_indices(values: np.ndarray) -> np.ndarray:
        """Limit plot points while retaining spectral peaks and troughs."""
        values = np.asarray(values)
        sample_count = len(values)
        if sample_count <= MAX_DISPLAY_POINTS:
            return np.arange(sample_count, dtype=np.int32)

        bucket_count = MAX_DISPLAY_POINTS // 2
        edges = np.linspace(0, sample_count, bucket_count + 1, dtype=np.int32)
        selected: list[int] = []
        for start, stop in zip(edges[:-1], edges[1:]):
            bucket = values[start:stop]
            extrema = {start + int(np.argmin(bucket)), start + int(np.argmax(bucket))}
            selected.extend(sorted(extrema))
        return np.asarray(selected, dtype=np.int32)

    def _update_plot(self):
        latest = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            if len(latest) == 3:
                sequence, ch1_raw, ch2_raw = latest
                self._frame_sequence = int(sequence)
            else:
                # Compatibility with locally supplied queues created by older tools.
                ch1_raw, ch2_raw = latest
                self._frame_sequence = None
            self._ch1_raw = np.asarray(ch1_raw, dtype=np.uint16).copy()
            self._ch2_raw = np.asarray(ch2_raw, dtype=np.uint16).copy()
            self._have_frame = True
            self._render_current_frame()
            self._on_series_frame_available()

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
                self._ch1_raw, 1, start, stop, pixel_width
            )
            _, self._ch2_data = self._min_max_envelope(
                self._ch2_raw, 2, start, stop, pixel_width
            )
        else:
            self._plot_x = np.arange(start, stop, dtype=np.int32)
            self._ch1_data = self._time_display_values(self._ch1_raw[start:stop], 1)
            self._ch2_data = self._time_display_values(self._ch2_raw[start:stop], 2)
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
        self, raw: np.ndarray, channel: int, start: int, stop: int, pixel_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Summarize each horizontal pixel column with its sample extrema."""
        values = self._time_display_values(raw[start:stop], channel)
        bucket_count = min(pixel_width, len(values))
        edges = np.linspace(0, len(values), bucket_count + 1, dtype=np.int32)
        bucket_starts = edges[:-1]
        bucket_ends = edges[1:]
        lows = np.minimum.reduceat(values, bucket_starts)
        highs = np.maximum.reduceat(values, bucket_starts)
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
        frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)[1:]
        window = {
            "hann": np.hanning(sample_count + 1)[:-1],
            "blackman": np.blackman(sample_count),
        }[self._fft_window]
        coherent_gain = window.mean()

        def spectrum_dbfs(codes: np.ndarray) -> np.ndarray:
            codes = np.asarray(codes, dtype=np.float64)
            codes -= np.mean(codes)
            spectrum = np.fft.rfft(codes * window)
            magnitude = np.abs(spectrum) / (sample_count * coherent_gain) * 2.0
            magnitude[-1] /= 2.0
            return 20.0 * np.log10(
                np.maximum(magnitude[1:] / (ADC_COUNTS / 2), 1e-12)
            )

        if self.ch1_enabled:
            levels = spectrum_dbfs(self._raw_to_display_codes(self._ch1_raw))
            indices = self._display_indices(levels)
            self._curve_ch1.setData(
                frequencies[indices],
                levels[indices],
                pen="y",
                symbol=None,
            )
        if self.ch2_enabled:
            levels = spectrum_dbfs(self._raw_to_display_codes(self._ch2_raw))
            indices = self._display_indices(levels)
            self._curve_ch2.setData(
                frequencies[indices],
                levels[indices],
                pen="c",
                symbol=None,
            )
