from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDial, QLabel, QRadioButton, QGroupBox
)
from PyQt5.QtCore import Qt
import sys


class OscilloscopeControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # Gain Dial
        self.gain_dial = QDial()
        self.gain_dial.setRange(1, 10)
        self.gain_dial.setNotchesVisible(True)
        layout.addWidget(QLabel("Gain"))
        layout.addWidget(self.gain_dial)

        # Offset Dial
        self.offset_dial = QDial()
        self.offset_dial.setRange(-100, 100)
        self.offset_dial.setNotchesVisible(True)
        layout.addWidget(QLabel("Offset"))
        layout.addWidget(self.offset_dial)

        # Trigger Level Dial
        self.trigger_dial = QDial()
        self.trigger_dial.setRange(-100, 100)
        self.trigger_dial.setNotchesVisible(True)
        layout.addWidget(QLabel("Trigger Level"))
        layout.addWidget(self.trigger_dial)

        # AC/DC Coupling
        coupling_group = QGroupBox("Coupling")
        coupling_layout = QVBoxLayout()
        self.ac_button = QRadioButton("AC")
        self.dc_button = QRadioButton("DC")
        self.dc_button.setChecked(True)
        coupling_layout.addWidget(self.ac_button)
        coupling_layout.addWidget(self.dc_button)
        coupling_group.setLayout(coupling_layout)
        layout.addWidget(coupling_group)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Oscilloscope GUI")
        self.setFixedSize(1000, 600)  # Fixed window size
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)

        main_layout = QHBoxLayout()

        # Left: Waveform Display (75% width)
        waveform_display = QLabel("Waveform Display Area")
        waveform_display.setMinimumSize(750, 600)
        waveform_display.setStyleSheet(
            "background-color: black; color: white;")
        waveform_display.setAlignment(Qt.AlignCenter)

        # Right: Control Panel (25% width)
        control_panel = OscilloscopeControlPanel()
        control_panel.setFixedWidth(250)

        main_layout.addWidget(waveform_display)
        main_layout.addWidget(control_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
