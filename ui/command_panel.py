from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLineEdit, QPushButton, QTextEdit, QLabel,
)
from PyQt6.QtCore import pyqtSignal as Signal, Qt
from PyQt6.QtGui import QTextCursor, QColor

# (display label, command template)
# Templates ending with a space expect the user to append a value.
_PRESETS = [
    ("── Presets ──", ""),
    ("CH1 gain …", "afe gain 1 "),
    ("CH2 gain …", "afe gain 2 "),
    ("CH1 offset …", "afe offset 1 "),
    ("CH2 offset …", "afe offset 2 "),
    ("CH1 atten 1:1", "afe atten 1 1"),
    ("CH1 atten 1:100", "afe atten 1 100"),
    ("CH2 atten 1:1", "afe atten 2 1"),
    ("CH2 atten 1:100", "afe atten 2 100"),
    ("CH1 coupling AC", "afe coupling 1 ac"),
    ("CH1 coupling DC", "afe coupling 1 dc"),
    ("CH2 coupling AC", "afe coupling 2 ac"),
    ("CH2 coupling DC", "afe coupling 2 dc"),
    ("Trigger AC", "afe trigger ac"),
    ("Trigger DC", "afe trigger dc"),
    ("Interleaved ON", "afe interleaved 1"),
    ("Interleaved OFF", "afe interleaved 0"),
]


class CommandPanel(QWidget):
    """Bottom command console: preset picker + free-text input + history log."""

    command_submitted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._hist_idx = -1
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── header label ──────────────────────────────────────────────────────
        header = QLabel("Command Console")
        header.setStyleSheet("font-weight: bold; color: #00ffff;")
        outer.addWidget(header)

        # ── input row ─────────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(6)

        self._preset_combo = QComboBox()
        self._preset_combo.setFixedWidth(180)
        for label, _ in _PRESETS:
            self._preset_combo.addItem(label)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        row.addWidget(self._preset_combo)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command and press Enter…")
        self._input.returnPressed.connect(self._send)
        self._input.installEventFilter(self)
        row.addWidget(self._input, stretch=1)

        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(70)
        send_btn.clicked.connect(self._send)
        row.addWidget(send_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._log.clear if hasattr(self, "_log") else lambda: None)
        row.addWidget(clear_btn)

        outer.addLayout(row)

        # ── log area ──────────────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(130)
        self._log.setStyleSheet(
            "QTextEdit { background-color: #111; color: #ccc;"
            " border: 1px solid #444; font-family: Consolas, monospace;"
            " font-size: 10pt; }"
        )
        outer.addWidget(self._log)

        # wire clear button now that _log exists
        clear_btn.clicked.disconnect()
        clear_btn.clicked.connect(self._log.clear)

    # ── preset selection ──────────────────────────────────────────────────────

    def _on_preset_selected(self, idx: int):
        template = _PRESETS[idx][1]
        if not template:
            return
        self._input.setText(template)
        self._input.setFocus()
        self._input.setCursorPosition(len(template))
        # reset combo so the same item can be selected again
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(0)
        self._preset_combo.blockSignals(False)

    # ── send ──────────────────────────────────────────────────────────────────

    def _send(self):
        cmd = self._input.text().strip()
        if not cmd:
            return
        self._history.append(cmd)
        self._hist_idx = len(self._history)
        self._input.clear()
        self._append_log(f"> {cmd}", "#00ffff")
        self.command_submitted.emit(cmd)

    # ── keyboard history navigation (↑ / ↓) ──────────────────────────────────

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Up and self._history:
                self._hist_idx = max(0, self._hist_idx - 1)
                self._input.setText(self._history[self._hist_idx])
                return True
            if key == Qt.Key.Key_Down and self._history:
                self._hist_idx = min(len(self._history), self._hist_idx + 1)
                self._input.setText(
                    self._history[self._hist_idx]
                    if self._hist_idx < len(self._history) else ""
                )
                return True
        return super().eventFilter(obj, event)

    # ── public API ────────────────────────────────────────────────────────────

    def log_info(self, msg: str):
        self._append_log(msg, "#aaaaaa")

    def log_ok(self, msg: str):
        self._append_log(msg, "#44ff88")

    def log_error(self, msg: str):
        self._append_log(msg, "#ff4444")

    # ── internal ─────────────────────────────────────────────────────────────

    def _append_log(self, text: str, color: str = "#cccccc"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.setTextColor(QColor("#666666"))
        self._log.insertPlainText(f"[{ts}] ")
        self._log.setTextColor(QColor(color))
        self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.MoveOperation.End)
