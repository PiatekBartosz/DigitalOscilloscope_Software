from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QDial, QLineEdit
from PyQt6.QtCore import Qt, QTimer

DIAL_DEBOUNCE_MS = 50


def create_dial_widget(label_text, min_val, max_val, init_val, parent_layout, callback):
    layout = QHBoxLayout()
    layout.setSpacing(6)
    layout.setContentsMargins(0, 0, 0, 0)

    label = QLabel(label_text)
    if label_text:
        label.setFixedWidth(115)
        label.setWordWrap(True)
    else:
        label.setFixedWidth(0)

    dial = QDial()
    dial.setRange(min_val, max_val)
    dial.setValue(init_val)
    dial.setFixedSize(48, 48)
    dial.setNotchesVisible(True)

    edit = QLineEdit(str(init_val))
    edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
    edit.setFixedWidth(55)

    debounce_timer = QTimer(dial)
    debounce_timer.setSingleShot(True)
    debounce_timer.setInterval(DIAL_DEBOUNCE_MS)
    debounce_timer.timeout.connect(callback)

    dial.valueChanged.connect(lambda val: edit.setText(str(val)))
    dial.valueChanged.connect(lambda _val: debounce_timer.start())

    def on_edit_finished():
        try:
            val = int(edit.text())
            if min_val <= val <= max_val:
                dial.setValue(val)
        except ValueError:
            pass

    edit.editingFinished.connect(on_edit_finished)

    layout.addWidget(label)
    layout.addWidget(dial)
    layout.addWidget(edit)
    parent_layout.addLayout(layout)
    return dial, edit


def create_float_dial_widget(
    label_text, min_val, max_val, init_val, parent_layout, callback, decimals=2
):
    """Create a coarse dial paired with a precise floating-point value field."""
    scale = 10**decimals
    layout = QHBoxLayout()
    layout.setSpacing(6)
    layout.setContentsMargins(0, 0, 0, 0)

    label = QLabel(label_text)
    if label_text:
        label.setFixedWidth(115)
        label.setWordWrap(True)
    else:
        label.setFixedWidth(0)

    dial = QDial()
    dial.setRange(round(min_val * scale), round(max_val * scale))
    dial.setValue(round(init_val * scale))
    dial.setFixedSize(48, 48)
    dial.setNotchesVisible(True)

    value = QDoubleSpinBox()
    value.setRange(min_val, max_val)
    value.setDecimals(decimals)
    value.setSingleStep(1.0 / scale)
    value.setValue(init_val)
    value.setAlignment(Qt.AlignmentFlag.AlignCenter)
    value.setFixedWidth(65)

    debounce_timer = QTimer(dial)
    debounce_timer.setSingleShot(True)
    debounce_timer.setInterval(DIAL_DEBOUNCE_MS)
    debounce_timer.timeout.connect(callback)

    def on_dial_change(raw_value):
        value.blockSignals(True)
        value.setValue(raw_value / scale)
        value.blockSignals(False)
        debounce_timer.start()

    def on_value_change(percent):
        raw_value = round(percent * scale)
        if raw_value != dial.value():
            dial.setValue(raw_value)
        else:
            debounce_timer.start()

    dial.valueChanged.connect(on_dial_change)
    value.valueChanged.connect(on_value_change)

    layout.addWidget(label)
    layout.addWidget(dial)
    layout.addWidget(value)
    parent_layout.addLayout(layout)
    return dial, value
