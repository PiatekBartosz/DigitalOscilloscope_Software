from PyQt6.QtWidgets import QVBoxLayout, QLabel, QDial, QLineEdit
from PyQt6.QtCore import Qt

def create_dial_widget(label_text, min_val, max_val, init_val, parent_layout, callback):
    layout = QVBoxLayout()
    label = QLabel(label_text)
    dial = QDial()
    dial.setRange(min_val, max_val)
    dial.setValue(init_val)
    edit = QLineEdit(str(init_val))
    edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
    edit.setMaximumWidth(60)

    # Dial → LineEdit
    dial.valueChanged.connect(lambda val: edit.setText(str(val)))
    dial.valueChanged.connect(callback)

    # LineEdit → Dial
    def on_edit_finished():
        try:
            val = int(edit.text())
            if min_val <= val <= max_val:
                dial.setValue(val)
        except ValueError:
            pass  # ignore bad input

    edit.editingFinished.connect(on_edit_finished)

    layout.addWidget(label)
    layout.addWidget(dial)
    layout.addWidget(edit)
    parent_layout.addLayout(layout)
    return dial, edit

