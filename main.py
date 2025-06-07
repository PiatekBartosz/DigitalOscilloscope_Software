import sys
from PyQt6.QtWidgets import QApplication
from ui.oscilloscope import Oscilloscope


def main():
    app = QApplication(sys.argv)
    osc = Oscilloscope()
    osc.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
