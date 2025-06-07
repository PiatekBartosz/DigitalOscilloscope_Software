import sys
import argparse
import logging
from PyQt6.QtWidgets import QApplication
from ui.oscilloscope import Oscilloscope

logger = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    return parser.parse_args()


def main():
    options = parse_arguments()
    logging.basicConfig(level=logging.DEBUG) if options.debug else logging.basicConfig(
        level=logging.INFO)

    logging.info("Starting oscilosope application")

    app = QApplication(sys.argv)
    osc = Oscilloscope()
    osc.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
