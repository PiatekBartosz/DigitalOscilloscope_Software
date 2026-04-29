import sys
import argparse
import asyncio
import logging
import threading
import queue

from PyQt6.QtWidgets import QApplication
from ui.oscilloscope import Oscilloscope
from core.connection_manager import ConnectionManager

logger = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--ip", default=None,
                        help="Device IP address (skips mDNS discovery)")
    parser.add_argument("--port", type=int, default=8888)
    return parser.parse_args()


def main():
    options = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if options.debug else logging.INFO)
    logger.info("Starting oscilloscope application")

    sample_queue: queue.Queue = queue.Queue(maxsize=4096)

    def on_sample(ch1: int, ch2: int):
        try:
            sample_queue.put_nowait((ch1, ch2))
        except queue.Full:
            pass

    async_loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(async_loop)
        async_loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    conn_mgr = ConnectionManager(sample_cb=on_sample)
    conn_mgr.start(async_loop, ip=options.ip, port=options.port)

    app = QApplication(sys.argv)
    osc = Oscilloscope(conn_mgr, sample_queue)
    osc.show()

    ret = app.exec()

    conn_mgr.stop()
    async_loop.call_soon_threadsafe(async_loop.stop)
    loop_thread.join(timeout=2)

    sys.exit(ret)


if __name__ == '__main__':
    main()
