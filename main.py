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
    parser.add_argument("--ip", default="192.168.0.109",
                        help="Device IP address")
    parser.add_argument("--port", type=int, default=8888)
    return parser.parse_args()


def main():
    options = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if options.debug else logging.INFO)
    logger.info("Starting oscilloscope application")

    # Queue holds complete frames (ch1, ch2 numpy arrays); depth 4 keeps only
    # recent captures so the display always shows the latest acquisition.
    frame_queue: queue.Queue = queue.Queue(maxsize=4)

    def on_frame(seq: int, ch1, ch2):
        try:
            frame_queue.put_nowait((ch1, ch2))
        except queue.Full:
            # Drop oldest frame and replace with latest
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait((ch1, ch2))
            except queue.Full:
                pass

    async_loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(async_loop)
        async_loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()

    app = QApplication(sys.argv)

    conn_mgr = ConnectionManager(frame_cb=on_frame)
    osc = Oscilloscope(conn_mgr, frame_queue)
    osc.show()

    conn_mgr.start(async_loop, ip=options.ip, port=options.port)

    ret = app.exec()

    conn_mgr.stop()
    async_loop.call_soon_threadsafe(async_loop.stop)
    loop_thread.join(timeout=2)

    sys.exit(ret)


if __name__ == '__main__':
    main()
