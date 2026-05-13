import asyncio
import logging

from PyQt6.QtCore import QObject, pyqtSignal as Signal

from core.command_client import CommandClient

logger = logging.getLogger(__name__)

_ACQUIRE_CMD = "acquire"


class ConnectionManager(QObject):
    PORT        = 8888
    RETRY_DELAY = 5

    connected         = Signal()
    disconnected      = Signal()
    connecting        = Signal()
    device_found      = Signal(str)
    response_received = Signal(str)   # firmware text reply
    acquisition_done  = Signal()      # emitted when a single-shot acquisition completes

    def __init__(self, frame_cb=None):
        super().__init__()
        self._client   = None
        self._ip       = None
        self._port     = self.PORT
        self._running  = False
        self._frame_cb = frame_cb
        self._loop     = None
        self._acq_mode = None   # None | "single" | "continuous"

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop, ip: str,
              port: int | None = None):
        self._loop = loop
        self._ip   = ip
        if port:
            self._port = port
        asyncio.run_coroutine_threadsafe(self._connect_loop(), loop)

    async def _connect_loop(self):
        self._running = True
        while self._running:
            self.connecting.emit()
            try:
                self._client = CommandClient(
                    self._ip, self._port,
                    frame_cb=self._on_frame,
                    text_cb=self.response_received.emit)
                await self._client.connect()
                self.connected.emit()
                await self._wait_for_disconnect()
            except Exception as e:
                logger.error("Connection error: %s", e)

            self._acq_mode = None
            self.disconnected.emit()
            if self._running:
                logger.info("Retrying in %ds…", self.RETRY_DELAY)
                await asyncio.sleep(self.RETRY_DELAY)

    async def _wait_for_disconnect(self):
        while self._running and self._client and self._client.connected:
            await asyncio.sleep(0.5)

    def stop(self):
        self._running  = False
        self._acq_mode = None
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(
                self._client.disconnect(), self._loop)

    # ── commands ───────────────────────────────────────────────────────────────

    def send_command(self, cmd: str):
        if self._loop and self._client and self._client.connected:
            asyncio.run_coroutine_threadsafe(
                self._client.send_command(cmd), self._loop)

    def start_acquisition(self, mode: str):
        """Send the first acquire command and enter single or continuous mode.

        mode: 'single' | 'continuous'
        No-op when not connected.
        """
        if not (self._loop and self._client and self._client.connected):
            return
        self._acq_mode = mode
        asyncio.run_coroutine_threadsafe(
            self._client.send_command(_ACQUIRE_CMD), self._loop)

    def stop_acquisition(self):
        """Prevent further acquire commands from being sent after the current frame."""
        self._acq_mode = None

    # ── internal frame routing ─────────────────────────────────────────────────

    def _on_frame(self, seq: int, ch1, ch2):
        # Forward to the UI frame queue.
        if self._frame_cb:
            self._frame_cb(seq, ch1, ch2)

        mode = self._acq_mode
        if mode == "single":
            self._acq_mode = None
            self.acquisition_done.emit()
        elif mode == "continuous" and self._client and self._client.connected:
            # Called from the asyncio receive-loop task — create_task is safe here.
            asyncio.create_task(self._client.send_command(_ACQUIRE_CMD))
