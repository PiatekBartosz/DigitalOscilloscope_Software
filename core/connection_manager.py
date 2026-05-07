import asyncio
import logging

from PyQt6.QtCore import QObject, pyqtSignal as Signal

from core.command_client import CommandClient

logger = logging.getLogger(__name__)


class ConnectionManager(QObject):
    PORT        = 8888
    RETRY_DELAY = 5

    connected         = Signal()
    disconnected      = Signal()
    connecting        = Signal()
    device_found      = Signal(str)
    response_received = Signal(str)   # firmware text reply (OK / ERR / …)

    def __init__(self, sample_cb=None):
        super().__init__()
        self._client    = None
        self._ip        = None
        self._port      = self.PORT
        self._running   = False
        self._sample_cb = sample_cb
        self._loop      = None

    def start(self, loop: asyncio.AbstractEventLoop, ip: str,
              port: int | None = None):
        self._loop = loop
        self._ip   = ip
        if port:
            self._port = port
        asyncio.run_coroutine_threadsafe(self.connect_loop(), loop)

    async def connect_loop(self):
        self._running = True
        while self._running:
            self.connecting.emit()
            try:
                self._client = CommandClient(
                    self._ip, self._port,
                    sample_cb=self._sample_cb,
                    text_cb=self.response_received.emit)
                await self._client.connect()
                self.connected.emit()
                await self._wait_for_disconnect()
            except Exception as e:
                logger.error("Connection error: %s", e)

            self.disconnected.emit()
            if self._running:
                logger.info("Retrying in %ds…", self.RETRY_DELAY)
                await asyncio.sleep(self.RETRY_DELAY)

    async def _wait_for_disconnect(self):
        while self._running and self._client and self._client.connected:
            await asyncio.sleep(0.5)

    def stop(self):
        self._running = False
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(
                self._client.disconnect(), self._loop)

    def send_command(self, cmd: str):
        if self._loop and self._client and self._client.connected:
            asyncio.run_coroutine_threadsafe(
                self._client.send_command(cmd), self._loop)
