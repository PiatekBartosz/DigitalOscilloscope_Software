import asyncio
import logging

from PyQt6.QtCore import QObject, pyqtSignal as Signal
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

from core.command_client import CommandClient

logger = logging.getLogger(__name__)

MDNS_SERVICE_TYPE = "_oscilloscope._tcp.local."


class _MDNSListener(ServiceListener):
    def __init__(self, found_cb):
        self._found_cb = found_cb

    def add_service(self, zc: Zeroconf, type_: str, name: str):
        info = zc.get_service_info(type_, name)
        if info and info.parsed_addresses():
            self._found_cb(info.parsed_addresses()[0], info.port)

    def update_service(self, zc, type_, name): pass
    def remove_service(self, zc, type_, name): pass


class ConnectionManager(QObject):
    PORT         = 8888
    RETRY_DELAY  = 5

    connected    = Signal()
    disconnected = Signal()
    connecting   = Signal()
    device_found = Signal(str)

    def __init__(self, sample_cb=None):
        """
        sample_cb : callable(ch1: int, ch2: int) invoked on each ADC frame.
                    Called from the asyncio thread — use a thread-safe queue.
        """
        super().__init__()
        self._client     = None
        self._ip         = None
        self._port       = self.PORT
        self._running    = False
        self._sample_cb  = sample_cb
        self._loop       = None

    async def mdns_discover(self) -> tuple[str, int] | None:
        """Browse for _oscilloscope._tcp.local. and return (host, port)."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        def found(host, port):
            if not future.done():
                future.set_result((host, port))

        zc = Zeroconf()
        browser = ServiceBrowser(zc, MDNS_SERVICE_TYPE, _MDNSListener(found))
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            return None
        finally:
            browser.cancel()
            zc.close()

    def start(self, loop: asyncio.AbstractEventLoop, ip: str | None = None,
              port: int | None = None):
        """Schedule the connection loop in the given asyncio event loop."""
        self._loop = loop
        if ip:
            self._ip   = ip
        if port:
            self._port = port
        asyncio.run_coroutine_threadsafe(self.connect_loop(), loop)

    async def connect_loop(self):
        self._running = True
        while self._running:
            self.connecting.emit()

            if not self._ip:
                logger.info("Searching for device via mDNS…")
                result = await self.mdns_discover()
                if result:
                    self._ip, self._port = result
                    logger.info("Found device at %s:%d", self._ip, self._port)
                    self.device_found.emit(f"{self._ip}:{self._port}")
                else:
                    logger.warning("mDNS discovery timed out, retrying…")
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue

            try:
                self._client = CommandClient(
                    self._ip, self._port, sample_cb=self._sample_cb)
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
