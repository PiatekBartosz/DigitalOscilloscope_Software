import asyncio
import logging
import socket

from PyQt6.QtCore import QObject, pyqtSignal as Signal

from core.command_client import CommandClient

logger = logging.getLogger(__name__)

_ACQUIRE_CMD = "acquire"
_MDNS_TYPE = "_oscilloscope._tcp.local."
_DISCOVERY_TIMEOUT = 30.0


class ConnectionManager(QObject):
    PORT = 8888
    RETRY_DELAY = 5

    connected = Signal()
    disconnected = Signal()
    connecting = Signal()
    device_found = Signal(str)
    response_received = Signal(str)
    acquisition_done = Signal()

    def __init__(self, frame_cb=None):
        super().__init__()
        self._client = None
        self._ip = None
        self._port = self.PORT
        self._running = False
        self._frame_cb = frame_cb
        self._loop = None
        self._acq_mode = None

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        ip: "str | None" = None,
        port: "int | None" = None,
    ):
        self._loop = loop
        self._ip = ip
        if port:
            self._port = port
        asyncio.run_coroutine_threadsafe(self._connect_loop(), loop)

    async def _discover_mdns(self) -> "str | None":
        """Discover the oscilloscope via mDNS. Returns IP string or None."""
        try:
            from zeroconf import ServiceStateChange
            from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf
        except ImportError:
            logger.warning("zeroconf not installed — skipping mDNS discovery")
            return None

        found_ip = None
        loop = asyncio.get_running_loop()
        found_fut: asyncio.Future = loop.create_future()

        def on_change(zeroconf, service_type, name, state_change):
            if state_change is ServiceStateChange.Added:
                loop.create_task(_fetch(zeroconf, service_type, name))

        async def _fetch(zeroconf, service_type, name):
            nonlocal found_ip
            info = await zeroconf.async_get_service_info(service_type, name)
            if info and info.addresses and not found_fut.done():
                found_ip = socket.inet_ntoa(info.addresses[0])
                self.device_found.emit(found_ip)
                found_fut.set_result(found_ip)

        logger.info("mDNS: searching for %s …", _MDNS_TYPE)
        azc = AsyncZeroconf()
        browser = AsyncServiceBrowser(azc.zeroconf, _MDNS_TYPE, handlers=[on_change])
        try:
            await asyncio.wait_for(
                asyncio.shield(found_fut), timeout=_DISCOVERY_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning("mDNS discovery timed out after %.0f s", _DISCOVERY_TIMEOUT)
        finally:
            await browser.async_cancel()
            await azc.async_close()

        return found_ip

    async def _connect_loop(self):
        self._running = True
        while self._running:
            self.connecting.emit()

            ip = self._ip
            if ip is None:
                ip = await self._discover_mdns()
                if ip is None:
                    logger.info("No device found; retrying in %ds…", self.RETRY_DELAY)
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue

            try:
                self._client = CommandClient(
                    ip,
                    self._port,
                    frame_cb=self._on_frame,
                    text_cb=self.response_received.emit,
                )
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
        self._running = False
        self._acq_mode = None
        if self._loop and self._client:
            asyncio.run_coroutine_threadsafe(self._client.disconnect(), self._loop)

    def send_command(self, cmd: str):
        if self._loop and self._client and self._client.connected:
            asyncio.run_coroutine_threadsafe(self._client.send_command(cmd), self._loop)

    def start_acquisition(self, mode: str):
        """Send the first acquire command and enter single or continuous mode.

        mode: 'single' | 'continuous'
        No-op when not connected.
        """
        if not (self._loop and self._client and self._client.connected):
            return
        self._acq_mode = mode
        asyncio.run_coroutine_threadsafe(
            self._client.send_command(_ACQUIRE_CMD), self._loop
        )

    def stop_acquisition(self):
        """Prevent further acquire commands from being sent after the current frame."""
        self._acq_mode = None

    def _on_frame(self, seq: int, ch1, ch2):
        if self._frame_cb:
            self._frame_cb(seq, ch1, ch2)

        mode = self._acq_mode
        if mode == "single":
            self._acq_mode = None
            self.acquisition_done.emit()
        elif mode == "continuous" and self._client and self._client.connected:
            asyncio.create_task(self._client.send_command(_ACQUIRE_CMD))
