import asyncio
import logging
from command_client import CommandClient
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("Connection manager")

class ConnectionManager(QObject):
    PORT = 8888
    RETRY_DELAY = 5
    connected = Signal()
    disconnected = Signal()
    connecting = Signal()
    device_found = Signal(str)

    def __init__(self):
        super().__init__()
        self._client = None
        self._ip = None
        self._port = self.PORT
        self._running = False

    async def mdns_discover(self):
        #TODO
        pass

    async def connect_loop(self):
            self._running = True
            while self._running:
                self.connecting.emit()
                try:
                    self._client = CommandClient(self._ip, self._port)
                    await self._client.connect()
                    self.connected.emit()
                    # Wait here until disconnected
                    await self._wait_for_disconnect()
                except Exception as e:
                    print(f"Connection error: {e}")
                self.disconnected.emit()
                await asyncio.sleep(5)  # retry delay

    async def _wait_for_disconnect(self):
        # Implement your logic to detect disconnection
        while self._running and self._client.connected:
            await asyncio.sleep(1)

    def stop(self):
        self._running = False
        if self._client:
            # close connections cleanly
            pass

    def send_command(self, cmd):
        if self._client and self._client.connected:
            return asyncio.create_task(self._client.send_command(cmd))

    


