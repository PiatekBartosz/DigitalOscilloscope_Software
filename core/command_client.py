import asyncio
import logging

logger = logging.getLogger(__name__)

FRAME_SYNC  = bytes([0xAD, 0xC1])
FRAME_LEN   = 6
ADC_MASK    = 0x3FFF


class CommandClient:
    def __init__(self, host: str, port: int, sample_cb=None):
        """
        host      : device IP address
        port      : TCP port (8888)
        sample_cb : callable(ch1: int, ch2: int) called for each ADC frame
        """
        self.host      = host
        self.port      = port
        self.sample_cb = sample_cb
        self.writer    = None
        self.connected = False
        self._recv_task = None

    async def connect(self):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        self.writer    = writer
        self.connected = True
        logger.info("Connected to %s:%d", self.host, self.port)
        self._recv_task = asyncio.create_task(self._receive_loop(reader))

    async def disconnect(self):
        self.connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
        logger.info("Disconnected")

    async def send_command(self, cmd: str):
        if not self.connected or not self.writer:
            return
        try:
            self.writer.write(cmd.encode() + b'\n')
            await self.writer.drain()
        except Exception as e:
            logger.error("Send error: %s", e)
            self.connected = False

    async def _receive_loop(self, reader: asyncio.StreamReader):
        buf = bytearray()
        try:
            while True:
                chunk = await reader.read(256)
                if not chunk:
                    break
                buf.extend(chunk)
                buf = self._parse_frames(buf)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Receive error: %s", e)
        finally:
            self.connected = False

    def _parse_frames(self, buf: bytearray) -> bytearray:
        """Extract all complete ADC frames from buf, call sample_cb for each."""
        while len(buf) >= FRAME_LEN:
            idx = buf.find(FRAME_SYNC)
            if idx == -1:
                buf = buf[-1:]
                break
            if idx > 0:
                del buf[:idx]
            if len(buf) < FRAME_LEN:
                break
            ch1 = ((buf[2] << 8) | buf[3]) & ADC_MASK
            ch2 = ((buf[4] << 8) | buf[5]) & ADC_MASK
            if self.sample_cb:
                try:
                    self.sample_cb(ch1, ch2)
                except Exception as e:
                    logger.error("sample_cb error: %s", e)
            del buf[:FRAME_LEN]
        return buf
