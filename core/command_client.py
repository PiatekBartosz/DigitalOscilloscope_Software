import asyncio
import logging

logger = logging.getLogger(__name__)

# Binary frame format (big-endian):
#   [0xAD][0xC1]           2 bytes  sync
#   [seq3..seq0]           4 bytes  uint32 waveform sequence number
#   [cnt1][cnt0]           2 bytes  uint16 sample count N
#   [N × (ch1_hi ch1_lo ch2_hi ch2_lo)]  N*4 bytes  samples
FRAME_SYNC       = bytes([0xAD, 0xC1])
FRAME_HEADER_LEN = 8
ADC_MASK         = 0x3FFF


class CommandClient:
    def __init__(self, host: str, port: int, sample_cb=None, text_cb=None):
        self.host       = host
        self.port       = port
        self.sample_cb  = sample_cb
        self.text_cb    = text_cb
        self.writer     = None
        self.connected  = False
        self._recv_task = None
        self._last_seq  = None

    async def connect(self):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        self.writer    = writer
        self.connected = True
        self._last_seq = None
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
                chunk = await reader.read(4096)
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
        while True:
            sync_idx = buf.find(FRAME_SYNC)
            nl_idx   = buf.find(b'\n')

            # Text reply arrives before the next binary frame (or no frame yet)
            if nl_idx != -1 and (sync_idx == -1 or nl_idx < sync_idx):
                line = buf[:nl_idx].decode(errors='replace').strip()
                del buf[:nl_idx + 1]
                if line and self.text_cb:
                    try:
                        self.text_cb(line)
                    except Exception as e:
                        logger.error("text_cb error: %s", e)
                continue

            if sync_idx == -1:
                break

            # Discard bytes before sync
            if sync_idx > 0:
                del buf[:sync_idx]

            # Wait for full header
            if len(buf) < FRAME_HEADER_LEN:
                break

            seq   = (buf[2] << 24) | (buf[3] << 16) | (buf[4] << 8) | buf[5]
            count = (buf[6] << 8) | buf[7]
            frame_len = FRAME_HEADER_LEN + count * 4

            # Wait for full payload
            if len(buf) < frame_len:
                break

            # Detect dropped waveforms
            if self._last_seq is not None:
                expected = (self._last_seq + 1) & 0xFFFFFFFF
                if seq != expected:
                    dropped = (seq - self._last_seq - 1) & 0xFFFFFFFF
                    logger.warning("Dropped %d waveform(s): seq %d -> %d", dropped, self._last_seq, seq)
            self._last_seq = seq

            # Deliver samples
            if self.sample_cb:
                for i in range(count):
                    offset = FRAME_HEADER_LEN + i * 4
                    ch1 = ((buf[offset]     << 8) | buf[offset + 1]) & ADC_MASK
                    ch2 = ((buf[offset + 2] << 8) | buf[offset + 3]) & ADC_MASK
                    try:
                        self.sample_cb(ch1, ch2)
                    except Exception as e:
                        logger.error("sample_cb error: %s", e)

            del buf[:frame_len]

        return buf
