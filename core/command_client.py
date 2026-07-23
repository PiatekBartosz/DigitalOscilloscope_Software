import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Frame: sync(2), sequence(4), count(2), samples(N*4), big-endian.
FRAME_SYNC       = bytes([0xAD, 0xC1])
FRAME_HEADER_LEN = 8
ADC_MASK         = 0x3FFF


class CommandClient:
    def __init__(self, host: str, port: int, frame_cb=None, text_cb=None):
        self.host      = host
        self.port      = port
        self.frame_cb  = frame_cb
        self.text_cb   = text_cb
        self.writer    = None
        self.connected = False
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
            logger.warning("Cannot send '%s': not connected", cmd)
            return
        try:
            self.writer.write(cmd.encode() + b'\n')
            await self.writer.drain()
            logger.info("Sent: %s", cmd)
        except Exception as e:
            logger.error("Send error: %s", e)
            self.connected = False

    async def _receive_loop(self, reader: asyncio.StreamReader):
        buf = bytearray()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    logger.info("Connection closed by remote")
                    break
                logger.debug("TCP chunk: %d bytes (buf now %d bytes)",
                             len(chunk), len(buf) + len(chunk))
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

            if nl_idx != -1 and (sync_idx == -1 or nl_idx < sync_idx):
                line = buf[:nl_idx].decode(errors='replace').strip()
                del buf[:nl_idx + 1]
                if line:
                    logger.debug("RX text: %s", line)
                    if self.text_cb:
                        try:
                            self.text_cb(line)
                        except Exception as e:
                            logger.error("text_cb error: %s", e)
                continue

            if sync_idx == -1:
                logger.debug("No sync in buf (%d bytes) — waiting", len(buf))
                break

            if sync_idx > 0:
                logger.debug("Discarding %d bytes before sync", sync_idx)
                del buf[:sync_idx]

            if len(buf) < FRAME_HEADER_LEN:
                logger.debug("Incomplete header: have %d/%d bytes",
                             len(buf), FRAME_HEADER_LEN)
                break

            seq   = (buf[2] << 24) | (buf[3] << 16) | (buf[4] << 8) | buf[5]
            count = (buf[6] << 8) | buf[7]
            frame_len = FRAME_HEADER_LEN + count * 4

            logger.debug("Frame header: seq=%d samples=%d expected_len=%d buf=%d",
                         seq, count, frame_len, len(buf))

            if len(buf) < frame_len:
                logger.debug("Incomplete payload: have %d/%d bytes",
                             len(buf), frame_len)
                break

            if self._last_seq is not None:
                expected = (self._last_seq + 1) & 0xFFFFFFFF
                if seq != expected:
                    dropped = (seq - self._last_seq - 1) & 0xFFFFFFFF
                    logger.warning("Dropped %d waveform(s): seq %d -> %d",
                                   dropped, self._last_seq, seq)
            self._last_seq = seq

            if self.frame_cb:
                payload = bytes(buf[FRAME_HEADER_LEN:frame_len])
                raw = np.frombuffer(payload, dtype='>u2').reshape(count, 2)
                ch1 = (raw[:, 0] & ADC_MASK).astype(np.uint16)
                ch2 = (raw[:, 1] & ADC_MASK).astype(np.uint16)
                logger.info("Frame OK  seq=%d samples=%d "
                            "ch1=[%d..%d] ch2=[%d..%d]",
                            seq, count,
                            int(ch1.min()), int(ch1.max()),
                            int(ch2.min()), int(ch2.max()))
                try:
                    self.frame_cb(seq, ch1, ch2)
                    logger.debug("frame_cb delivered seq=%d", seq)
                except Exception as e:
                    logger.error("frame_cb error: %s", e)

            del buf[:frame_len]

        return buf
