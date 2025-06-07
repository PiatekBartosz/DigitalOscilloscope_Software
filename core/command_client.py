import asyncio

class CommandClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.writer = None
        self.connected = False

    async def connect(self):
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            self.writer = writer
            self.connected = True
            print("Connected to STM32 command socket")
        except Exception as e:
            print(f"Connection failed: {e}")

    async def send_command(self, cmd: str):
        if self.connected and self.writer:
            try:
                self.writer.write(cmd.encode() + b'\n')
                await self.writer.drain()
            except Exception as e:
                print(f"Error sending command: {e}")