from asyncio import sleep
from enum import Enum

from tdmclient import ClientAsync


program = """
onevent request_sound
    call sound.system(event.args[0])
"""


class Sound(Enum):
    Boop = 2
    OkThen = 3
    Shock = 4
    Beep = 5
    Confirm = 6
    Affirmative = 7


class Color(Enum):
    Red = 0
    Green = 1
    Blue = 2


class Robot:
    def __init__(self, play_sound: bool):
        self.client = None
        self.should_play_sound = play_sound

    def __enter__(self):
        try:
            print("🤖 Connecting to Thymio...")
            self.client = ClientAsync().__enter__()

        except ConnectionRefusedError:
            print("⚠️ Unable to connect to Thymio")

        return self

    def __exit__(self, *args):
        if self.client:
            print("🛜 Disconnecting from Thymio...")
            self.client.__exit__(*args)

    async def prepare(self):
        if not self.client:
            return

        print("🖨️ Locking node and compiling program...")
        with await self.client.lock() as node:  # type: ignore
            await node.register_events([("request_sound", 1)])

            result = await node.compile(program)

            if result is not None:
                print("⚠️ Compilation failed:", result)

            await node.run()

    async def dance(self):
        if not self.client:
            return

        print("🕺 Dancing!")
        await self.move(200, -200, 0.25)
        await self.move(-200, 200, 0.5)
        await self.move(200, -200, 0.5)
        await self.move(-200, 200, 0.25)

    async def move_away(self):
        if not self.client:
            return

        print("🏃‍♂️ Moving away!")
        await self.move(-400, -400, 1)

    async def move(self, left: int, right: int, duration: float):
        await self.send({"motor.left.target": [left], "motor.right.target": [right]})
        await sleep(duration)
        await self.send({"motor.left.target": [0], "motor.right.target": [0]})

    async def play_sound(self, sound: Sound):
        if not self.client:
            return

        if self.should_play_sound:
            with await self.client.lock() as node:  # type: ignore
                await node.send_events({"request_sound": [sound.value]})

    async def top_led(self, color: Color):
        if not self.client:
            return

        match color:
            case Color.Red:
                value = [32, 0, 0]
            case Color.Green:
                value = [0, 32, 0]
            case Color.Blue:
                value = [0, 0, 32]

        with await self.client.lock() as node:  # type: ignore
            await node.set_variables({"leds.top": value})

    async def circle(self, on: bool):
        if not self.client:
            return

        value = 32 if on else 0
        value = [value] * 8

        with await self.client.lock() as node:  # type: ignore
            await node.set_variables({"leds.circle": value})

    async def send(self, commands: dict):
        assert self.client

        with await self.client.lock() as node:  # type: ignore
            await node.set_variables(commands)
