from asyncio import sleep

from tdmclient import ClientAsync


class Robot:
    def __init__(self):
        self.client = None

    def __enter__(self):
        try:
            print("Connecting to Thymio...")
            self.client = ClientAsync().__enter__()

        except ConnectionRefusedError:
            print("Unable to connect to Thymio")

    def __exit__(self, *args):
        if self.client:
            print("Disconnecting from Thymio...")
            self.client.__exit__(*args)

    async def dance(self):
        print("Dancing!")

    async def move_away(self):
        if not self.client:
            return

        print("Moving away!")

        await self.send({"motor.left.target": [-200], "motor.right.target": [-200]})
        await sleep(2)
        await self.send({"motor.left.target": [0], "motor.right.target": [0]})

    async def send(self, commands: dict):
        assert self.client

        with await self.client.lock(wait_for_busy_node=False) as node:
            await node.set_variables(commands)
