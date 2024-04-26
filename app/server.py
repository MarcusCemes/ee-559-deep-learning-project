from aiohttp import web


class Server:

    def __init__(self):
        self.app = web.Application()
        self.runner = web.AppRunner(self.app)
        self.sockets = set()
        self.status = "idle"

        self.app.add_routes([web.get("/", self.handle), web.get("/ws", self.socket)])

    async def __aenter__(self):
        await self.runner.setup()

        site = web.TCPSite(self.runner, "0.0.0.0", 8080)
        await site.start()

        return self

    async def __aexit__(self, *_):
        await self.app.shutdown()
        await self.runner.cleanup()

    async def broadcast(self, status: str):
        for ws in self.sockets:
            await ws.send_str(status)

    async def handle(self, _: web.Request):
        return web.FileResponse("assets/index.html")

    async def socket(self, request: web.Request):
        ws = web.WebSocketResponse()
        self.sockets.add(ws)

        try:
            await ws.prepare(request)
            await ws.send_str("idle")

            async for _ in ws:
                pass

        finally:
            self.sockets.remove(ws)

        return ws
