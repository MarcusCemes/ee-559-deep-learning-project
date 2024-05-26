from dataclasses import asdict
from os.path import isfile
from weakref import WeakSet

from aiohttp import WSCloseCode, web

from .state import state


class Server:

    def __init__(self, port: int):
        self.app = web.Application()
        self.runner = web.AppRunner(self.app)
        self.sockets = WeakSet()
        self.port = port

        self.app.add_routes(
            [web.get("/ws", self.socket), web.get("/{tail:.*}", self.handle)]
        )

        self.app.on_shutdown.append(self.on_shutdown)

    async def __aenter__(self):
        await self.runner.setup()
        self.runner._shutdown_timeout = 3

        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()

        print(f"🚀 Server running on http://localhost:{self.port}")

        return self

    async def __aexit__(self, *_):
        print("🤚 Stopping server...")
        await self.runner.cleanup()

    async def on_shutdown(self, _):
        for ws in set(self.sockets):
            await ws.close(code=WSCloseCode.GOING_AWAY, message="Server shutdown")

    async def broadcast(self):
        for ws in self.sockets:
            await ws.send_json(asdict(state))

    async def handle(self, request: web.Request):
        path = "/index.html" if request.path == "/" else request.path
        path = "ui/dist" + path

        if ".." in path:
            raise web.HTTPForbidden()

        if not isfile(path):
            raise web.HTTPNotFound()

        return web.FileResponse(path)

    async def socket(self, request: web.Request):

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.sockets.add(ws)

        try:
            await ws.send_json(asdict(state))

            async for _ in ws:
                pass

        finally:
            self.sockets.discard(ws)

        return ws
