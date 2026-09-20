from __future__ import annotations
import asyncio
from fastapi import WebSocket


class RunConnectionManager:
    """Fan-out broadcaster for a single run_id's WebSocket subscribers."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, run_id: str, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(run_id, []).append(ws)

    async def disconnect(self, run_id: str, ws: WebSocket):
        async with self._lock:
            conns = self._connections.get(run_id, [])
            if ws in conns:
                conns.remove(ws)

    async def broadcast(self, run_id: str, message: dict):
        conns = self._connections.get(run_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(run_id, ws)


manager = RunConnectionManager()
