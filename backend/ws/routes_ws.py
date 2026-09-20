from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .manager import manager
from backend.auth import API_KEY

router = APIRouter()


@router.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str):
    if API_KEY and websocket.query_params.get("api_key") != API_KEY:
        await websocket.close(code=4401)  # custom close code: unauthorized
        return
    await manager.connect(run_id, websocket)
    try:
        while True:
            # Client doesn't need to send anything; keep the socket alive
            # and simply discard any inbound pings/messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(run_id, websocket)
