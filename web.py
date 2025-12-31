"""
Web UI for AI Phone Agent

FastAPI server with:
- REST API for making calls
- WebSocket for live updates
- Simple HTML frontend
"""

import asyncio
import json
import logging
from typing import Optional
from datetime import datetime
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from agent import PhoneAgent, CallRequest, CallResult
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Phone Agent")

# Store for active calls and websocket connections
active_calls: dict = {}
websocket_connections: list[WebSocket] = []


class CallRequestModel(BaseModel):
    phone: str
    objective: str
    context: dict = {}


class CallHistoryItem(BaseModel):
    id: str
    timestamp: str
    phone: str
    objective: str
    success: bool
    summary: str
    duration: float


async def broadcast(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    for ws in websocket_connections[:]:
        try:
            await ws.send_json(message)
        except:
            websocket_connections.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Phone Agent</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            margin-bottom: 30px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        h1::before { content: '📞'; }

        .card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }

        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #aaa;
        }
        input, textarea {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #333;
            background: #0a0a0a;
            color: #fff;
            font-size: 16px;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #4a9eff;
        }
        textarea { resize: vertical; min-height: 80px; }

        .context-fields { margin-top: 10px; }
        .context-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .context-row input { flex: 1; }
        .context-row button {
            padding: 8px 16px;
            background: #333;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
        }

        .btn {
            padding: 14px 28px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4a9eff, #2d7dd2);
            color: #fff;
        }
        .btn-primary:hover { transform: translateY(-2px); }
        .btn-primary:disabled {
            background: #333;
            cursor: not-allowed;
            transform: none;
        }
        .btn-danger { background: #dc3545; color: #fff; }
        .btn-secondary { background: #333; color: #fff; }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: #0a0a0a;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #666;
        }
        .status-dot.idle { background: #666; }
        .status-dot.dialing { background: #ffc107; animation: pulse 1s infinite; }
        .status-dot.ringing { background: #ffc107; animation: pulse 0.5s infinite; }
        .status-dot.connected { background: #28a745; }
        .status-dot.speaking { background: #4a9eff; animation: pulse 0.3s infinite; }
        .status-dot.ended { background: #666; }
        .status-dot.failed { background: #dc3545; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .transcript {
            background: #0a0a0a;
            border-radius: 8px;
            padding: 16px;
            max-height: 400px;
            overflow-y: auto;
        }
        .transcript-entry {
            margin-bottom: 12px;
            padding: 10px 14px;
            border-radius: 8px;
        }
        .transcript-entry.user {
            background: #1e3a5f;
            margin-left: 40px;
        }
        .transcript-entry.assistant {
            background: #2d2d2d;
            margin-right: 40px;
        }
        .transcript-role {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
        }

        .result-card {
            background: #1a2e1a;
            border: 1px solid #2d5a2d;
        }
        .result-card.failed {
            background: #2e1a1a;
            border: 1px solid #5a2d2d;
        }
        .result-info { margin-top: 16px; }
        .result-info dt { color: #888; font-size: 14px; }
        .result-info dd { margin-bottom: 12px; margin-left: 0; }

        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: #0a0a0a;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .history-item .phone { font-weight: 600; }
        .history-item .objective { color: #888; font-size: 14px; }
        .history-item .status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .history-item .status.success { background: #1e3a1e; color: #4ade80; }
        .history-item .status.failed { background: #3a1e1e; color: #f87171; }

        .add-context-btn {
            color: #4a9eff;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            padding: 8px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Phone Agent</h1>

        <!-- Call Form -->
        <div class="card" id="call-form-card">
            <div class="form-group">
                <label>Phone Number</label>
                <input type="tel" id="phone" placeholder="775-555-1234" />
            </div>

            <div class="form-group">
                <label>Objective</label>
                <textarea id="objective" placeholder="What do you want the AI to accomplish on this call?"></textarea>
            </div>

            <div class="form-group">
                <label>Context (optional)</label>
                <div class="context-fields" id="context-fields">
                    <div class="context-row">
                        <input type="text" placeholder="Key (e.g., name)" class="context-key" />
                        <input type="text" placeholder="Value (e.g., Scott)" class="context-value" />
                        <button onclick="removeContextRow(this)">✕</button>
                    </div>
                </div>
                <button class="add-context-btn" onclick="addContextRow()">+ Add context field</button>
            </div>

            <button class="btn btn-primary" id="start-btn" onclick="startCall()">
                Start Call
            </button>
        </div>

        <!-- Active Call Status -->
        <div class="card" id="active-call-card" style="display: none;">
            <div class="status-bar">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Idle</span>
                <button class="btn btn-danger" onclick="endCall()" style="margin-left: auto;">End Call</button>
            </div>

            <div class="transcript" id="transcript"></div>
        </div>

        <!-- Result -->
        <div class="card result-card" id="result-card" style="display: none;">
            <h3 id="result-title">Call Complete</h3>
            <p id="result-summary"></p>
            <dl class="result-info" id="result-info"></dl>
        </div>

        <!-- History -->
        <div class="card">
            <h3 style="margin-bottom: 16px;">Recent Calls</h3>
            <div id="history"></div>
        </div>
    </div>

    <script>
        let ws;
        let currentCallId = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                setTimeout(connectWebSocket, 1000);
            };
        }

        function handleMessage(data) {
            if (data.type === 'status') {
                updateStatus(data.status);
            } else if (data.type === 'transcript') {
                addTranscript(data.role, data.text);
            } else if (data.type === 'result') {
                showResult(data);
            }
        }

        function updateStatus(status) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            dot.className = 'status-dot ' + status;
            text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }

        function addTranscript(role, text) {
            const container = document.getElementById('transcript');
            const entry = document.createElement('div');
            entry.className = 'transcript-entry ' + role;
            entry.innerHTML = `
                <div class="transcript-role">${role === 'user' ? '👤 Caller' : '🤖 AI'}</div>
                <div>${text}</div>
            `;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        function showResult(data) {
            const card = document.getElementById('result-card');
            const title = document.getElementById('result-title');
            const summary = document.getElementById('result-summary');
            const info = document.getElementById('result-info');

            card.style.display = 'block';
            card.className = 'card result-card ' + (data.success ? '' : 'failed');
            title.textContent = data.success ? '✓ Call Complete' : '✗ Call Failed';
            summary.textContent = data.summary;

            info.innerHTML = '';
            if (data.collected_info) {
                for (const [key, value] of Object.entries(data.collected_info)) {
                    info.innerHTML += `<dt>${key}</dt><dd>${value}</dd>`;
                }
            }
            info.innerHTML += `<dt>Duration</dt><dd>${data.duration.toFixed(1)}s</dd>`;

            // Reset form
            document.getElementById('call-form-card').style.display = 'block';
            document.getElementById('active-call-card').style.display = 'none';
            document.getElementById('start-btn').disabled = false;

            loadHistory();
        }

        function addContextRow() {
            const container = document.getElementById('context-fields');
            const row = document.createElement('div');
            row.className = 'context-row';
            row.innerHTML = `
                <input type="text" placeholder="Key" class="context-key" />
                <input type="text" placeholder="Value" class="context-value" />
                <button onclick="removeContextRow(this)">✕</button>
            `;
            container.appendChild(row);
        }

        function removeContextRow(btn) {
            btn.parentElement.remove();
        }

        function getContext() {
            const context = {};
            const rows = document.querySelectorAll('.context-row');
            rows.forEach(row => {
                const key = row.querySelector('.context-key').value.trim();
                const value = row.querySelector('.context-value').value.trim();
                if (key && value) {
                    context[key] = value;
                }
            });
            return context;
        }

        async function startCall() {
            const phone = document.getElementById('phone').value.trim();
            const objective = document.getElementById('objective').value.trim();
            const context = getContext();

            if (!phone || !objective) {
                alert('Please enter phone number and objective');
                return;
            }

            document.getElementById('start-btn').disabled = true;
            document.getElementById('call-form-card').style.display = 'none';
            document.getElementById('active-call-card').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';
            document.getElementById('transcript').innerHTML = '';

            try {
                const response = await fetch('/api/call', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phone, objective, context })
                });

                const data = await response.json();
                currentCallId = data.call_id;
            } catch (error) {
                alert('Failed to start call: ' + error);
                document.getElementById('start-btn').disabled = false;
                document.getElementById('call-form-card').style.display = 'block';
                document.getElementById('active-call-card').style.display = 'none';
            }
        }

        async function endCall() {
            if (currentCallId) {
                await fetch(`/api/call/${currentCallId}/end`, { method: 'POST' });
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const history = await response.json();

                const container = document.getElementById('history');
                container.innerHTML = history.map(item => `
                    <div class="history-item">
                        <div>
                            <div class="phone">${item.phone}</div>
                            <div class="objective">${item.objective}</div>
                        </div>
                        <span class="status ${item.success ? 'success' : 'failed'}">
                            ${item.success ? 'Success' : 'Failed'}
                        </span>
                    </div>
                `).join('') || '<p style="color: #666;">No calls yet</p>';
            } catch (error) {
                console.error('Failed to load history:', error);
            }
        }

        // Initialize
        connectWebSocket();
        loadHistory();
    </script>
</body>
</html>
"""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


@app.post("/api/call")
async def start_call(request: CallRequestModel):
    """Start a new AI phone call"""
    call_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create agent
    agent = PhoneAgent()

    # Set up callbacks for live updates
    def on_state(category, state):
        asyncio.create_task(broadcast({
            "type": "status",
            "status": state
        }))

    def on_transcript(role, text):
        asyncio.create_task(broadcast({
            "type": "transcript",
            "role": role,
            "text": text
        }))

    agent.on_state_change(on_state)
    agent.on_transcript(on_transcript)

    # Store agent
    active_calls[call_id] = agent

    # Start call in background
    async def run_call():
        try:
            result = await agent.call(CallRequest(
                phone=request.phone,
                objective=request.objective,
                context=request.context
            ))

            # Send result
            await broadcast({
                "type": "result",
                "success": result.success,
                "summary": result.summary,
                "collected_info": result.collected_info,
                "duration": result.duration_seconds
            })
        finally:
            if call_id in active_calls:
                del active_calls[call_id]

    asyncio.create_task(run_call())

    return {"call_id": call_id, "status": "started"}


@app.post("/api/call/{call_id}/end")
async def end_call(call_id: str):
    """End an active call"""
    if call_id not in active_calls:
        raise HTTPException(404, "Call not found")

    await active_calls[call_id].stop()
    return {"status": "ended"}


@app.get("/api/history")
async def get_history():
    """Get call history"""
    history = []

    if os.path.exists(config.CALLS_DIR):
        for filename in sorted(os.listdir(config.CALLS_DIR), reverse=True):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(config.CALLS_DIR, filename)) as f:
                        data = json.load(f)
                        history.append({
                            "id": filename,
                            "timestamp": data.get("timestamp", ""),
                            "phone": data.get("phone", ""),
                            "objective": data.get("objective", ""),
                            "success": data.get("success", False),
                            "summary": data.get("summary", ""),
                            "duration": data.get("duration_seconds", 0)
                        })
                except:
                    pass

    return history[:20]  # Last 20 calls


def main():
    """Run the web server"""
    print("\n" + "=" * 60)
    print("AI Phone Agent - Web UI")
    print("=" * 60)
    print("\nOpen http://localhost:8765 in your browser")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
