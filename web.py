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

from agent_local import PhoneAgentLocal, CallRequest, CallResult
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
            cursor: pointer;
            transition: background 0.2s;
        }
        .history-item:hover { background: #1a1a1a; }
        .history-item .phone { font-weight: 600; }
        .history-item .objective { color: #888; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 400px; }
        .history-item .status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .history-item .status.success { background: #1e3a1e; color: #4ade80; }
        .history-item .status.failed { background: #3a1e1e; color: #f87171; }

        /* Modal styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: #1a1a1a;
            border-radius: 12px;
            max-width: 700px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid #333;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #333;
        }
        .modal-header h3 { margin: 0; }
        .modal-close {
            background: none;
            border: none;
            color: #888;
            font-size: 24px;
            cursor: pointer;
        }
        .modal-close:hover { color: #fff; }
        .modal-body { padding: 20px; }
        .modal-meta {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }
        .modal-meta-item label { display: block; color: #888; font-size: 12px; margin-bottom: 4px; }
        .modal-meta-item span { font-size: 14px; }
        .modal-transcript {
            background: #0a0a0a;
            border-radius: 8px;
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
        }
        .modal-transcript h4 { margin: 0 0 12px 0; color: #888; font-size: 14px; }

        .add-context-btn {
            color: #4a9eff;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            padding: 8px 0;
        }

        /* Cheat Sheet Styles */
        .cheatsheet {
            margin-top: 16px;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }
        .cheatsheet-toggle {
            width: 100%;
            background: #1a1a1a;
            color: #888;
            border: none;
            padding: 12px 16px;
            text-align: left;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cheatsheet-toggle:hover { background: #222; color: #aaa; }
        .cheatsheet-toggle .arrow { transition: transform 0.2s; }
        .cheatsheet-toggle.open .arrow { transform: rotate(180deg); }
        .cheatsheet-content {
            display: none;
            padding: 16px;
            background: #111;
            font-size: 13px;
            line-height: 1.6;
        }
        .cheatsheet-content.open { display: block; }
        .cheatsheet-section { margin-bottom: 16px; }
        .cheatsheet-section:last-child { margin-bottom: 0; }
        .cheatsheet-section h4 {
            color: #4a9eff;
            margin: 0 0 8px 0;
            font-size: 13px;
        }
        .cheatsheet-section code {
            background: #222;
            padding: 2px 6px;
            border-radius: 4px;
            color: #7cb342;
            font-family: monospace;
        }
        .cheatsheet-section pre {
            background: #1a1a1a;
            padding: 12px;
            border-radius: 6px;
            margin: 8px 0;
            overflow-x: auto;
            color: #ccc;
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

                <!-- Cheat Sheet -->
                <div class="cheatsheet">
                    <button class="cheatsheet-toggle" onclick="toggleCheatsheet(this)">
                        <span>📋 Context Cheat Sheet</span>
                        <span class="arrow">▼</span>
                    </button>
                    <div class="cheatsheet-content">
                        <div class="cheatsheet-section">
                            <h4>🔄 Call Transfer (Conference to your phone)</h4>
                            <pre>TRANSFER_TO: 7025551234
TRANSFER_IF: They ask to speak to a human</pre>
                            <p>AI will say "Please hold" and conference you into the call when condition is met.</p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>📞 Lead Qualification</h4>
                            <pre>WHO YOU ARE: Sales rep for ABC Company
TRANSFER_TO: 7025551234
TRANSFER_IF: Budget over $1000 and timeline under 30 days</pre>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>🍕 Order Placement</h4>
                            <pre>WHO YOU ARE: Customer ordering for Scott
DELIVERY ADDRESS: 123 Main St
PAYMENT: Cash on delivery</pre>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>📅 Appointment Reminder</h4>
                            <pre>WHO YOU ARE: AI assistant for Dr. Smith's office
APPOINTMENT: Tomorrow at 2:30 PM
PATIENT NAME: John Doe</pre>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>💡 Tips</h4>
                            <p>• Keys are shown to AI as context</p>
                            <p>• <code>TRANSFER_TO</code> + <code>TRANSFER_IF</code> enables call transfer</p>
                            <p>• Keep objectives short and clear</p>
                            <p>• AI won't make up info it doesn't have</p>
                        </div>
                    </div>
                </div>
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

    <!-- Call Detail Modal -->
    <div class="modal-overlay" id="call-modal" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <div class="modal-header">
                <h3 id="modal-title">Call Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-meta">
                    <div class="modal-meta-item">
                        <label>Phone</label>
                        <span id="modal-phone"></span>
                    </div>
                    <div class="modal-meta-item">
                        <label>Duration</label>
                        <span id="modal-duration"></span>
                    </div>
                    <div class="modal-meta-item">
                        <label>Date</label>
                        <span id="modal-date"></span>
                    </div>
                    <div class="modal-meta-item">
                        <label>Status</label>
                        <span id="modal-status"></span>
                    </div>
                </div>
                <div class="modal-meta-item" style="margin-bottom: 16px;">
                    <label>Objective</label>
                    <span id="modal-objective"></span>
                </div>
                <div class="modal-meta-item" style="margin-bottom: 16px;">
                    <label>Summary</label>
                    <span id="modal-summary"></span>
                </div>
                <div class="modal-transcript">
                    <h4>Transcript</h4>
                    <div id="modal-transcript"></div>
                </div>
            </div>
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

        function toggleCheatsheet(btn) {
            btn.classList.toggle('open');
            const content = btn.nextElementSibling;
            content.classList.toggle('open');
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
                    <div class="history-item" onclick="showCallDetails('${item.id}')">
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

        async function showCallDetails(callId) {
            try {
                const response = await fetch(`/api/call/${callId}`);
                const call = await response.json();

                // Populate modal
                document.getElementById('modal-phone').textContent = call.phone;
                document.getElementById('modal-duration').textContent = call.duration.toFixed(1) + 's';
                document.getElementById('modal-date').textContent = new Date(call.timestamp).toLocaleString();
                document.getElementById('modal-status').innerHTML = call.success
                    ? '<span style="color: #4ade80;">Success</span>'
                    : '<span style="color: #f87171;">Failed</span>';
                document.getElementById('modal-objective').textContent = call.objective;
                document.getElementById('modal-summary').textContent = call.summary || 'No summary';

                // Populate transcript
                const transcriptContainer = document.getElementById('modal-transcript');
                transcriptContainer.innerHTML = call.transcript.map(msg => `
                    <div class="transcript-entry ${msg.role}">
                        <div class="transcript-role">${msg.role === 'user' ? '👤 Caller' : '🤖 AI'}</div>
                        <div>${msg.content}</div>
                    </div>
                `).join('') || '<p style="color: #666;">No transcript</p>';

                // Show modal
                document.getElementById('call-modal').classList.add('active');
            } catch (error) {
                console.error('Failed to load call details:', error);
            }
        }

        function closeModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('call-modal').classList.remove('active');
            }
        }

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

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

    # Create agent with local STT/TTS
    agent = PhoneAgentLocal()

    # Set up callbacks for live updates
    def on_state(state):
        # Map ConversationState to UI status
        status_map = {
            "idle": "idle",
            "listening": "connected",
            "processing": "speaking",
            "speaking": "speaking",
            "completed": "ended",
            "failed": "failed"
        }
        status = status_map.get(state.value, state.value)
        asyncio.create_task(broadcast({
            "type": "status",
            "status": status
        }))

    def on_transcript(role, text):
        asyncio.create_task(broadcast({
            "type": "transcript",
            "role": role,
            "text": text
        }))

    # Register callbacks on the agent
    agent.on_state_change(on_state)
    agent.on_transcript(on_transcript)

    # Store agent
    active_calls[call_id] = agent

    # Start call in background
    async def run_call():
        try:
            # Broadcast dialing status
            await broadcast({
                "type": "status",
                "status": "dialing"
            })

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
                "collected_info": {},  # Not used in local engine
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

    agent = active_calls[call_id]
    # End the call by setting flag and hanging up
    agent._call_active = False
    try:
        agent.modem.hangup()
    except:
        pass
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


@app.get("/api/call/{call_id}")
async def get_call_details(call_id: str):
    """Get full details for a specific call"""
    file_path = os.path.join(config.CALLS_DIR, call_id)

    if not os.path.exists(file_path):
        raise HTTPException(404, "Call not found")

    try:
        with open(file_path) as f:
            data = json.load(f)
            return {
                "id": call_id,
                "timestamp": data.get("timestamp", ""),
                "phone": data.get("phone", ""),
                "objective": data.get("objective", ""),
                "context": data.get("context", {}),
                "success": data.get("success", False),
                "summary": data.get("summary", ""),
                "transcript": data.get("transcript", []),
                "duration": data.get("duration_seconds", 0),
                "recording_path": data.get("recording_path", "")
            }
    except Exception as e:
        raise HTTPException(500, f"Failed to read call: {str(e)}")


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
