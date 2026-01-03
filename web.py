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
from agent_incoming import IncomingCallHandler
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Phone Agent")

# Store for active calls and websocket connections
active_calls: dict = {}
websocket_connections: list[WebSocket] = []

# Incoming call listener
incoming_handler: Optional[IncomingCallHandler] = None
incoming_listener_task: Optional[asyncio.Task] = None

# Settings file path
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


def load_settings() -> dict:
    """Load settings from file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}


def save_settings(settings: dict):
    """Save settings to file"""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


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

        /* Tab styles */
        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
            padding-bottom: 0;
        }
        .tab-btn {
            padding: 12px 24px;
            background: none;
            border: none;
            color: #888;
            font-size: 14px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: all 0.2s;
        }
        .tab-btn:hover { color: #ccc; }
        .tab-btn.active {
            color: #4a9eff;
            border-bottom-color: #4a9eff;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .settings-group {
            margin-bottom: 24px;
        }
        .settings-group h4 {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        .settings-grid .form-group { margin-bottom: 0; }
        .settings-saved {
            color: #4ade80;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .settings-saved.show { opacity: 1; }

        .saved-indicator {
            color: #4ade80;
            font-size: 14px;
            margin-left: 12px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .saved-indicator.show { opacity: 1; }

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

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('calls')">Make a Call</button>
            <button class="tab-btn" onclick="switchTab('incoming')">Incoming</button>
            <button class="tab-btn" onclick="switchTab('settings')">Settings</button>
            <button class="tab-btn" onclick="switchTab('apikeys')">API Keys</button>
            <button class="tab-btn" onclick="switchTab('integrations')">Integrations</button>
        </div>

        <!-- Calls Tab -->
        <div class="tab-content active" id="tab-calls">

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
                            <h4>ℹ️ Your Settings are Auto-Included</h4>
                            <p>Info from the Settings tab (name, address, vehicle, etc.) is automatically available to the AI. You only need to add <strong>per-call context</strong> here.</p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>📞 Callback Request</h4>
                            <pre>TRANSFER_IF: They ask to speak to a human</pre>
                            <p>If the caller wants to speak to someone, AI will offer a callback: <em>"I'll have them call you right back!"</em> and end the call. You'll see this in the call log.</p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>🚗 Example: Get a Quote</h4>
                            <p><strong>Objective:</strong> Get a quote for window tinting with the darkest legal tint</p>
                            <pre>TRANSFER_IF: They want to speak to the owner</pre>
                            <p><em>Vehicle info comes from Settings.</em></p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>🍕 Example: Place an Order</h4>
                            <p><strong>Objective:</strong> Order a large pepperoni pizza for delivery</p>
                            <pre>SPECIAL_INSTRUCTIONS: Extra crispy
STRICT_BUDGET: $25</pre>
                            <p><em>Name, address, payment come from Settings.</em></p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>💰 Budget Control</h4>
                            <pre>STRICT_BUDGET: $50</pre>
                            <p>AI will decline upsells and extras that would push the total over the specified amount.</p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>📅 Example: Make Appointment</h4>
                            <p><strong>Objective:</strong> Schedule an oil change for tomorrow morning</p>
                            <pre>PREFERRED_TIME: Between 9am and 11am</pre>
                            <p><em>Name, callback, vehicle come from Settings.</em></p>
                        </div>

                        <div class="cheatsheet-section">
                            <h4>💡 Tips</h4>
                            <p>• Settings are merged with per-call context</p>
                            <p>• AI introduces itself as an AI assistant</p>
                            <p>• Keep objectives clear and specific</p>
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

        </div><!-- End Calls Tab -->

        <!-- Settings Tab -->
        <div class="tab-content" id="tab-settings">
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                    <h3>My Information</h3>
                    <span class="settings-saved" id="settings-saved">✓ Saved</span>
                </div>
                <p style="color: #888; margin-bottom: 24px;">This information will be available to the AI when making calls on your behalf.</p>

                <div class="settings-group">
                    <h4>Personal Details</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Full Name</label>
                            <input type="text" id="setting-name" placeholder="Scott Stevenson" />
                        </div>
                        <div class="form-group">
                            <label>Callback Number</label>
                            <input type="tel" id="setting-callback" placeholder="702-555-1234" />
                        </div>
                        <div class="form-group">
                            <label>Email</label>
                            <input type="email" id="setting-email" placeholder="scott@example.com" />
                        </div>
                        <div class="form-group">
                            <label>Company (optional)</label>
                            <input type="text" id="setting-company" placeholder="Acme Inc" />
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>SMS Notifications</h4>
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 12px; cursor: pointer;">
                            <input type="checkbox" id="setting-sms-enabled" style="width: 20px; height: 20px;" />
                            <span>Send SMS summary after each call</span>
                        </label>
                        <p style="color: #888; font-size: 12px; margin-top: 8px;">
                            AI will text a call summary to your callback number when calls end.
                        </p>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Address</h4>
                    <div class="form-group">
                        <label>Street Address</label>
                        <input type="text" id="setting-address" placeholder="123 Main Street" />
                    </div>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>City</label>
                            <input type="text" id="setting-city" placeholder="Las Vegas" />
                        </div>
                        <div class="form-group">
                            <label>State</label>
                            <input type="text" id="setting-state" placeholder="NV" />
                        </div>
                        <div class="form-group">
                            <label>ZIP Code</label>
                            <input type="text" id="setting-zip" placeholder="89101" />
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Payment (for orders)</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Card Number</label>
                            <input type="text" id="setting-card" placeholder="•••• •••• •••• 1234" />
                        </div>
                        <div class="form-group">
                            <label>Expiration</label>
                            <input type="text" id="setting-exp" placeholder="MM/YY" />
                        </div>
                        <div class="form-group">
                            <label>CVV</label>
                            <input type="text" id="setting-cvv" placeholder="•••" />
                        </div>
                        <div class="form-group">
                            <label>Billing ZIP</label>
                            <input type="text" id="setting-billing-zip" placeholder="89101" />
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Vehicle (for auto services)</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Year</label>
                            <input type="text" id="setting-vehicle-year" placeholder="2008" />
                        </div>
                        <div class="form-group">
                            <label>Make</label>
                            <input type="text" id="setting-vehicle-make" placeholder="Mercury" />
                        </div>
                        <div class="form-group">
                            <label>Model</label>
                            <input type="text" id="setting-vehicle-model" placeholder="Mariner" />
                        </div>
                        <div class="form-group">
                            <label>Color</label>
                            <input type="text" id="setting-vehicle-color" placeholder="Silver" />
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
            </div>
        </div><!-- End Settings Tab -->

        <!-- Incoming Calls Tab -->
        <div class="tab-content" id="tab-incoming">
            <!-- Listener Status Card -->
            <div class="card" style="margin-bottom: 20px;">
                <h3>Incoming Call Listener</h3>
                <div style="display: flex; align-items: center; gap: 16px; margin-top: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div id="listener-dot" style="width: 12px; height: 12px; border-radius: 50%; background: #666;"></div>
                        <span id="listener-status">Not listening</span>
                    </div>
                    <button class="btn btn-primary" id="listener-toggle-btn" onclick="toggleListener()">
                        Start Listening
                    </button>
                </div>
                <div id="incoming-call-alert" style="display: none; margin-top: 16px; padding: 16px; background: #1e3a1e; border-radius: 8px; border: 1px solid #2d5a2d;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 20px;">📞</span>
                        <span><strong>Incoming call from:</strong> <span id="incoming-caller-id">Unknown</span></span>
                    </div>
                </div>
                <div id="incoming-transcript" style="display: none; margin-top: 16px; max-height: 300px; overflow-y: auto; background: #0a0a0a; border-radius: 8px; padding: 16px;"></div>
            </div>

            <!-- Settings Card -->
            <div class="card">
                <h3>Incoming Call Settings</h3>
                <p style="color: #888; margin-bottom: 24px;">Configure how the AI handles incoming calls.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 12px; cursor: pointer;">
                            <input type="checkbox" id="incoming-enabled" style="width: 20px; height: 20px;" />
                            <span>Enable incoming call handling</span>
                        </label>
                        <p style="color: #888; font-size: 12px; margin-top: 8px;">
                            When enabled, AI will automatically answer incoming calls.
                        </p>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>AI Persona</h4>
                    <div class="form-group">
                        <label>Persona / Instructions</label>
                        <textarea id="incoming-persona" rows="4" placeholder="You are a personal assistant answering calls for {MY_NAME}. Be helpful and professional." style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff; resize: vertical;"></textarea>
                        <p style="color: #888; font-size: 12px; margin-top: 8px;">
                            Use {MY_NAME}, {COMPANY}, etc. to insert your settings.
                        </p>
                    </div>
                </div>

                <div class="settings-group">
                    <h4>Greeting</h4>
                    <div class="form-group">
                        <label>Initial Greeting</label>
                        <textarea id="incoming-greeting" rows="2" placeholder="Hello, you've reached {MY_NAME}'s AI assistant. How can I help you today?" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff; resize: vertical;"></textarea>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveIncomingSettings()">Save Incoming Settings</button>
                <span id="incoming-saved" class="saved-indicator">Saved!</span>
            </div>
        </div><!-- End Incoming Tab -->

        <!-- API Keys Tab -->
        <div class="tab-content" id="tab-apikeys">
            <div class="card">
                <h3>API Keys</h3>
                <p style="color: #888; margin-bottom: 24px;">Configure your LLM provider and API keys.</p>

                <div class="settings-group">
                    <h4>LLM Provider</h4>
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="api-provider" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                            <option value="claude">Claude (Anthropic)</option>
                            <option value="openai">OpenAI</option>
                            <option value="ollama">Ollama (Local)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Model (optional - leave blank for default)</label>
                        <input type="text" id="api-model" placeholder="e.g., claude-3-5-haiku-latest, gpt-4o, llama3:8b" />
                    </div>
                </div>

                <div class="settings-group" id="anthropic-key-group">
                    <h4>Anthropic API Key</h4>
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-anthropic-key" placeholder="sk-ant-..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-anthropic-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <div class="settings-group" id="openai-key-group">
                    <h4>OpenAI API Key</h4>
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-openai-key" placeholder="sk-..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-openai-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save API Keys</button>
                <span id="apikeys-saved" class="saved-indicator">Saved!</span>
            </div>
        </div><!-- End API Keys Tab -->

        <!-- Integrations Tab -->
        <div class="tab-content" id="tab-integrations">
            <div class="card">
                <h3>Integrations</h3>
                <p style="color: #888; margin-bottom: 24px;">Connect calendar and other services.</p>

                <div class="settings-group">
                    <h4>Calendar Provider</h4>
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="calendar-provider" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                            <option value="">None</option>
                            <option value="cal.com">Cal.com</option>
                            <option value="calendly">Calendly</option>
                        </select>
                    </div>
                </div>

                <div class="settings-group" id="calcom-settings" style="display: none;">
                    <h4>Cal.com Settings</h4>
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="calcom-api-key" placeholder="cal_live_..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('calcom-api-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Event Type ID</label>
                        <input type="text" id="calcom-event-type" placeholder="123456" />
                    </div>
                </div>

                <div class="settings-group" id="calendly-settings" style="display: none;">
                    <h4>Calendly Settings</h4>
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="calendly-api-key" placeholder="eyJraWQ..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('calendly-api-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>User URI</label>
                        <input type="text" id="calendly-user-uri" placeholder="https://api.calendly.com/users/..." />
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveIntegrations()">Save Integrations</button>
                <span id="integrations-saved" class="saved-indicator">Saved!</span>
            </div>
        </div><!-- End Integrations Tab -->

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
            } else if (data.type === 'incoming_listener_status') {
                updateListenerStatus(data.listening);
            } else if (data.type === 'incoming_call') {
                showIncomingCall(data.caller_id);
            } else if (data.type === 'incoming_transcript') {
                addIncomingTranscript(data.role, data.text);
            } else if (data.type === 'incoming_status') {
                if (data.status === 'ended') {
                    hideIncomingCall();
                }
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

        // Tab switching
        function switchTab(tabName) {
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Update content
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        // Settings field mapping
        const settingsFields = {
            'MY_NAME': 'setting-name',
            'CALLBACK_NUMBER': 'setting-callback',
            'EMAIL': 'setting-email',
            'COMPANY': 'setting-company',
            'ADDRESS': 'setting-address',
            'CITY': 'setting-city',
            'STATE': 'setting-state',
            'ZIP': 'setting-zip',
            'CARD_NUMBER': 'setting-card',
            'CARD_EXP': 'setting-exp',
            'CARD_CVV': 'setting-cvv',
            'BILLING_ZIP': 'setting-billing-zip',
            'VEHICLE_YEAR': 'setting-vehicle-year',
            'VEHICLE_MAKE': 'setting-vehicle-make',
            'VEHICLE_MODEL': 'setting-vehicle-model',
            'VEHICLE_COLOR': 'setting-vehicle-color'
        };

        // Checkbox settings (handled differently)
        const checkboxFields = {
            'SMS_ENABLED': 'setting-sms-enabled'
        };

        async function loadSettings() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();

                // Load text fields
                for (const [key, fieldId] of Object.entries(settingsFields)) {
                    const field = document.getElementById(fieldId);
                    if (field && settings[key]) {
                        field.value = settings[key];
                    }
                }

                // Load checkbox fields
                for (const [key, fieldId] of Object.entries(checkboxFields)) {
                    const field = document.getElementById(fieldId);
                    if (field) {
                        field.checked = settings[key] === true || settings[key] === 'true';
                    }
                }
            } catch (error) {
                console.error('Failed to load settings:', error);
            }
        }

        async function saveSettings() {
            const settings = {};

            // Save text fields
            for (const [key, fieldId] of Object.entries(settingsFields)) {
                const field = document.getElementById(fieldId);
                if (field && field.value.trim()) {
                    settings[key] = field.value.trim();
                }
            }

            // Save checkbox fields
            for (const [key, fieldId] of Object.entries(checkboxFields)) {
                const field = document.getElementById(fieldId);
                if (field) {
                    settings[key] = field.checked;
                }
            }

            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                // Show saved indicator
                const saved = document.getElementById('settings-saved');
                saved.classList.add('show');
                setTimeout(() => saved.classList.remove('show'), 2000);
            } catch (error) {
                alert('Failed to save settings: ' + error);
            }
        }

        // Toggle password visibility
        function togglePassword(fieldId) {
            const field = document.getElementById(fieldId);
            const btn = field.nextElementSibling;
            if (field.type === 'password') {
                field.type = 'text';
                btn.textContent = 'Hide';
            } else {
                field.type = 'password';
                btn.textContent = 'Show';
            }
        }

        // Incoming Call Settings
        async function loadIncomingSettings() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                const incoming = settings.incoming || {};

                document.getElementById('incoming-enabled').checked = incoming.ENABLED === true;
                document.getElementById('incoming-persona').value = incoming.PERSONA || '';
                document.getElementById('incoming-greeting').value = incoming.GREETING || '';
            } catch (error) {
                console.error('Failed to load incoming settings:', error);
            }
        }

        async function saveIncomingSettings() {
            const incoming = {
                ENABLED: document.getElementById('incoming-enabled').checked,
                PERSONA: document.getElementById('incoming-persona').value,
                GREETING: document.getElementById('incoming-greeting').value
            };

            try {
                // Get current settings and merge
                const response = await fetch('/api/settings');
                const settings = await response.json();
                settings.incoming = incoming;

                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                const saved = document.getElementById('incoming-saved');
                saved.classList.add('show');
                setTimeout(() => saved.classList.remove('show'), 2000);
            } catch (error) {
                alert('Failed to save incoming settings: ' + error);
            }
        }

        // API Keys
        async function loadApiKeys() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                const apiKeys = settings.api_keys || {};

                document.getElementById('api-provider').value = apiKeys.LLM_PROVIDER || 'ollama';
                document.getElementById('api-model').value = apiKeys.LLM_MODEL || '';
                document.getElementById('api-anthropic-key').value = apiKeys.ANTHROPIC_API_KEY || '';
                document.getElementById('api-openai-key').value = apiKeys.OPENAI_API_KEY || '';
            } catch (error) {
                console.error('Failed to load API keys:', error);
            }
        }

        async function saveApiKeys() {
            const apiKeys = {
                LLM_PROVIDER: document.getElementById('api-provider').value,
                LLM_MODEL: document.getElementById('api-model').value,
                ANTHROPIC_API_KEY: document.getElementById('api-anthropic-key').value,
                OPENAI_API_KEY: document.getElementById('api-openai-key').value
            };

            try {
                // Get current settings and merge
                const response = await fetch('/api/settings');
                const settings = await response.json();
                settings.api_keys = apiKeys;

                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                const saved = document.getElementById('apikeys-saved');
                saved.classList.add('show');
                setTimeout(() => saved.classList.remove('show'), 2000);
            } catch (error) {
                alert('Failed to save API keys: ' + error);
            }
        }

        // Integrations
        async function loadIntegrations() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                const integrations = settings.integrations || {};

                document.getElementById('calendar-provider').value = integrations.CALENDAR_PROVIDER || '';
                document.getElementById('calcom-api-key').value = integrations.CAL_COM_API_KEY || '';
                document.getElementById('calcom-event-type').value = integrations.CAL_COM_EVENT_TYPE_ID || '';
                document.getElementById('calendly-api-key').value = integrations.CALENDLY_API_KEY || '';
                document.getElementById('calendly-user-uri').value = integrations.CALENDLY_USER_URI || '';

                // Show/hide provider-specific settings
                updateCalendarProviderUI();
            } catch (error) {
                console.error('Failed to load integrations:', error);
            }
        }

        function updateCalendarProviderUI() {
            const provider = document.getElementById('calendar-provider').value;
            document.getElementById('calcom-settings').style.display = provider === 'cal.com' ? 'block' : 'none';
            document.getElementById('calendly-settings').style.display = provider === 'calendly' ? 'block' : 'none';
        }

        async function saveIntegrations() {
            const integrations = {
                CALENDAR_PROVIDER: document.getElementById('calendar-provider').value,
                CAL_COM_API_KEY: document.getElementById('calcom-api-key').value,
                CAL_COM_EVENT_TYPE_ID: document.getElementById('calcom-event-type').value,
                CALENDLY_API_KEY: document.getElementById('calendly-api-key').value,
                CALENDLY_USER_URI: document.getElementById('calendly-user-uri').value
            };

            try {
                // Get current settings and merge
                const response = await fetch('/api/settings');
                const settings = await response.json();
                settings.integrations = integrations;

                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                const saved = document.getElementById('integrations-saved');
                saved.classList.add('show');
                setTimeout(() => saved.classList.remove('show'), 2000);
            } catch (error) {
                alert('Failed to save integrations: ' + error);
            }
        }

        // Calendar provider change handler
        document.addEventListener('DOMContentLoaded', function() {
            const calendarProvider = document.getElementById('calendar-provider');
            if (calendarProvider) {
                calendarProvider.addEventListener('change', updateCalendarProviderUI);
            }
        });

        // Incoming call listener functions
        let isListening = false;

        function updateListenerStatus(listening) {
            isListening = listening;
            const dot = document.getElementById('listener-dot');
            const status = document.getElementById('listener-status');
            const btn = document.getElementById('listener-toggle-btn');

            if (listening) {
                dot.style.background = '#28a745';
                dot.style.animation = 'pulse 2s infinite';
                status.textContent = 'Listening for calls...';
                btn.textContent = 'Stop Listening';
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-danger');
            } else {
                dot.style.background = '#666';
                dot.style.animation = 'none';
                status.textContent = 'Not listening';
                btn.textContent = 'Start Listening';
                btn.classList.remove('btn-danger');
                btn.classList.add('btn-primary');
            }
        }

        async function toggleListener() {
            try {
                if (isListening) {
                    await fetch('/api/incoming/stop', { method: 'POST' });
                } else {
                    const response = await fetch('/api/incoming/start', { method: 'POST' });
                    if (!response.ok) {
                        const data = await response.json();
                        alert('Failed to start listener: ' + (data.detail || 'Unknown error'));
                    }
                }
            } catch (error) {
                alert('Error: ' + error);
            }
        }

        function showIncomingCall(callerId) {
            const alert = document.getElementById('incoming-call-alert');
            const callerIdSpan = document.getElementById('incoming-caller-id');
            const transcript = document.getElementById('incoming-transcript');

            callerIdSpan.textContent = callerId;
            alert.style.display = 'block';
            transcript.style.display = 'block';
            transcript.innerHTML = '';
        }

        function hideIncomingCall() {
            const alert = document.getElementById('incoming-call-alert');
            alert.style.display = 'none';
        }

        function addIncomingTranscript(role, text) {
            const container = document.getElementById('incoming-transcript');
            const entry = document.createElement('div');
            entry.className = 'transcript-entry ' + role;
            entry.innerHTML = `
                <div class="transcript-role">${role === 'user' ? '👤 Caller' : '🤖 AI'}</div>
                <div>${text}</div>
            `;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        async function loadListenerStatus() {
            try {
                const response = await fetch('/api/incoming/status');
                const data = await response.json();
                updateListenerStatus(data.listening);
            } catch (error) {
                console.error('Failed to load listener status:', error);
            }
        }

        // Initialize
        connectWebSocket();
        loadHistory();
        loadSettings();
        loadIncomingSettings();
        loadApiKeys();
        loadIntegrations();
        loadListenerStatus();
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

    # Merge saved settings with call context
    settings = load_settings()
    merged_context = {**settings, **request.context}  # Call context overrides settings

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
                context=merged_context
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


@app.get("/api/settings")
async def get_settings():
    """Get user settings"""
    return load_settings()


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update user settings"""
    save_settings(settings)
    return {"status": "saved"}


# Incoming call listener endpoints
@app.get("/api/incoming/status")
async def get_incoming_status():
    """Get incoming call listener status"""
    global incoming_handler, incoming_listener_task

    is_listening = (incoming_listener_task is not None and
                    not incoming_listener_task.done())

    return {
        "listening": is_listening,
        "enabled": load_settings().get("incoming", {}).get("ENABLED", False)
    }


@app.post("/api/incoming/start")
async def start_incoming_listener():
    """Start the incoming call listener"""
    global incoming_handler, incoming_listener_task

    # Check if already running
    if incoming_listener_task and not incoming_listener_task.done():
        return {"status": "already_running"}

    # Check if enabled in settings
    settings = load_settings()
    if not settings.get("incoming", {}).get("ENABLED", False):
        raise HTTPException(400, "Incoming calls not enabled in settings")

    # Create handler
    incoming_handler = IncomingCallHandler()

    # Set up callbacks
    def on_incoming(caller_id):
        asyncio.create_task(broadcast({
            "type": "incoming_call",
            "caller_id": caller_id
        }))

    def on_state(state):
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
            "type": "incoming_status",
            "status": status
        }))

    def on_transcript(role, text):
        asyncio.create_task(broadcast({
            "type": "incoming_transcript",
            "role": role,
            "text": text
        }))

    incoming_handler.on_incoming_call(on_incoming)
    incoming_handler.on_state_change(on_state)
    incoming_handler.on_transcript(on_transcript)

    # Start listener in background
    incoming_listener_task = asyncio.create_task(incoming_handler.start_listening())

    await broadcast({
        "type": "incoming_listener_status",
        "listening": True
    })

    return {"status": "started"}


@app.post("/api/incoming/stop")
async def stop_incoming_listener():
    """Stop the incoming call listener"""
    global incoming_handler, incoming_listener_task

    if incoming_handler:
        incoming_handler.stop_listening()

    if incoming_listener_task:
        incoming_listener_task.cancel()
        try:
            await incoming_listener_task
        except asyncio.CancelledError:
            pass
        incoming_listener_task = None

    incoming_handler = None

    await broadcast({
        "type": "incoming_listener_status",
        "listening": False
    })

    return {"status": "stopped"}


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
