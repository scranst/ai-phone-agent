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
from sms_ai import SMSAIHandler
import knowledge_base
import config
import database


def split_sms(text: str, max_len: int = 155) -> list:
    """Split a long message into SMS-sized chunks, trying to break at word boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Find last space before max_len
        break_point = text.rfind(' ', 0, max_len)
        if break_point == -1:
            # No space found, hard break
            break_point = max_len

        chunks.append(text[:break_point].strip())
        text = text[break_point:].strip()

    return chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Versabox v0.4-alpha")

# Store for active calls and websocket connections
active_calls: dict = {}
websocket_connections: list[WebSocket] = []

# Incoming call listener
incoming_handler: Optional[IncomingCallHandler] = None
incoming_listener_task: Optional[asyncio.Task] = None

# SMS AI handler
sms_handler: Optional[SMSAIHandler] = None
sms_monitor_task: Optional[asyncio.Task] = None

# Shared modem status (updated by SMS monitor, read by status endpoint)
import threading
shared_modem = None
shared_modem_lock = threading.Lock()
modem_status_cache = {"connected": False, "signal_strength": 0}

# Main event loop reference (for thread-safe async calls)
main_event_loop = None

# Pre-loaded conversation engine for fast call startup
from conversation_local import LocalConversationEngine
preloaded_conversation: Optional[LocalConversationEngine] = None


@app.on_event("startup")
async def startup_event():
    """Pre-load AI models and start listeners on server startup"""
    global preloaded_conversation, main_event_loop

    # Store main event loop for thread-safe async calls
    main_event_loop = asyncio.get_event_loop()

    logger.info("Pre-loading AI models for fast call startup...")

    # Create and initialize conversation engine in background
    preloaded_conversation = LocalConversationEngine()

    # Run initialization in executor to not block startup
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preloaded_conversation.initialize)

    logger.info("AI models pre-loaded and ready!")

    # Auto-start modem-dependent services
    # NOTE: Only ONE can use the modem at a time, so we prioritize:
    # 1. SMS monitor (handles both SMS commands AND can trigger calls)
    # 2. Incoming call listener (if SMS monitor not configured)
    settings = load_settings()
    # Use PRIMARY_PHONE from sms settings, or fall back to CALLBACK_NUMBER
    primary_phone = settings.get("sms", {}).get("PRIMARY_PHONE")
    if not primary_phone:
        # Try CALLBACK_NUMBER and normalize it
        callback = settings.get("CALLBACK_NUMBER") or settings.get("personal", {}).get("CALLBACK_NUMBER")
        if callback:
            import re
            primary_phone = re.sub(r'\D', '', callback)  # Strip non-digits
    incoming_enabled = settings.get("incoming", {}).get("ENABLED", False)

    if primary_phone:
        # SMS monitor takes priority - it can handle commands to make calls
        try:
            logger.info("Auto-starting SMS command monitor...")
            await start_sms_monitor()
        except Exception as e:
            logger.warning(f"Could not auto-start SMS monitor: {e}")
            # Fall back to incoming listener if SMS fails
            if incoming_enabled:
                try:
                    logger.info("Falling back to incoming call listener...")
                    await start_incoming_listener()
                except Exception as e2:
                    logger.warning(f"Could not start incoming listener either: {e2}")
    elif incoming_enabled:
        # No SMS configured, just start incoming listener
        try:
            logger.info("Auto-starting incoming call listener...")
            await start_incoming_listener()
        except Exception as e:
            logger.warning(f"Could not auto-start incoming listener: {e}")


# Settings are now stored in the database - these functions use api_keys module
def load_settings() -> dict:
    """Load settings from database"""
    import api_keys
    return api_keys.get_settings()


def save_settings(settings: dict):
    """Save settings to database"""
    import api_keys
    api_keys.save_settings(settings)


class CallRequestModel(BaseModel):
    phone: str
    objective: str = ""
    context: dict = {}
    agent_id: str = ""


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
    <title>Versabox v0.4-alpha</title>
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
        h1::before { content: '📦'; }

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
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
            <h1 style="margin-bottom: 0;">Versabox v0.4-alpha</h1>
            <div id="modem-status" style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: #1a1a1a; border-radius: 8px; border: 1px solid #333;">
                <div id="modem-dot" style="width: 10px; height: 10px; border-radius: 50%; background: #666;"></div>
                <span id="modem-text" style="font-size: 13px; color: #888;">Modem: Checking...</span>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('calls')">Call</button>
            <button class="tab-btn" onclick="switchTab('inbox')">Inbox</button>
            <button class="tab-btn" onclick="switchTab('leads')">Leads</button>
            <button class="tab-btn" onclick="switchTab('email')">Email</button>
            <button class="tab-btn" onclick="switchTab('agents')">Agents</button>
            <button class="tab-btn" onclick="switchTab('tools')">Tools</button>
            <button class="tab-btn" onclick="switchTab('settings')">Settings</button>
        </div>

        <!-- Calls Tab -->
        <div class="tab-content active" id="tab-calls">

        <!-- Call Form -->
        <div class="card" id="call-form-card">
            <div class="settings-grid" style="margin-bottom: 16px;">
                <div class="form-group">
                    <label>Phone Number</label>
                    <input type="tel" id="phone" placeholder="775-555-1234" />
                </div>
                <div class="form-group">
                    <label>Agent</label>
                    <select id="call-agent" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                        <option value="">Loading agents...</option>
                    </select>
                </div>
            </div>

            <div class="form-group">
                <label>Objective <span style="color: #888; font-size: 12px;">(optional - uses agent's default if blank)</span></label>
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

        <!-- Leads Tab -->
        <div class="tab-content" id="tab-leads">
            <!-- Leads Toolbar -->
            <div class="card" style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <button class="btn btn-primary" onclick="showAddLeadModal()">+ Add Lead</button>
                        <label class="btn btn-secondary" style="cursor: pointer;">
                            Import CSV
                            <input type="file" id="csv-import" accept=".csv" style="display: none;" onchange="handleCsvImport(this)">
                        </label>
                        <button class="btn btn-secondary" onclick="showListsModal()">Lists</button>
                    </div>
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <input type="text" id="leads-search" placeholder="Search leads..." style="width: 200px;" onkeyup="debounceSearch()">
                        <select id="leads-status-filter" style="padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;" onchange="loadLeads()">
                            <option value="">All Status</option>
                            <option value="NEW">New</option>
                            <option value="CONTACTED">Contacted</option>
                            <option value="ENGAGED">Engaged</option>
                            <option value="QUALIFIED">Qualified</option>
                            <option value="MEETING_BOOKED">Meeting Booked</option>
                            <option value="WON">Won</option>
                            <option value="LOST">Lost</option>
                        </select>
                    </div>
                </div>
                <!-- Bulk Actions Bar (hidden by default) -->
                <div id="bulk-actions-bar" style="display: none; margin-top: 16px; padding-top: 16px; border-top: 1px solid #333;">
                    <div style="display: flex; gap: 12px; align-items: center;">
                        <span id="selected-count" style="color: #4a9eff; font-weight: 500;">0 selected</span>
                        <button class="btn btn-secondary" onclick="addSelectedToList()">Add to List</button>
                        <button class="btn" style="background: #dc3545;" onclick="deleteSelectedLeads()">Delete Selected</button>
                        <button class="btn btn-secondary" onclick="clearSelection()">Clear Selection</button>
                    </div>
                </div>
            </div>

            <!-- Leads Stats -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px;">
                <div class="card" style="text-align: center; padding: 16px;">
                    <div style="font-size: 28px; font-weight: bold;" id="stat-total">0</div>
                    <div style="color: #888; font-size: 12px;">Total Leads</div>
                </div>
                <div class="card" style="text-align: center; padding: 16px;">
                    <div style="font-size: 28px; font-weight: bold; color: #4a9eff;" id="stat-new">0</div>
                    <div style="color: #888; font-size: 12px;">New</div>
                </div>
                <div class="card" style="text-align: center; padding: 16px;">
                    <div style="font-size: 28px; font-weight: bold; color: #ffc107;" id="stat-engaged">0</div>
                    <div style="color: #888; font-size: 12px;">Engaged</div>
                </div>
                <div class="card" style="text-align: center; padding: 16px;">
                    <div style="font-size: 28px; font-weight: bold; color: #28a745;" id="stat-booked">0</div>
                    <div style="color: #888; font-size: 12px;">Booked</div>
                </div>
            </div>

            <!-- Leads Table -->
            <div class="card">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333; text-align: left;">
                            <th style="padding: 12px; width: 40px;">
                                <input type="checkbox" id="select-all-leads" onchange="toggleSelectAll(this)" style="width: 18px; height: 18px; cursor: pointer;">
                            </th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Name</th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Company</th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Phone</th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Status</th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Last Contact</th>
                            <th style="padding: 12px; color: #888; font-weight: 500;">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="leads-table-body">
                        <tr><td colspan="7" style="padding: 40px; text-align: center; color: #666;">Loading leads...</td></tr>
                    </tbody>
                </table>
                <div id="leads-pagination" style="display: flex; justify-content: center; gap: 8px; margin-top: 16px;"></div>
            </div>
        </div><!-- End Leads Tab -->

        <!-- Add/Edit Lead Modal -->
        <div class="modal-overlay" id="lead-modal" onclick="closeLeadModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 600px;">
                <div class="modal-header">
                    <h3 id="lead-modal-title">Add Lead</h3>
                    <button class="modal-close" onclick="closeLeadModal()">&times;</button>
                </div>
                <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                    <input type="hidden" id="lead-id">

                    <!-- Section Tabs -->
                    <div style="display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid #444; padding-bottom: 8px;">
                        <button class="btn btn-secondary lead-tab-btn active" onclick="showLeadTab('basic')" data-tab="basic">Basic Info</button>
                        <button class="btn btn-secondary lead-tab-btn" onclick="showLeadTab('company')" data-tab="company">Company</button>
                        <button class="btn btn-secondary lead-tab-btn" onclick="showLeadTab('location')" data-tab="location">Location</button>
                        <button class="btn btn-secondary lead-tab-btn" onclick="showLeadTab('personalization')" data-tab="personalization">Personalization</button>
                    </div>

                    <!-- Basic Info Tab -->
                    <div class="lead-tab" id="lead-tab-basic">
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>First Name</label>
                                <input type="text" id="lead-first-name">
                            </div>
                            <div class="form-group">
                                <label>Last Name</label>
                                <input type="text" id="lead-last-name">
                            </div>
                            <div class="form-group">
                                <label>Email <span style="color: #888; font-size: 0.85em;">(or phone required)</span></label>
                                <input type="email" id="lead-email">
                            </div>
                            <div class="form-group">
                                <label>Phone <span style="color: #888; font-size: 0.85em;">(or email required)</span></label>
                                <input type="tel" id="lead-phone">
                            </div>
                            <div class="form-group">
                                <label>Phone Type</label>
                                <select id="lead-phone-type" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="mobile">Mobile</option>
                                    <option value="direct">Direct</option>
                                    <option value="office">Office</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>LinkedIn URL</label>
                                <input type="url" id="lead-linkedin-url" placeholder="https://linkedin.com/in/...">
                            </div>
                            <div class="form-group">
                                <label>Status</label>
                                <select id="lead-status" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="NEW">New</option>
                                    <option value="CONTACTED">Contacted</option>
                                    <option value="ENGAGED">Engaged</option>
                                    <option value="QUALIFIED">Qualified</option>
                                    <option value="MEETING_BOOKED">Meeting Booked</option>
                                    <option value="WON">Won</option>
                                    <option value="LOST">Lost</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Sentiment</label>
                                <select id="lead-sentiment" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Not Set</option>
                                    <option value="HOT">🔥 Hot</option>
                                    <option value="WARM">☀️ Warm</option>
                                    <option value="COLD">❄️ Cold</option>
                                    <option value="DNC">🚫 Do Not Contact</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Company Tab -->
                    <div class="lead-tab" id="lead-tab-company" style="display: none;">
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>Company</label>
                                <input type="text" id="lead-company">
                            </div>
                            <div class="form-group">
                                <label>Title</label>
                                <input type="text" id="lead-title">
                            </div>
                            <div class="form-group">
                                <label>Industry</label>
                                <input type="text" id="lead-industry">
                            </div>
                            <div class="form-group">
                                <label>Website</label>
                                <input type="url" id="lead-website" placeholder="https://...">
                            </div>
                            <div class="form-group">
                                <label>Company LinkedIn</label>
                                <input type="url" id="lead-company-linkedin" placeholder="https://linkedin.com/company/...">
                            </div>
                            <div class="form-group">
                                <label>Company Size</label>
                                <select id="lead-company-size" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="1-10">1-10</option>
                                    <option value="11-50">11-50</option>
                                    <option value="51-200">51-200</option>
                                    <option value="201-500">201-500</option>
                                    <option value="501-1000">501-1000</option>
                                    <option value="1001-5000">1001-5000</option>
                                    <option value="5001+">5001+</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Revenue</label>
                                <select id="lead-revenue" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="0-1M">$0 - $1M</option>
                                    <option value="1M-10M">$1M - $10M</option>
                                    <option value="10M-50M">$10M - $50M</option>
                                    <option value="50M-100M">$50M - $100M</option>
                                    <option value="100M+">$100M+</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Seniority</label>
                                <select id="lead-seniority" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="c_level">C-Level</option>
                                    <option value="vp">VP</option>
                                    <option value="director">Director</option>
                                    <option value="manager">Manager</option>
                                    <option value="senior">Senior</option>
                                    <option value="entry">Entry</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Department</label>
                                <select id="lead-department" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="sales">Sales</option>
                                    <option value="marketing">Marketing</option>
                                    <option value="engineering">Engineering</option>
                                    <option value="product">Product</option>
                                    <option value="operations">Operations</option>
                                    <option value="finance">Finance</option>
                                    <option value="hr">HR</option>
                                    <option value="it">IT</option>
                                    <option value="executive">Executive</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Funding Stage</label>
                                <select id="lead-funding-stage" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="">Unknown</option>
                                    <option value="bootstrapped">Bootstrapped</option>
                                    <option value="seed">Seed</option>
                                    <option value="series_a">Series A</option>
                                    <option value="series_b">Series B</option>
                                    <option value="series_c">Series C+</option>
                                    <option value="public">Public</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Technologies</label>
                                <input type="text" id="lead-technologies" placeholder="e.g., React, AWS, Salesforce">
                            </div>
                        </div>
                    </div>

                    <!-- Location Tab -->
                    <div class="lead-tab" id="lead-tab-location" style="display: none;">
                        <div class="settings-grid">
                            <div class="form-group" style="grid-column: span 2;">
                                <label>Address</label>
                                <input type="text" id="lead-address">
                            </div>
                            <div class="form-group">
                                <label>City</label>
                                <input type="text" id="lead-city">
                            </div>
                            <div class="form-group">
                                <label>State</label>
                                <input type="text" id="lead-state">
                            </div>
                            <div class="form-group">
                                <label>Country</label>
                                <input type="text" id="lead-country">
                            </div>
                            <div class="form-group">
                                <label>Timezone</label>
                                <input type="text" id="lead-timezone" placeholder="e.g., America/Los_Angeles">
                            </div>
                        </div>
                    </div>

                    <!-- Personalization Tab -->
                    <div class="lead-tab" id="lead-tab-personalization" style="display: none;">
                        <div class="form-group">
                            <label>Icebreaker</label>
                            <textarea id="lead-icebreaker" rows="2" style="width: 100%;" placeholder="Personalized intro or shared connection..."></textarea>
                        </div>
                        <div class="form-group">
                            <label>Trigger Event</label>
                            <textarea id="lead-trigger-event" rows="2" style="width: 100%;" placeholder="Why reaching out now (e.g., new funding, job change)..."></textarea>
                        </div>
                        <div class="form-group">
                            <label>Pain Points</label>
                            <textarea id="lead-pain-points" rows="2" style="width: 100%;" placeholder="Known challenges or needs..."></textarea>
                        </div>
                        <div class="form-group">
                            <label>Notes</label>
                            <textarea id="lead-notes" rows="3" style="width: 100%;" placeholder="General notes..."></textarea>
                        </div>
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>Custom 1</label>
                                <input type="text" id="lead-custom-1">
                            </div>
                            <div class="form-group">
                                <label>Custom 2</label>
                                <input type="text" id="lead-custom-2">
                            </div>
                            <div class="form-group">
                                <label>Custom 3</label>
                                <input type="text" id="lead-custom-3">
                            </div>
                            <div class="form-group">
                                <label>Custom 4</label>
                                <input type="text" id="lead-custom-4">
                            </div>
                            <div class="form-group">
                                <label>Custom 5</label>
                                <input type="text" id="lead-custom-5">
                            </div>
                            <div class="form-group">
                                <label>Source</label>
                                <input type="text" id="lead-source" placeholder="e.g., linkedin, referral, conference">
                            </div>
                        </div>
                    </div>

                    <div style="display: flex; gap: 12px; justify-content: flex-end; margin-top: 20px; border-top: 1px solid #444; padding-top: 16px;">
                        <button class="btn btn-secondary" onclick="closeLeadModal()">Cancel</button>
                        <button class="btn btn-primary" onclick="saveLead()">Save Lead</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- CSV Import Modal -->
        <div class="modal-overlay" id="csv-modal" onclick="closeCsvModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 700px;">
                <div class="modal-header">
                    <h3>Import CSV</h3>
                    <button class="modal-close" onclick="closeCsvModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <p style="color: #888; margin-bottom: 16px;">Map your CSV columns to lead fields:</p>
                    <div id="csv-mapping"></div>
                    <div id="csv-preview" style="margin-top: 16px;"></div>
                    <div style="display: flex; gap: 12px; justify-content: flex-end; margin-top: 20px;">
                        <button class="btn btn-secondary" onclick="closeCsvModal()">Cancel</button>
                        <button class="btn btn-primary" onclick="importCsv()">Import Leads</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Lead Detail Modal -->
        <div class="modal-overlay" id="lead-detail-modal" onclick="closeLeadDetailModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 800px;">
                <div class="modal-header">
                    <h3 id="lead-detail-title">Lead Details</h3>
                    <button class="modal-close" onclick="closeLeadDetailModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div id="lead-detail-content"></div>
                    <div style="margin-top: 20px;">
                        <h4 style="margin-bottom: 12px; color: #888;">Interaction History</h4>
                        <div id="lead-interactions" style="max-height: 300px; overflow-y: auto;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Inbox Tab - Unified Inbox -->
        <div class="tab-content" id="tab-inbox">
            <!-- Header with filters -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding: 0 4px;">
                <h2 style="margin: 0; display: flex; align-items: center; gap: 12px;">
                    Inbox
                    <span id="unread-badge" style="background: #dc3545; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 13px; display: none;">0</span>
                </h2>
                <button class="btn btn-primary" onclick="showSendSmsModal()">+ New Message</button>
            </div>

            <!-- Filter Bar -->
            <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center;">
                <!-- Channel Filter -->
                <div style="display: flex; gap: 4px; background: #222; border-radius: 8px; padding: 4px;">
                    <button class="inbox-filter-btn active" data-channel="" onclick="setInboxFilter('channel', '')">All</button>
                    <button class="inbox-filter-btn" data-channel="sms" onclick="setInboxFilter('channel', 'sms')">📱 SMS</button>
                    <button class="inbox-filter-btn" data-channel="email" onclick="setInboxFilter('channel', 'email')">📧 Email</button>
                    <button class="inbox-filter-btn" data-channel="call" onclick="setInboxFilter('channel', 'call')">📞 Calls</button>
                </div>

                <!-- Direction Filter -->
                <select id="inbox-direction-filter" onchange="setInboxFilter('direction', this.value)" style="padding: 8px 12px; background: #222; border: 1px solid #333; border-radius: 8px; color: #fff;">
                    <option value="">All Directions</option>
                    <option value="inbound">Inbound</option>
                    <option value="outbound">Outbound</option>
                </select>

                <!-- Search -->
                <div style="flex: 1; min-width: 200px;">
                    <input type="text" id="inbox-search" placeholder="Search messages..."
                           style="width: 100%; padding: 8px 12px; background: #222; border: 1px solid #333; border-radius: 8px; color: #fff;"
                           onkeyup="debounceInboxSearch(this.value)">
                </div>
            </div>

            <!-- Autopilot Queue (if pending responses) -->
            <div id="autopilot-queue-container" style="display: none; margin-bottom: 16px;">
                <div style="background: linear-gradient(135deg, #1a3a2a 0%, #0f2a1a 100%); border: 1px solid #2d5a4d; border-radius: 12px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #4ade80; display: flex; align-items: center; gap: 8px;">
                            🤖 AI Pending Responses
                            <span id="autopilot-queue-count" style="background: #4ade80; color: #000; padding: 2px 8px; border-radius: 10px; font-size: 12px;">0</span>
                        </h4>
                        <button class="btn btn-secondary" onclick="toggleAutopilotQueue()" style="padding: 4px 8px; font-size: 11px;">Hide</button>
                    </div>
                    <div id="autopilot-queue-list" style="display: flex; flex-direction: column; gap: 8px;"></div>
                </div>
            </div>

            <!-- Split View: Conversation List + Thread -->
            <div style="display: grid; grid-template-columns: 350px 1fr; gap: 16px; height: calc(100vh - 280px); min-height: 400px;">
                <!-- Left: Conversation List -->
                <div style="background: #111; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column;">
                    <div id="inbox-conversation-list" style="flex: 1; overflow-y: auto;">
                        <div style="text-align: center; padding: 60px 20px; color: #666;">
                            Loading conversations...
                        </div>
                    </div>
                </div>

                <!-- Right: Message Thread -->
                <div id="inbox-thread-panel" style="background: #111; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden;">
                    <!-- Empty State -->
                    <div id="inbox-thread-empty" style="flex: 1; display: flex; align-items: center; justify-content: center; color: #666;">
                        <div style="text-align: center;">
                            <p style="font-size: 48px; margin-bottom: 16px;">💬</p>
                            <p>Select a conversation to view messages</p>
                        </div>
                    </div>

                    <!-- Thread Header (hidden until conversation selected) -->
                    <div id="inbox-thread-header" style="display: none; border-bottom: 1px solid #333; padding: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <div id="inbox-thread-avatar" style="width: 40px; height: 40px; background: #4a9eff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 16px;">?</div>
                                <div>
                                    <h3 id="inbox-thread-title" style="margin: 0; font-size: 16px;">Contact</h3>
                                    <div id="inbox-thread-channels" style="display: flex; gap: 4px; margin-top: 4px;"></div>
                                </div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <button class="btn btn-secondary" id="inbox-autopilot-toggle" onclick="toggleThreadAutopilotInline()" style="padding: 6px 12px; font-size: 12px;" title="Toggle AI auto-reply">
                                    🤖 On
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Messages Container (hidden until conversation selected) -->
                    <div id="inbox-thread-messages" style="display: none; flex: 1; overflow-y: auto; padding: 16px; background: #0a0a0a;"></div>

                    <!-- Reply Input (hidden until conversation selected) -->
                    <div id="inbox-thread-input" style="display: none; border-top: 1px solid #333; padding: 12px; background: #111;">
                        <div style="display: flex; gap: 8px; align-items: flex-end;">
                            <textarea id="inbox-reply-message" rows="1" placeholder="Type a message..."
                                      style="flex: 1; padding: 10px 14px; background: #222; border: 1px solid #333; border-radius: 20px; color: #fff; resize: none; max-height: 100px; font-size: 15px;"
                                      oninput="this.style.height='auto'; this.style.height=Math.min(this.scrollHeight, 100)+'px';"
                                      onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();sendInboxReply();}"></textarea>
                            <button class="btn btn-primary" onclick="sendInboxReply()" style="padding: 10px 16px; border-radius: 20px; min-width: 60px;">
                                <span style="font-size: 16px;">↑</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <style>
            .inbox-filter-btn {
                padding: 6px 12px;
                border: none;
                background: transparent;
                color: #888;
                border-radius: 6px;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.2s;
            }
            .inbox-filter-btn:hover {
                background: #333;
                color: #fff;
            }
            .inbox-filter-btn.active {
                background: #4a9eff;
                color: #fff;
            }
            .inbox-conv-item {
                padding: 12px 16px;
                border-bottom: 1px solid #222;
                cursor: pointer;
                transition: background 0.2s;
            }
            .inbox-conv-item:hover {
                background: #1a1a1a;
            }
            .inbox-conv-item.active {
                background: #1a2a3a;
                border-left: 3px solid #4a9eff;
            }
            .inbox-conv-item.unread {
                background: #111;
            }
            .inbox-conv-item.unread .conv-name {
                font-weight: 600;
            }
            .channel-badge {
                display: inline-flex;
                align-items: center;
                gap: 2px;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                background: #333;
            }
            .channel-badge.sms { background: #2d3a4d; color: #4a9eff; }
            .channel-badge.email { background: #3d2d4d; color: #9a6eff; }
            .channel-badge.call { background: #2d4a3d; color: #4ade80; }
            .msg-bubble {
                max-width: 75%;
                padding: 10px 14px;
                border-radius: 18px;
                margin-bottom: 8px;
                font-size: 15px;
                line-height: 1.4;
            }
            .msg-bubble.inbound {
                background: #333;
                color: #fff;
                border-bottom-left-radius: 4px;
                align-self: flex-start;
            }
            .msg-bubble.outbound {
                background: #0b84fe;
                color: #fff;
                border-bottom-right-radius: 4px;
                align-self: flex-end;
            }
            .msg-bubble.ai-generated {
                background: linear-gradient(135deg, #1a3a2a 0%, #0b6644 100%);
                border: 1px solid #4ade80;
            }
            .msg-bubble.ai-generated::before {
                content: '🤖 ';
                font-size: 12px;
            }
            .msg-meta {
                font-size: 11px;
                color: #666;
                margin-top: 4px;
            }
            .autopilot-pending-card {
                background: #1a2a2a;
                border: 1px solid #2d5a4d;
                border-radius: 8px;
                padding: 12px;
            }
        </style>

        <!-- Legacy Conversation Detail Modal (for backward compatibility) -->
        <div class="modal-overlay" id="conversation-modal" onclick="closeConversationModal(event)" style="display: none;">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 500px; height: 85vh; display: flex; flex-direction: column; border-radius: 16px;">
                <div class="modal-header" style="border-bottom: 1px solid #333; padding: 16px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 40px; height: 40px; background: #4a9eff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 16px;" id="conversation-avatar">?</div>
                        <div>
                            <h3 id="conversation-title" style="margin: 0; font-size: 16px;">Conversation</h3>
                            <p id="conversation-channels" style="color: #888; font-size: 12px; margin: 2px 0 0 0;"></p>
                        </div>
                    </div>
                    <button class="modal-close" onclick="closeConversationModal()">&times;</button>
                </div>
                <div id="conversation-messages" style="flex: 1; overflow-y: auto; padding: 16px; background: #0a0a0a;"></div>
                <div style="border-top: 1px solid #333; padding: 12px; background: #111;">
                    <div style="display: flex; gap: 8px; align-items: flex-end;">
                        <textarea id="reply-message" rows="1" placeholder="Message" style="flex: 1; padding: 10px 14px; background: #222; border: 1px solid #333; border-radius: 20px; color: #fff; resize: none; max-height: 100px; font-size: 15px;"></textarea>
                        <button class="btn btn-primary" onclick="sendReply()" style="padding: 10px 16px; border-radius: 20px; min-width: 60px;">↑</button>
                    </div>
                </div>
            </div>
        </div><!-- End Inbox Tab -->

        <!-- Lead Lists Modal -->
        <div class="modal-overlay" id="lists-modal" onclick="closeListsModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 600px;">
                <div class="modal-header">
                    <h3>Lead Lists</h3>
                    <button class="modal-close" onclick="closeListsModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <!-- Create New List -->
                    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                        <input type="text" id="new-list-name" placeholder="New list name..." style="flex: 1;">
                        <button class="btn btn-primary" onclick="createNewList()">Create List</button>
                    </div>

                    <!-- Lists Table -->
                    <div id="lists-container">
                        <p style="color: #666; text-align: center;">Loading lists...</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Add to List Modal -->
        <div class="modal-overlay" id="add-to-list-modal" onclick="closeAddToListModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 400px;">
                <div class="modal-header">
                    <h3>Add to List</h3>
                    <button class="modal-close" onclick="closeAddToListModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <p style="margin-bottom: 16px;"><span id="add-to-list-count">0</span> lead(s) selected</p>
                    <div id="list-options-container">
                        <p style="color: #666; text-align: center;">Loading lists...</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Send SMS Modal -->
        <div class="modal-overlay" id="sms-modal" onclick="closeSmsModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 500px;">
                <div class="modal-header">
                    <h3>Send SMS</h3>
                    <button class="modal-close" onclick="closeSmsModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label>To (Phone Number)</label>
                        <input type="tel" id="sms-to" placeholder="775-555-1234" />
                    </div>
                    <div class="form-group">
                        <label>Message</label>
                        <textarea id="sms-message" rows="4" placeholder="Enter your message..." style="width: 100%;"></textarea>
                        <p style="color: #888; font-size: 12px; margin-top: 4px;"><span id="sms-char-count">0</span>/160 characters</p>
                    </div>
                    <div style="display: flex; gap: 12px; margin-top: 16px;">
                        <button class="btn btn-secondary" style="flex: 1;" onclick="closeSmsModal()">Cancel</button>
                        <button class="btn btn-primary" style="flex: 1;" onclick="sendSms()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Email Tab -->
        <div class="tab-content" id="tab-email">
            <div class="card" style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3>Email Accounts</h3>
                        <p style="color: #888;">Configure SMTP accounts for email campaigns. Emails are sent round-robin across all active accounts.</p>
                    </div>
                    <button class="btn btn-primary" onclick="openEmailAccountModal()">+ Add Account</button>
                </div>
            </div>

            <!-- Email Stats -->
            <div class="card" id="email-stats-card" style="margin-bottom: 20px;">
                <h4>Today's Stats</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px;">
                    <div style="background: #333; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;" id="stat-total-accounts">0</div>
                        <div style="color: #888; font-size: 12px;">Total Accounts</div>
                    </div>
                    <div style="background: #333; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold; color: #4ade80;" id="stat-active-accounts">0</div>
                        <div style="color: #888; font-size: 12px;">Active</div>
                    </div>
                    <div style="background: #333; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;" id="stat-sent-today">0</div>
                        <div style="color: #888; font-size: 12px;">Sent Today</div>
                    </div>
                    <div style="background: #333; padding: 16px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 24px; font-weight: bold;" id="stat-remaining-capacity">0</div>
                        <div style="color: #888; font-size: 12px;">Remaining Capacity</div>
                    </div>
                </div>
            </div>

            <!-- Email Accounts List -->
            <div class="card">
                <div id="email-accounts-list">
                    <p style="color: #888; text-align: center; padding: 40px;">Loading email accounts...</p>
                </div>
            </div>
        </div><!-- End Email Tab -->

        <!-- Add/Edit Email Account Modal -->
        <div class="modal-overlay" id="email-modal" onclick="closeEmailModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 600px;">
                <div class="modal-header">
                    <h3 id="email-modal-title">Add Email Account</h3>
                    <button class="modal-close" onclick="closeEmailModal()">&times;</button>
                </div>
                <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                    <input type="hidden" id="email-account-id">

                    <!-- Provider Preset -->
                    <div class="form-group" style="margin-bottom: 24px;">
                        <label>Provider Preset</label>
                        <select id="email-preset" onchange="applyEmailPreset()" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                            <option value="">Custom / Manual</option>
                            <option value="gmail">Gmail (Google Workspace)</option>
                            <option value="outlook">Outlook / Microsoft 365</option>
                            <option value="yahoo">Yahoo Mail</option>
                            <option value="zoho">Zoho Mail</option>
                            <option value="sendgrid">SendGrid</option>
                            <option value="mailgun">Mailgun</option>
                        </select>
                        <p id="preset-notes" style="color: #888; font-size: 12px; margin-top: 4px;"></p>
                    </div>

                    <!-- Basic Info -->
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Email Address *</label>
                            <input type="email" id="email-email" placeholder="sales@company.com">
                        </div>
                        <div class="form-group">
                            <label>Display Name</label>
                            <input type="text" id="email-display-name" placeholder="John from Company">
                        </div>
                    </div>

                    <!-- SMTP Settings -->
                    <h4 style="margin: 24px 0 16px;">SMTP Settings (Sending)</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>SMTP Host *</label>
                            <input type="text" id="email-smtp-host" placeholder="smtp.gmail.com">
                        </div>
                        <div class="form-group">
                            <label>Port *</label>
                            <input type="number" id="email-smtp-port" placeholder="587" value="587">
                        </div>
                    </div>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Username *</label>
                            <input type="text" id="email-smtp-username" placeholder="username@gmail.com">
                        </div>
                        <div class="form-group">
                            <label>Password / App Password *</label>
                            <input type="password" id="email-smtp-password" placeholder="App password or SMTP key">
                        </div>
                    </div>
                    <div class="form-group" style="margin-bottom: 24px;">
                        <label style="display: flex; align-items: center; gap: 8px;">
                            <input type="checkbox" id="email-smtp-tls" checked> Use TLS (recommended)
                        </label>
                    </div>

                    <!-- IMAP Settings (for reply detection) -->
                    <h4 style="margin: 24px 0 16px;">IMAP Settings (for reply detection - optional)</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>IMAP Host</label>
                            <input type="text" id="email-imap-host" placeholder="imap.gmail.com">
                        </div>
                        <div class="form-group">
                            <label>Port</label>
                            <input type="number" id="email-imap-port" placeholder="993" value="993">
                        </div>
                    </div>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Username</label>
                            <input type="text" id="email-imap-username" placeholder="Same as SMTP if blank">
                        </div>
                        <div class="form-group">
                            <label>Password</label>
                            <input type="password" id="email-imap-password" placeholder="Same as SMTP if blank">
                        </div>
                    </div>

                    <!-- Limits -->
                    <h4 style="margin: 24px 0 16px;">Sending Limits</h4>
                    <div class="settings-grid">
                        <div class="form-group">
                            <label>Daily Limit</label>
                            <input type="number" id="email-daily-limit" placeholder="100" value="100">
                        </div>
                        <div class="form-group">
                            <label>Hourly Limit</label>
                            <input type="number" id="email-hourly-limit" placeholder="20" value="20">
                        </div>
                        <div class="form-group">
                            <label>Min Delay (seconds)</label>
                            <input type="number" id="email-delay" placeholder="60" value="60">
                        </div>
                    </div>

                    <!-- Warmup -->
                    <div class="form-group" style="margin-top: 16px;">
                        <label style="display: flex; align-items: center; gap: 8px;">
                            <input type="checkbox" id="email-warmup"> Enable warmup mode (start slow, gradually increase volume)
                        </label>
                        <p style="color: #888; font-size: 12px; margin-top: 4px;">
                            Recommended for new accounts. Starts at 10/day and increases over 45 days.
                        </p>
                    </div>

                    <!-- Signature -->
                    <h4 style="margin: 24px 0 16px;">Email Signature (optional)</h4>
                    <div class="form-group">
                        <label>HTML Signature</label>
                        <textarea id="email-signature-html" rows="4" placeholder="<p>Best regards,<br>John Doe</p>"></textarea>
                    </div>
                </div>
                <div class="modal-footer" style="padding: 16px 24px; border-top: 1px solid #333; display: flex; gap: 12px; justify-content: space-between;">
                    <button class="btn btn-secondary" onclick="testEmailAccount()" id="test-email-btn">Test Connection</button>
                    <div style="display: flex; gap: 12px;">
                        <button class="btn btn-secondary" onclick="closeEmailModal()">Cancel</button>
                        <button class="btn btn-primary" onclick="saveEmailAccount()">Save Account</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Send Test Email Modal -->
        <div class="modal-overlay" id="test-email-modal" onclick="closeTestEmailModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 400px;">
                <div class="modal-header">
                    <h3>Send Test Email</h3>
                    <button class="modal-close" onclick="closeTestEmailModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <input type="hidden" id="test-email-account-id">
                    <div class="form-group">
                        <label>Send test email to:</label>
                        <input type="email" id="test-email-to" placeholder="your@email.com">
                    </div>
                </div>
                <div class="modal-footer" style="padding: 16px 24px; border-top: 1px solid #333; display: flex; gap: 12px; justify-content: flex-end;">
                    <button class="btn btn-secondary" onclick="closeTestEmailModal()">Cancel</button>
                    <button class="btn btn-primary" onclick="sendTestEmail()">Send Test</button>
                </div>
            </div>
        </div>

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

        <!-- Agents Tab -->
        <div class="tab-content" id="tab-agents">
            <div class="card" style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3>AI Agents</h3>
                        <p style="color: #888;">Configure different AI personas for different roles.</p>
                    </div>
                </div>
            </div>

            <div id="agents-list">
                <div class="card" style="text-align: center; padding: 40px; color: #666;">
                    Loading agents...
                </div>
            </div>
        </div>

        <!-- Agent Edit Modal -->
        <div class="modal-overlay" id="agent-modal" onclick="closeAgentModal(event)">
            <div class="modal" onclick="event.stopPropagation()" style="max-width: 800px;">
                <div class="modal-header">
                    <h3 id="agent-modal-title">Edit Agent</h3>
                    <button class="modal-close" onclick="closeAgentModal()">&times;</button>
                </div>
                <div class="modal-body" style="max-height: 75vh; overflow-y: auto;">
                    <input type="hidden" id="agent-id">

                    <!-- Agent Tabs -->
                    <div style="display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 1px solid #444; padding-bottom: 8px;">
                        <button class="btn btn-secondary agent-tab-btn active" onclick="showAgentTab('general')" data-tab="general">General</button>
                        <button class="btn btn-secondary agent-tab-btn" onclick="showAgentTab('persona')" data-tab="persona">Persona</button>
                        <button class="btn btn-secondary agent-tab-btn" onclick="showAgentTab('knowledge')" data-tab="knowledge">Knowledge</button>
                    </div>

                    <!-- General Tab -->
                    <div class="agent-tab" id="agent-tab-general">
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>Name</label>
                                <input type="text" id="agent-name">
                            </div>
                            <div class="form-group">
                                <label>Icon (emoji)</label>
                                <input type="text" id="agent-icon" style="width: 80px;">
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Objective</label>
                            <input type="text" id="agent-objective" placeholder="What is this agent's main goal?">
                            <p style="color: #888; font-size: 12px; margin-top: 4px;">
                                A short description of what this agent does.
                            </p>
                        </div>
                        <div class="settings-grid">
                            <div class="form-group">
                                <label>Model Tier</label>
                                <select id="agent-model-tier" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                                    <option value="haiku">Fast (Haiku) - Quick, cheap</option>
                                    <option value="sonnet">Smart (Sonnet) - Balanced</option>
                                    <option value="opus">Reasoning (Opus) - Best</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label style="display: flex; align-items: center; gap: 12px; cursor: pointer; padding-top: 32px;">
                                    <input type="checkbox" id="agent-enabled" style="width: 20px; height: 20px;">
                                    <span>Enabled</span>
                                </label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Tools</label>
                            <div id="agent-tools-list" style="display: flex; flex-wrap: wrap; gap: 8px;">
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-search_web" value="search_web"> search_web
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-get_movie_showtimes" value="get_movie_showtimes"> get_movie_showtimes
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-make_call" value="make_call"> make_call
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-send_sms" value="send_sms"> send_sms
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-search_contacts" value="search_contacts"> search_contacts
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-book_calendar" value="book_calendar"> book_calendar
                                </label>
                                <label style="display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #222; border-radius: 6px; cursor: pointer;">
                                    <input type="checkbox" id="tool-check_calendar" value="check_calendar"> check_calendar
                                </label>
                            </div>
                        </div>
                    </div>

                    <!-- Persona Tab -->
                    <div class="agent-tab" id="agent-tab-persona" style="display: none;">
                        <div class="form-group">
                            <label>Persona / Instructions</label>
                            <textarea id="agent-persona" rows="20" style="font-family: monospace; font-size: 13px;"></textarea>
                            <p style="color: #888; font-size: 12px; margin-top: 8px;">
                                Use {company_name}, {my_name}, {location} to insert context values.
                            </p>
                        </div>
                    </div>

                    <!-- Knowledge Tab -->
                    <div class="agent-tab" id="agent-tab-knowledge" style="display: none;">
                        <div class="form-group">
                            <label>Knowledge Base</label>
                            <p style="color: #888; font-size: 12px; margin-bottom: 12px;">
                                Information this agent can reference during conversations. Use Markdown format.
                            </p>
                            <textarea id="agent-knowledge" rows="25" style="font-family: monospace; font-size: 13px;" placeholder="# Company Info

## Products
- Product A: $99, does X
- Product B: $199, does Y

## FAQ
Q: What are your hours?
A: 9am-5pm Monday-Friday

## Policies
- 30-day return policy
- Free shipping over $50"></textarea>
                        </div>
                    </div>

                    <div style="display: flex; gap: 12px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #333;">
                        <button class="btn btn-secondary" style="flex: 1;" onclick="closeAgentModal()">Cancel</button>
                        <button class="btn btn-primary" style="flex: 1;" onclick="saveAgent()">Save Agent</button>
                    </div>
                </div>
            </div>
        </div><!-- End Agents Tab -->

        <!-- Tools Tab -->
        <div class="tab-content" id="tab-tools">
            <div class="card" style="margin-bottom: 20px;">
                <h3>LLM Provider</h3>
                <p style="color: #888; margin-bottom: 24px;">Configure your AI provider. Each agent can use a different model tier.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="api-provider" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;">
                            <option value="claude">Claude (Anthropic)</option>
                            <option value="openai">OpenAI</option>
                        </select>
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

            <div class="card" style="margin-bottom: 20px;">
                <h3>Google Custom Search</h3>
                <p style="color: #888; margin-bottom: 24px;">Enable web search capability for the AI.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>Google API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-google-key" placeholder="AIza..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-google-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Custom Search Engine ID</label>
                        <input type="text" id="api-google-cse-id" placeholder="abc123..." />
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save</button>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <h3>Apify</h3>
                <p style="color: #888; margin-bottom: 16px;">Enable advanced web scraping and data extraction.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>Apify API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-apify-key" placeholder="apify_api_..." style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-apify-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <div style="background: #1a1a1a; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; color: #888;">Available Actors</h4>
                    <div style="display: grid; gap: 12px;">
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="font-size: 20px;">🔍</span>
                            <div>
                                <div style="font-weight: 500; color: #fff;">code_crafter/leads-finder</div>
                                <div style="font-size: 12px; color: #888;">Find leads from LinkedIn profiles and searches</div>
                            </div>
                        </div>
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="font-size: 20px;">📇</span>
                            <div>
                                <div style="font-weight: 500; color: #fff;">vdrmota/contact-info-scraper</div>
                                <div style="font-size: 12px; color: #888;">Scrape contact info (emails, phones) from websites</div>
                            </div>
                        </div>
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="font-size: 20px;">🏠</span>
                            <div>
                                <div style="font-weight: 500; color: #fff;">tri_angle/airbnb-scraper</div>
                                <div style="font-size: 12px; color: #888;">Scrape Airbnb listings, prices, and availability</div>
                            </div>
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save</button>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <h3>Amadeus</h3>
                <p style="color: #888; margin-bottom: 16px;">Enable flight search and booking capabilities.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-amadeus-key" placeholder="Your Amadeus API Key" style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-amadeus-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>API Secret</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-amadeus-secret" placeholder="Your Amadeus API Secret" style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-amadeus-secret')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <div style="background: #1a1a1a; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; color: #888;">Capabilities</h4>
                    <div style="display: grid; gap: 8px; font-size: 13px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>✈️</span> Search flights by route, date, and passengers
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>💰</span> Get real-time pricing and availability
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>🎫</span> Book flights on behalf of users
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save</button>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <h3>NeverBounce</h3>
                <p style="color: #888; margin-bottom: 16px;">Verify email addresses before outreach.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-neverbounce-key" placeholder="Your NeverBounce API Key" style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-neverbounce-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <div style="background: #1a1a1a; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; color: #888;">Verification Results</h4>
                    <div style="display: grid; gap: 8px; font-size: 13px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #4ade80;">✓</span> Valid - Safe to send
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #fbbf24;">⚠</span> Catch-all - Accept all emails (risky)
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #f87171;">✗</span> Invalid - Don't send (bounces)
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #888;">?</span> Unknown - Could not verify
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save</button>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <h3>PhoneValidator</h3>
                <p style="color: #888; margin-bottom: 16px;">Validate phone numbers and detect line type.</p>

                <div class="settings-group">
                    <div class="form-group">
                        <label>API Key</label>
                        <div style="display: flex; gap: 8px;">
                            <input type="password" id="api-phonevalidator-key" placeholder="Your PhoneValidator API Key" style="flex: 1;" />
                            <button class="btn" onclick="togglePassword('api-phonevalidator-key')" style="padding: 8px 12px;">Show</button>
                        </div>
                    </div>
                </div>

                <div style="background: #1a1a1a; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 12px 0; font-size: 14px; color: #888;">Line Type Detection</h4>
                    <div style="display: grid; gap: 8px; font-size: 13px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>📱</span> Mobile - Best for SMS campaigns
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>📞</span> Landline - Voice calls only
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>🌐</span> VoIP - Internet-based (variable quality)
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span>❌</span> Invalid - Number doesn't exist
                        </div>
                    </div>
                </div>

                <button class="btn btn-primary" onclick="saveApiKeys()">Save</button>
            </div>

            <div class="card">
                <h3>Calendar Integration</h3>
                <p style="color: #888; margin-bottom: 24px;">Connect calendar for booking appointments.</p>

                <div class="settings-group">
                    <h4>Calendar Provider</h4>
                    <div class="form-group">
                        <label>Provider</label>
                        <select id="calendar-provider" style="width: 100%; padding: 12px; background: #333; border: 1px solid #444; border-radius: 8px; color: #fff;" onchange="toggleCalendarSettings()">
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

                <button class="btn btn-primary" onclick="saveIntegrations()">Save Calendar</button>
                <span id="integrations-saved" class="saved-indicator">Saved!</span>
            </div>
        </div><!-- End Tools Tab -->

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
                // Show active call card for SMS-triggered calls
                if (data.source === 'sms_call' && data.status !== 'idle' && data.status !== 'ended') {
                    document.getElementById('active-call-card').style.display = 'block';
                    document.getElementById('call-form-card').style.display = 'none';
                } else if (data.source === 'sms_call' && (data.status === 'idle' || data.status === 'ended')) {
                    document.getElementById('active-call-card').style.display = 'none';
                    document.getElementById('call-form-card').style.display = 'block';
                }
            } else if (data.type === 'transcript') {
                addTranscript(data.role, data.text);
                // Make sure card is visible for SMS-triggered calls
                if (data.source === 'sms_call') {
                    document.getElementById('active-call-card').style.display = 'block';
                }
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
            } else if (data.type === 'sms_monitor_status') {
                updateSmsMonitorStatus(data.monitoring);
            } else if (data.type === 'sms_received') {
                if (data.authorized) {
                    addSmsLog(data.sender, data.message, data.response);
                }
            } else if (data.type === 'sms_call_started') {
                addSmsLog('System', `Making call to ${data.contact || data.phone}`, data.objective);
                // Show active call card and clear transcript
                document.getElementById('active-call-card').style.display = 'block';
                document.getElementById('call-form-card').style.display = 'none';
                document.getElementById('transcript').innerHTML = '';
            }
            // Unified inbox events
            else if (data.type === 'new_message') {
                // Refresh inbox when new message arrives
                loadInbox();
                if (selectedInboxConversation === data.conversation) {
                    selectInboxConversation(selectedInboxConversation);
                }
            } else if (data.type === 'pending_response') {
                // New AI pending response - reload queue
                loadAutopilotQueue();
            } else if (data.type === 'unread_count') {
                // Update unread badge
                const badge = document.getElementById('unread-badge');
                if (data.total > 0) {
                    badge.textContent = data.total;
                    badge.style.display = 'inline';
                } else {
                    badge.style.display = 'none';
                }
            } else if (data.type === 'autopilot_approved' || data.type === 'autopilot_cancelled') {
                loadAutopilotQueue();
                loadInbox();
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

        async function loadCallAgents() {
            try {
                const response = await fetch('/api/agents');
                const agents = await response.json();
                const select = document.getElementById('call-agent');

                select.innerHTML = agents
                    .filter(a => a.enabled)
                    .map(a => `<option value="${a.id}">${a.icon || '🤖'} ${a.name}</option>`)
                    .join('');

                // Default to personal_assistant if available
                if (agents.find(a => a.id === 'personal_assistant')) {
                    select.value = 'personal_assistant';
                }
            } catch (error) {
                console.error('Failed to load agents:', error);
            }
        }

        async function startCall() {
            const phone = document.getElementById('phone').value.trim();
            const objective = document.getElementById('objective').value.trim();
            const agentId = document.getElementById('call-agent').value;
            const context = getContext();

            if (!phone) {
                alert('Please enter a phone number');
                return;
            }

            if (!agentId) {
                alert('Please select an agent');
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
                    body: JSON.stringify({ phone, objective, context, agent_id: agentId })
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

        // ========================================
        // MAIN OBJECTIVE
        // ========================================

        async function loadMainObjective() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                const objective = settings.MAIN_OBJECTIVE || 'No main objective set. Click to configure.';
                document.getElementById('main-objective-text').textContent = objective;
            } catch (error) {
                console.error('Failed to load main objective:', error);
            }
        }

        function editMainObjective() {
            const text = document.getElementById('main-objective-text').textContent;
            document.getElementById('main-objective-input').value = text;
            document.getElementById('main-objective-display').style.display = 'none';
            document.getElementById('main-objective-edit').style.display = 'block';
        }

        function cancelMainObjective() {
            document.getElementById('main-objective-display').style.display = 'block';
            document.getElementById('main-objective-edit').style.display = 'none';
        }

        async function saveMainObjective() {
            const objective = document.getElementById('main-objective-input').value.trim();
            if (!objective) {
                alert('Objective cannot be empty');
                return;
            }

            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                settings.MAIN_OBJECTIVE = objective;

                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                document.getElementById('main-objective-text').textContent = objective;
                cancelMainObjective();
            } catch (error) {
                alert('Failed to save: ' + error);
            }
        }

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
                const sms = settings.sms || {};
                const autopilot = settings.autopilot || {};

                document.getElementById('incoming-enabled').checked = incoming.ENABLED === true;
                document.getElementById('incoming-persona').value = incoming.PERSONA || '';
                document.getElementById('incoming-greeting').value = incoming.GREETING || '';
                document.getElementById('sms-primary-phone').value = sms.PRIMARY_PHONE || '';

                // Autopilot settings
                document.getElementById('autopilot-enabled').checked = autopilot.ENABLED === true;
                document.getElementById('autopilot-delay-min').value = autopilot.REPLY_DELAY_MIN || 30;
                document.getElementById('autopilot-delay-max').value = autopilot.REPLY_DELAY_MAX || 120;
            } catch (error) {
                console.error('Failed to load incoming settings:', error);
            }
        }

        async function saveInboxSettings() {
            const incoming = {
                ENABLED: document.getElementById('incoming-enabled').checked,
                PERSONA: document.getElementById('incoming-persona').value,
                GREETING: document.getElementById('incoming-greeting').value
            };

            const sms = {
                PRIMARY_PHONE: document.getElementById('sms-primary-phone').value
            };

            const autopilot = {
                ENABLED: document.getElementById('autopilot-enabled').checked,
                REPLY_DELAY_MIN: parseInt(document.getElementById('autopilot-delay-min').value) || 30,
                REPLY_DELAY_MAX: parseInt(document.getElementById('autopilot-delay-max').value) || 120
            };

            try {
                // Get current settings and merge
                const response = await fetch('/api/settings');
                const settings = await response.json();
                settings.incoming = incoming;
                settings.sms = sms;
                settings.autopilot = autopilot;

                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                const saved = document.getElementById('inbox-saved');
                saved.classList.add('show');
                setTimeout(() => saved.classList.remove('show'), 2000);
            } catch (error) {
                alert('Failed to save settings: ' + error);
            }
        }

        // API Keys
        async function loadApiKeys() {
            try {
                const response = await fetch('/api/settings');
                const settings = await response.json();
                const apiKeys = settings.api_keys || {};

                document.getElementById('api-provider').value = apiKeys.LLM_PROVIDER || 'claude';
                document.getElementById('api-anthropic-key').value = apiKeys.ANTHROPIC_API_KEY || '';
                document.getElementById('api-openai-key').value = apiKeys.OPENAI_API_KEY || '';
                document.getElementById('api-google-key').value = apiKeys.GOOGLE_API_KEY || '';
                document.getElementById('api-google-cse-id').value = apiKeys.GOOGLE_CSE_ID || '';
                document.getElementById('api-apify-key').value = apiKeys.APIFY_API_KEY || '';
                document.getElementById('api-amadeus-key').value = apiKeys.AMADEUS_API_KEY || '';
                document.getElementById('api-amadeus-secret').value = apiKeys.AMADEUS_API_SECRET || '';
                document.getElementById('api-neverbounce-key').value = apiKeys.NEVERBOUNCE_API_KEY || '';
                document.getElementById('api-phonevalidator-key').value = apiKeys.PHONEVALIDATOR_API_KEY || '';
            } catch (error) {
                console.error('Failed to load API keys:', error);
            }
        }

        async function saveApiKeys() {
            const apiKeys = {
                LLM_PROVIDER: document.getElementById('api-provider').value,
                ANTHROPIC_API_KEY: document.getElementById('api-anthropic-key').value,
                OPENAI_API_KEY: document.getElementById('api-openai-key').value,
                GOOGLE_API_KEY: document.getElementById('api-google-key').value,
                GOOGLE_CSE_ID: document.getElementById('api-google-cse-id').value,
                APIFY_API_KEY: document.getElementById('api-apify-key').value,
                AMADEUS_API_KEY: document.getElementById('api-amadeus-key').value,
                AMADEUS_API_SECRET: document.getElementById('api-amadeus-secret').value,
                NEVERBOUNCE_API_KEY: document.getElementById('api-neverbounce-key').value,
                PHONEVALIDATOR_API_KEY: document.getElementById('api-phonevalidator-key').value
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

        // ========================================
        // SMS COMMANDS
        // ========================================

        let isSmsMonitoring = false;

        function updateSmsMonitorStatus(monitoring) {
            isSmsMonitoring = monitoring;
            const dot = document.getElementById('sms-dot');
            const status = document.getElementById('sms-status');
            const btn = document.getElementById('sms-toggle-btn');
            const log = document.getElementById('sms-log');

            if (monitoring) {
                dot.style.background = '#28a745';
                dot.style.animation = 'pulse 2s infinite';
                status.textContent = 'Monitoring for SMS...';
                btn.textContent = 'Stop Monitoring';
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-danger');
                log.style.display = 'block';
            } else {
                dot.style.background = '#666';
                dot.style.animation = 'none';
                status.textContent = 'Not monitoring';
                btn.textContent = 'Start Monitoring';
                btn.classList.remove('btn-danger');
                btn.classList.add('btn-primary');
            }
        }

        async function toggleSmsMonitor() {
            try {
                if (isSmsMonitoring) {
                    await fetch('/api/sms/stop', { method: 'POST' });
                } else {
                    const response = await fetch('/api/sms/start', { method: 'POST' });
                    if (!response.ok) {
                        const data = await response.json();
                        alert('Failed to start SMS monitor: ' + (data.detail || 'Unknown error'));
                    }
                }
            } catch (error) {
                alert('Error: ' + error);
            }
        }

        function addSmsLog(sender, message, response) {
            const log = document.getElementById('sms-log');
            log.style.display = 'block';
            const entry = document.createElement('div');
            entry.style.marginBottom = '12px';
            entry.style.paddingBottom = '12px';
            entry.style.borderBottom = '1px solid #333';
            entry.innerHTML = `
                <div style="color: #4a9eff; font-size: 12px;">${new Date().toLocaleTimeString()}</div>
                <div><strong>From:</strong> ${sender}</div>
                <div><strong>Command:</strong> ${message}</div>
                <div style="color: #28a745;"><strong>Response:</strong> ${response}</div>
            `;
            log.insertBefore(entry, log.firstChild);
        }

        async function loadSmsStatus() {
            try {
                const response = await fetch('/api/sms/status');
                const data = await response.json();
                updateSmsMonitorStatus(data.monitoring);
            } catch (error) {
                console.error('Failed to load SMS status:', error);
            }
        }

        // ========================================
        // MODEM STATUS
        // ========================================

        async function loadModemStatus() {
            try {
                const response = await fetch('/api/modem/status');
                const data = await response.json();
                updateModemStatus(data);
            } catch (error) {
                console.error('Failed to load modem status:', error);
                updateModemStatus({ connected: false, error: 'Failed to check' });
            }
        }

        function updateModemStatus(data) {
            const dot = document.getElementById('modem-dot');
            const text = document.getElementById('modem-text');

            if (data.connected) {
                dot.style.background = '#28a745';
                const signal = data.signal_strength || 0;
                const signalBars = signal > 25 ? '▁▃▅▇' : signal > 15 ? '▁▃▅' : signal > 5 ? '▁▃' : '▁';
                text.textContent = `Modem: Connected ${signalBars}`;
                text.style.color = '#fff';
            } else {
                dot.style.background = '#dc3545';
                text.textContent = 'Modem: Not Connected';
                text.style.color = '#888';
            }
        }

        // Poll modem status every 30 seconds
        setInterval(loadModemStatus, 30000);

        // ========================================
        // CONVERSATIONS / INBOX
        // ========================================

        let currentConversation = null;

        async function loadConversations() {
            try {
                const response = await fetch('/api/conversations');
                const conversations = await response.json();

                const container = document.getElementById('conversation-list');

                if (!conversations || conversations.length === 0) {
                    container.innerHTML = `
                        <div style="text-align: center; padding: 40px; color: #666;">
                            No conversations yet. Messages will appear here.
                        </div>
                    `;
                    return;
                }

                container.innerHTML = conversations.map((conv, idx) => {
                    const name = conv.display_name || conv.contact_address;
                    const initial = name.charAt(0).toUpperCase();
                    const unreadDot = conv.unread_count > 0 ? '<div style="width: 10px; height: 10px; background: #4a9eff; border-radius: 50%; margin-right: 12px;"></div>' : '';
                    const preview = conv.last_message_preview || 'No messages';
                    const previewColor = conv.unread_count > 0 ? '#fff' : '#888';
                    const nameWeight = conv.unread_count > 0 ? '600' : '400';
                    const borderBottom = idx < conversations.length - 1 ? 'border-bottom: 1px solid #222;' : '';

                    return `
                        <div onclick="openConversation('${conv.contact_address}')" style="display: flex; align-items: center; padding: 12px 16px; cursor: pointer; transition: background 0.15s; ${borderBottom}" onmouseover="this.style.background='#1a1a1a'" onmouseout="this.style.background='transparent'">
                            ${unreadDot}
                            <div style="width: 44px; height: 44px; background: #4a9eff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 18px; margin-right: 12px; flex-shrink: 0;">${initial}</div>
                            <div style="flex: 1; min-width: 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                                    <span style="font-weight: ${nameWeight}; color: #fff; font-size: 15px;">${name}</span>
                                    <span style="color: #666; font-size: 13px;">${formatRelativeTime(conv.last_message_at)}</span>
                                </div>
                                <div style="color: ${previewColor}; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    ${preview}
                                </div>
                            </div>
                            <div style="color: #444; margin-left: 8px;">›</div>
                        </div>
                    `;
                }).join('');

                // Update unread badge
                const totalUnread = conversations.reduce((sum, c) => sum + (c.unread_count || 0), 0);
                const badge = document.getElementById('unread-badge');
                if (totalUnread > 0) {
                    badge.textContent = totalUnread;
                    badge.style.display = 'inline';
                } else {
                    badge.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to load conversations:', error);
            }
        }

        function formatRelativeTime(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMs / 3600000);
            const diffDays = Math.floor(diffMs / 86400000);

            if (diffMins < 1) return 'now';
            if (diffMins < 60) return `${diffMins}m`;
            if (diffHours < 24) return `${diffHours}h`;
            if (diffDays < 7) return `${diffDays}d`;
            return date.toLocaleDateString();
        }

        let currentThreadAutopilot = true;  // Default: autopilot on

        async function openConversation(contactAddress) {
            currentConversation = contactAddress;

            try {
                // Load messages and conversation info in parallel
                const [messagesResponse, convResponse, autopilotResponse] = await Promise.all([
                    fetch(`/api/conversations/${encodeURIComponent(contactAddress)}/messages`),
                    fetch('/api/conversations'),
                    fetch(`/api/conversations/${encodeURIComponent(contactAddress)}/autopilot`)
                ]);

                const messages = await messagesResponse.json();
                const conversations = await convResponse.json();
                const conv = conversations.find(c => c.contact_address === contactAddress);

                // Get autopilot status for this thread
                if (autopilotResponse.ok) {
                    const autopilotData = await autopilotResponse.json();
                    currentThreadAutopilot = autopilotData.enabled !== false;  // Default to true
                } else {
                    currentThreadAutopilot = true;
                }

                // Update avatar
                const name = conv?.display_name || contactAddress;
                const initial = name.charAt(0).toUpperCase();
                document.getElementById('conversation-avatar').textContent = initial;

                // Update header
                document.getElementById('conversation-title').textContent = name;
                const channels = conv?.channels || ['sms'];
                document.getElementById('conversation-channels').textContent = channels.map(ch => {
                    if (ch === 'sms') return 'SMS';
                    if (ch === 'email') return 'Email';
                    if (ch === 'call') return 'Call';
                    return ch;
                }).join(', ');

                // Update autopilot button
                updateAutopilotButton();

                // Render messages (iOS style bubbles)
                const container = document.getElementById('conversation-messages');
                if (!messages || messages.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: #666; padding: 40px;">No messages yet</div>';
                } else {
                    container.innerHTML = messages.map(msg => {
                        const isOutbound = msg.direction === 'outbound';
                        // iOS-style bubbles: green for outbound, gray for inbound
                        const bubbleColor = isOutbound ? '#34c759' : '#3a3a3c';
                        const bubbleRadius = isOutbound ? '18px 18px 4px 18px' : '18px 18px 18px 4px';
                        return `
                            <div style="display: flex; justify-content: ${isOutbound ? 'flex-end' : 'flex-start'}; margin-bottom: 6px;">
                                <div style="max-width: 75%; padding: 10px 14px; border-radius: ${bubbleRadius}; background: ${bubbleColor};">
                                    <div style="color: #fff; font-size: 15px; line-height: 1.4; word-wrap: break-word;">${msg.body || ''}</div>
                                    <div style="font-size: 11px; color: rgba(255,255,255,0.6); margin-top: 4px; text-align: right;">
                                        ${formatRelativeTime(msg.created_at)}
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('');

                    // Scroll to bottom
                    setTimeout(() => container.scrollTop = container.scrollHeight, 50);
                }

                // Mark as read
                await fetch(`/api/conversations/${encodeURIComponent(contactAddress)}/read`, { method: 'POST' });

                // Show modal
                document.getElementById('conversation-modal').classList.add('active');

                // Refresh conversation list to update unread counts
                loadConversations();
            } catch (error) {
                console.error('Failed to open conversation:', error);
            }
        }

        function updateAutopilotButton() {
            const btn = document.getElementById('autopilot-toggle-btn');
            if (currentThreadAutopilot) {
                btn.innerHTML = '🤖 On';
                btn.style.background = '#2d5a2d';
                btn.title = 'AI auto-reply is ON for this thread. Click to disable.';
            } else {
                btn.innerHTML = '🤖 Off';
                btn.style.background = '#5a2d2d';
                btn.title = 'AI auto-reply is OFF for this thread. Click to enable.';
            }
        }

        async function toggleThreadAutopilot() {
            if (!currentConversation) return;

            currentThreadAutopilot = !currentThreadAutopilot;
            updateAutopilotButton();

            // Save to server
            try {
                await fetch(`/api/conversations/${encodeURIComponent(currentConversation)}/autopilot`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: currentThreadAutopilot })
                });
            } catch (error) {
                console.error('Failed to update autopilot:', error);
            }
        }

        function closeConversationModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('conversation-modal').classList.remove('active');
                currentConversation = null;
            }
        }

        async function sendReply() {
            if (!currentConversation) return;

            const messageInput = document.getElementById('reply-message');
            const sendBtn = document.querySelector('#conversation-modal .btn-primary');
            const message = messageInput.value.trim();
            if (!message) return;

            // Disable button and show sending state
            sendBtn.disabled = true;
            sendBtn.textContent = 'Sending...';

            try {
                const response = await fetch('/api/sms/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        phone: currentConversation,
                        message: message
                    })
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to send SMS');
                }

                messageInput.value = '';

                // Refresh the conversation
                await openConversation(currentConversation);
            } catch (error) {
                alert('Failed to send message: ' + error.message);
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }

        // Poll conversations every 15 seconds
        setInterval(loadConversations, 15000);

        // Legacy function for compatibility
        async function loadSmsMessages() {
            // Now redirects to loadConversations
            await loadConversations();
        }

        // ========================================
        // UNIFIED INBOX (Enhanced Split-View)
        // ========================================

        let inboxFilters = { channel: '', direction: '', search: '' };
        let inboxSearchTimeout = null;
        let selectedInboxConversation = null;
        let selectedInboxAutopilot = true;

        function setInboxFilter(filterType, value) {
            inboxFilters[filterType] = value;

            // Update active state on buttons
            if (filterType === 'channel') {
                document.querySelectorAll('.inbox-filter-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.channel === value);
                });
            }

            loadInbox();
        }

        function debounceInboxSearch(value) {
            clearTimeout(inboxSearchTimeout);
            inboxSearchTimeout = setTimeout(() => {
                inboxFilters.search = value;
                loadInbox();
            }, 300);
        }

        async function loadInbox() {
            try {
                // Build query params
                const params = new URLSearchParams();
                if (inboxFilters.channel) params.append('channel', inboxFilters.channel);
                if (inboxFilters.direction) params.append('direction', inboxFilters.direction);
                if (inboxFilters.search) params.append('search', inboxFilters.search);

                const response = await fetch(`/api/inbox?${params}`);
                const data = await response.json();
                const conversations = data.conversations || [];

                const container = document.getElementById('inbox-conversation-list');

                if (conversations.length === 0) {
                    container.innerHTML = `
                        <div style="text-align: center; padding: 40px 20px; color: #666;">
                            <p style="font-size: 32px; margin-bottom: 12px;">📭</p>
                            <p>${inboxFilters.search ? 'No messages match your search' : 'No conversations yet'}</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = conversations.map(conv => {
                    const name = conv.display_name || conv.contact_address;
                    const initial = name.charAt(0).toUpperCase();
                    const isActive = selectedInboxConversation === conv.contact_address;
                    const unreadClass = conv.unread_count > 0 ? 'unread' : '';
                    const activeClass = isActive ? 'active' : '';

                    // Channel badges
                    const channels = conv.channels || [];
                    const badges = channels.map(ch => {
                        if (ch === 'sms') return '<span class="channel-badge sms">📱</span>';
                        if (ch === 'email') return '<span class="channel-badge email">📧</span>';
                        if (ch === 'call') return '<span class="channel-badge call">📞</span>';
                        return '';
                    }).join('');

                    return `
                        <div class="inbox-conv-item ${unreadClass} ${activeClass}" onclick="selectInboxConversation('${conv.contact_address}')">
                            <div style="display: flex; gap: 12px; align-items: flex-start;">
                                <div style="width: 40px; height: 40px; background: #4a9eff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0;">${initial}</div>
                                <div style="flex: 1; min-width: 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span class="conv-name" style="font-size: 14px;">${name}</span>
                                        <span style="font-size: 11px; color: #666;">${formatRelativeTime(conv.last_message_at)}</span>
                                    </div>
                                    <div style="display: flex; gap: 4px; margin: 4px 0;">${badges}</div>
                                    <div style="font-size: 13px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                        ${conv.last_message_preview || 'No messages'}
                                    </div>
                                    ${conv.unread_count > 0 ? `<span style="background: #dc3545; color: #fff; padding: 2px 6px; border-radius: 10px; font-size: 10px; margin-top: 4px; display: inline-block;">${conv.unread_count} new</span>` : ''}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');

                // Update unread badge
                const totalUnread = conversations.reduce((sum, c) => sum + (c.unread_count || 0), 0);
                const badge = document.getElementById('unread-badge');
                if (totalUnread > 0) {
                    badge.textContent = totalUnread;
                    badge.style.display = 'inline';
                } else {
                    badge.style.display = 'none';
                }

            } catch (error) {
                console.error('Failed to load inbox:', error);
            }

            // Also load autopilot queue
            loadAutopilotQueue();
        }

        async function selectInboxConversation(contactAddress) {
            selectedInboxConversation = contactAddress;

            // Update active state in list
            document.querySelectorAll('.inbox-conv-item').forEach(item => {
                item.classList.toggle('active', item.onclick.toString().includes(contactAddress));
            });

            // Show thread UI elements
            document.getElementById('inbox-thread-empty').style.display = 'none';
            document.getElementById('inbox-thread-header').style.display = 'block';
            document.getElementById('inbox-thread-messages').style.display = 'block';
            document.getElementById('inbox-thread-input').style.display = 'block';

            try {
                // Load messages and autopilot status
                const [messagesResponse, autopilotResponse] = await Promise.all([
                    fetch(`/api/inbox/${encodeURIComponent(contactAddress)}/messages`),
                    fetch(`/api/conversations/${encodeURIComponent(contactAddress)}/autopilot`)
                ]);

                const messages = await messagesResponse.json();

                // Get autopilot status
                if (autopilotResponse.ok) {
                    const autopilotData = await autopilotResponse.json();
                    selectedInboxAutopilot = autopilotData.enabled !== false;
                } else {
                    selectedInboxAutopilot = true;
                }

                // Get display name from first message or contact
                const displayName = messages.length > 0 ?
                    (messages[0].first_name ? `${messages[0].first_name} ${messages[0].last_name || ''}`.trim() : contactAddress) :
                    contactAddress;
                const initial = displayName.charAt(0).toUpperCase();

                // Update header
                document.getElementById('inbox-thread-avatar').textContent = initial;
                document.getElementById('inbox-thread-title').textContent = displayName;

                // Channel badges in header
                const channels = [...new Set(messages.map(m => m.channel))];
                document.getElementById('inbox-thread-channels').innerHTML = channels.map(ch => {
                    if (ch === 'sms') return '<span class="channel-badge sms">📱 SMS</span>';
                    if (ch === 'email') return '<span class="channel-badge email">📧 Email</span>';
                    if (ch === 'call') return '<span class="channel-badge call">📞 Call</span>';
                    return '';
                }).join('');

                // Update autopilot button
                updateInboxAutopilotButton();

                // Render messages
                const container = document.getElementById('inbox-thread-messages');
                if (messages.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: #666; padding: 40px;">No messages yet</div>';
                } else {
                    container.innerHTML = `<div style="display: flex; flex-direction: column; gap: 8px;">` +
                        messages.map(msg => {
                            const isOutbound = msg.direction === 'outbound';
                            const isAI = msg.is_ai || msg.ai_generated;
                            const aiClass = isAI ? 'ai-generated' : '';

                            // Channel icon
                            let channelIcon = '';
                            if (msg.channel === 'email') channelIcon = '📧 ';
                            else if (msg.channel === 'call') channelIcon = '📞 ';

                            // For calls, show summary/transcript
                            let content = msg.body || '';
                            if (msg.channel === 'call') {
                                content = msg.ai_summary || msg.body || 'Call transcript available';
                                if (msg.call_duration) {
                                    content = `Duration: ${Math.round(msg.call_duration / 60)}min<br>${content}`;
                                }
                            }

                            return `
                                <div class="msg-bubble ${isOutbound ? 'outbound' : 'inbound'} ${aiClass}" style="align-self: ${isOutbound ? 'flex-end' : 'flex-start'};">
                                    ${channelIcon}${content}
                                    <div class="msg-meta">${formatRelativeTime(msg.created_at)}</div>
                                </div>
                            `;
                        }).join('') +
                    `</div>`;

                    // Scroll to bottom
                    setTimeout(() => container.scrollTop = container.scrollHeight, 50);
                }

                // Mark as read
                await fetch(`/api/conversations/${encodeURIComponent(contactAddress)}/read`, { method: 'POST' });

                // Refresh conversation list
                loadInbox();

            } catch (error) {
                console.error('Failed to load conversation:', error);
            }
        }

        function updateInboxAutopilotButton() {
            const btn = document.getElementById('inbox-autopilot-toggle');
            if (selectedInboxAutopilot) {
                btn.innerHTML = '🤖 On';
                btn.style.background = '#2d5a2d';
                btn.title = 'AI auto-reply is ON. Click to disable.';
            } else {
                btn.innerHTML = '🤖 Off';
                btn.style.background = '#5a2d2d';
                btn.title = 'AI auto-reply is OFF. Click to enable.';
            }
        }

        async function toggleThreadAutopilotInline() {
            if (!selectedInboxConversation) return;

            selectedInboxAutopilot = !selectedInboxAutopilot;
            updateInboxAutopilotButton();

            try {
                await fetch(`/api/conversations/${encodeURIComponent(selectedInboxConversation)}/autopilot`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: selectedInboxAutopilot })
                });
            } catch (error) {
                console.error('Failed to update autopilot:', error);
            }
        }

        async function sendInboxReply() {
            if (!selectedInboxConversation) return;

            const messageInput = document.getElementById('inbox-reply-message');
            const message = messageInput.value.trim();
            if (!message) return;

            try {
                const response = await fetch('/api/sms/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        phone: selectedInboxConversation,
                        message: message
                    })
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to send');
                }

                messageInput.value = '';
                messageInput.style.height = 'auto';

                // Refresh the conversation
                selectInboxConversation(selectedInboxConversation);

            } catch (error) {
                alert('Failed to send: ' + error.message);
            }
        }

        // ========================================
        // AUTOPILOT QUEUE
        // ========================================

        let autopilotQueueHidden = false;

        async function loadAutopilotQueue() {
            try {
                const response = await fetch('/api/autopilot/queue');
                const data = await response.json();
                const pending = data.pending || [];

                const container = document.getElementById('autopilot-queue-container');
                const list = document.getElementById('autopilot-queue-list');
                const count = document.getElementById('autopilot-queue-count');

                if (pending.length === 0 || autopilotQueueHidden) {
                    container.style.display = 'none';
                    return;
                }

                container.style.display = 'block';
                count.textContent = pending.length;

                list.innerHTML = pending.map(item => `
                    <div class="autopilot-pending-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div>
                                <strong>${item.display_name || item.contact_address}</strong>
                                <span style="color: #888; font-size: 12px; margin-left: 8px;">${item.channel || 'sms'}</span>
                            </div>
                            <span style="color: #888; font-size: 11px;">${formatRelativeTime(item.created_at)}</span>
                        </div>
                        <div style="background: #0a0a0a; padding: 10px; border-radius: 6px; margin-bottom: 8px; font-size: 14px; color: #ccc;">
                            "${item.proposed_message}"
                        </div>
                        <div style="display: flex; gap: 8px; justify-content: flex-end;">
                            <button class="btn btn-secondary" onclick="cancelAutopilotResponse(${item.id})" style="padding: 6px 12px; font-size: 12px;">Cancel</button>
                            <button class="btn btn-secondary" onclick="editAutopilotResponse(${item.id})" style="padding: 6px 12px; font-size: 12px;">Edit</button>
                            <button class="btn btn-primary" onclick="approveAutopilotResponse(${item.id})" style="padding: 6px 12px; font-size: 12px;">Send Now</button>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Failed to load autopilot queue:', error);
            }
        }

        function toggleAutopilotQueue() {
            autopilotQueueHidden = !autopilotQueueHidden;
            document.getElementById('autopilot-queue-container').style.display = autopilotQueueHidden ? 'none' : 'block';
        }

        async function approveAutopilotResponse(queueId) {
            try {
                const response = await fetch(`/api/autopilot/queue/${queueId}/approve`, { method: 'POST' });
                if (!response.ok) throw new Error('Failed to approve');
                loadAutopilotQueue();
                loadInbox();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function cancelAutopilotResponse(queueId) {
            try {
                const response = await fetch(`/api/autopilot/queue/${queueId}/cancel`, { method: 'POST' });
                if (!response.ok) throw new Error('Failed to cancel');
                loadAutopilotQueue();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function editAutopilotResponse(queueId) {
            const newMessage = prompt('Edit the AI response:');
            if (!newMessage) return;

            try {
                const response = await fetch(`/api/autopilot/queue/${queueId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ proposed_message: newMessage })
                });
                if (!response.ok) throw new Error('Failed to update');
                loadAutopilotQueue();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        // Poll unified inbox and autopilot queue
        setInterval(loadInbox, 15000);

        function showSendSmsModal() {
            document.getElementById('sms-to').value = '';
            document.getElementById('sms-message').value = '';
            document.getElementById('sms-char-count').textContent = '0';
            document.getElementById('sms-modal').classList.add('active');
        }

        function closeSmsModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('sms-modal').classList.remove('active');
            }
        }

        // Character counter for SMS
        document.addEventListener('DOMContentLoaded', function() {
            const smsMessage = document.getElementById('sms-message');
            if (smsMessage) {
                smsMessage.addEventListener('input', function() {
                    document.getElementById('sms-char-count').textContent = this.value.length;
                });
            }
        });

        async function sendSms() {
            const to = document.getElementById('sms-to').value.trim();
            const message = document.getElementById('sms-message').value.trim();

            if (!to || !message) {
                alert('Phone number and message are required');
                return;
            }

            try {
                const response = await fetch('/api/sms/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phone: to, message: message })
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to send');
                }

                closeSmsModal();
                loadSmsMessages();
                alert('SMS sent successfully!');
            } catch (error) {
                alert('Error sending SMS: ' + error.message);
            }
        }

        // ========================================
        // KNOWLEDGE BASE MANAGEMENT
        // ========================================

        async function loadKnowledgeBases() {
            try {
                const response = await fetch('/api/knowledge');
                const bases = await response.json();

                const container = document.getElementById('kb-list');

                if (!bases || bases.length === 0) {
                    container.innerHTML = `
                        <div class="card" style="text-align: center; padding: 40px; color: #666;">
                            <p>No knowledge bases yet.</p>
                            <p style="margin-top: 8px;">Create one to store products, services, policies, and other information the AI can reference during calls.</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = bases.map(kb => `
                    <div class="card" style="margin-bottom: 12px; cursor: pointer;" onclick="openKbDetail('${kb.id}')">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin-bottom: 4px;">${kb.name}</h4>
                                <p style="color: #888; font-size: 14px; margin-bottom: 4px;">${kb.description || 'No description'}</p>
                                <span style="display: inline-block; padding: 2px 8px; background: #333; border-radius: 4px; font-size: 12px; color: #888;">${kb.category}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 24px; font-weight: bold;">${kb.document_count || 0}</div>
                                <div style="color: #888; font-size: 12px;">documents</div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load knowledge bases:', error);
            }
        }

        function showAddKbModal() {
            document.getElementById('kb-modal-title').textContent = 'New Knowledge Base';
            document.getElementById('kb-id').value = '';
            document.getElementById('kb-name').value = '';
            document.getElementById('kb-description').value = '';
            document.getElementById('kb-category').value = 'general';
            document.getElementById('kb-objective').value = '';
            document.getElementById('kb-modal').classList.add('active');
        }

        function closeKbModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('kb-modal').classList.remove('active');
            }
        }

        async function saveKb() {
            const kbId = document.getElementById('kb-id').value;
            const objectiveStr = document.getElementById('kb-objective').value.trim();
            const data = {
                name: document.getElementById('kb-name').value,
                description: document.getElementById('kb-description').value,
                category: document.getElementById('kb-category').value,
                objective_keywords: objectiveStr ? objectiveStr.split(',').map(k => k.trim().toLowerCase()).filter(k => k) : []
            };

            if (!data.name) {
                alert('Name is required');
                return;
            }

            try {
                const url = kbId ? `/api/knowledge/${kbId}` : '/api/knowledge';
                const method = kbId ? 'PUT' : 'POST';

                const response = await fetch(url, {
                    method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to save');
                }

                closeKbModal();
                loadKnowledgeBases();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function openKbDetail(kbId) {
            try {
                const response = await fetch(`/api/knowledge/${kbId}`);
                const kb = await response.json();

                document.getElementById('kb-detail-id').value = kbId;
                document.getElementById('kb-detail-title').textContent = kb.name;

                // Load documents
                const docsResponse = await fetch(`/api/knowledge/${kbId}/documents`);
                const docs = await docsResponse.json();

                const container = document.getElementById('kb-documents');
                if (!docs || docs.length === 0) {
                    container.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No documents yet. Add your first document.</p>';
                } else {
                    container.innerHTML = docs.map(doc => `
                        <div style="background: #222; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h5 style="margin-bottom: 4px;">${doc.title}</h5>
                                    <span style="display: inline-block; padding: 2px 6px; background: #333; border-radius: 4px; font-size: 11px; color: #888;">${doc.type}</span>
                                </div>
                                <button class="btn btn-danger" style="padding: 4px 8px; font-size: 12px;" onclick="deleteDoc('${kbId}', '${doc.id}')">Delete</button>
                            </div>
                            <p style="color: #888; font-size: 13px; margin-top: 8px;">${doc.content_preview}</p>
                        </div>
                    `).join('');
                }

                document.getElementById('kb-detail-modal').classList.add('active');
            } catch (error) {
                alert('Failed to load knowledge base: ' + error);
            }
        }

        function closeKbDetailModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('kb-detail-modal').classList.remove('active');
            }
        }

        async function deleteCurrentKb() {
            const kbId = document.getElementById('kb-detail-id').value;
            if (!confirm('Delete this knowledge base and all its documents?')) return;

            try {
                await fetch(`/api/knowledge/${kbId}`, { method: 'DELETE' });
                closeKbDetailModal();
                loadKnowledgeBases();
            } catch (error) {
                alert('Failed to delete: ' + error);
            }
        }

        function showAddDocModal() {
            const kbId = document.getElementById('kb-detail-id').value;
            document.getElementById('doc-kb-id').value = kbId;
            document.getElementById('doc-title').value = '';
            document.getElementById('doc-type').value = 'text';
            document.getElementById('doc-content').value = '';
            document.getElementById('doc-tags').value = '';
            document.getElementById('doc-modal').classList.add('active');
        }

        function closeDocModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('doc-modal').classList.remove('active');
            }
        }

        async function saveDoc() {
            const kbId = document.getElementById('doc-kb-id').value;
            const data = {
                title: document.getElementById('doc-title').value,
                content: document.getElementById('doc-content').value,
                doc_type: document.getElementById('doc-type').value,
                tags: document.getElementById('doc-tags').value.split(',').map(t => t.trim()).filter(t => t)
            };

            if (!data.title || !data.content) {
                alert('Title and content are required');
                return;
            }

            try {
                const response = await fetch(`/api/knowledge/${kbId}/documents`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to save');
                }

                closeDocModal();
                openKbDetail(kbId);  // Refresh documents
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function deleteDoc(kbId, docId) {
            if (!confirm('Delete this document?')) return;

            try {
                await fetch(`/api/knowledge/${kbId}/documents/${docId}`, { method: 'DELETE' });
                openKbDetail(kbId);  // Refresh
            } catch (error) {
                alert('Failed to delete: ' + error);
            }
        }

        // ========================================
        // LEADS MANAGEMENT
        // ========================================

        let leadsCurrentPage = 1;
        let leadsPageSize = 20;
        let searchTimeout = null;
        let csvData = null;
        let csvHeaders = [];

        function debounceSearch() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(loadLeads, 300);
        }

        async function loadLeads() {
            const search = document.getElementById('leads-search').value;
            const status = document.getElementById('leads-status-filter').value;

            let url = `/api/leads?page=${leadsCurrentPage}&page_size=${leadsPageSize}`;
            if (search) url += `&search=${encodeURIComponent(search)}`;
            if (status) url += `&status=${encodeURIComponent(status)}`;

            try {
                const response = await fetch(url);
                const data = await response.json();

                renderLeadsTable(data.leads);
                renderPagination(data.total, data.page, data.page_size);
            } catch (error) {
                console.error('Failed to load leads:', error);
            }
        }

        async function loadLeadStats() {
            try {
                const response = await fetch('/api/leads/stats');
                const stats = await response.json();

                document.getElementById('stat-total').textContent = stats.total || 0;
                document.getElementById('stat-new').textContent = stats.new || 0;
                document.getElementById('stat-engaged').textContent = stats.engaged || 0;
                document.getElementById('stat-booked').textContent = stats.booked || 0;
            } catch (error) {
                console.error('Failed to load lead stats:', error);
            }
        }

        let selectedLeadIds = new Set();

        function renderLeadsTable(leads) {
            const tbody = document.getElementById('leads-table-body');

            if (!leads || leads.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="padding: 40px; text-align: center; color: #666;">No leads found</td></tr>';
                return;
            }

            tbody.innerHTML = leads.map(lead => `
                <tr style="border-bottom: 1px solid #222;" data-lead-id="${lead.id}">
                    <td style="padding: 12px;" onclick="event.stopPropagation();">
                        <input type="checkbox" class="lead-checkbox" value="${lead.id}"
                               ${selectedLeadIds.has(lead.id) ? 'checked' : ''}
                               onchange="toggleLeadSelection(${lead.id}, this.checked)"
                               style="width: 18px; height: 18px; cursor: pointer;">
                    </td>
                    <td style="padding: 12px; cursor: pointer;" onclick="showLeadDetails(${lead.id})">${lead.first_name || ''} ${lead.last_name || ''}</td>
                    <td style="padding: 12px; color: #888; cursor: pointer;" onclick="showLeadDetails(${lead.id})">${lead.company || '-'}</td>
                    <td style="padding: 12px; cursor: pointer;" onclick="showLeadDetails(${lead.id})">${lead.phone || '-'}</td>
                    <td style="padding: 12px; cursor: pointer;" onclick="showLeadDetails(${lead.id})">
                        <span style="padding: 4px 8px; border-radius: 12px; font-size: 11px; background: ${getStatusColor(lead.status)}; color: #fff;">
                            ${lead.status || 'NEW'}
                        </span>
                    </td>
                    <td style="padding: 12px; color: #888; cursor: pointer;" onclick="showLeadDetails(${lead.id})">${formatDate(lead.last_contacted_at)}</td>
                    <td style="padding: 12px;">
                        <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="event.stopPropagation(); editLead(${lead.id})">Edit</button>
                        <button class="btn" style="padding: 6px 12px; font-size: 12px; background: #2d5a2d;" onclick="event.stopPropagation(); callLead(${lead.id}, '${lead.phone}')">Call</button>
                        <button class="btn" style="padding: 6px 12px; font-size: 12px; background: #dc3545;" onclick="event.stopPropagation(); deleteLead(${lead.id})">Delete</button>
                    </td>
                </tr>
            `).join('');

            updateBulkActionsBar();
        }

        function toggleLeadSelection(leadId, isSelected) {
            if (isSelected) {
                selectedLeadIds.add(leadId);
            } else {
                selectedLeadIds.delete(leadId);
            }
            updateBulkActionsBar();
        }

        function toggleSelectAll(checkbox) {
            const checkboxes = document.querySelectorAll('.lead-checkbox');
            checkboxes.forEach(cb => {
                cb.checked = checkbox.checked;
                const leadId = parseInt(cb.value);
                if (checkbox.checked) {
                    selectedLeadIds.add(leadId);
                } else {
                    selectedLeadIds.delete(leadId);
                }
            });
            updateBulkActionsBar();
        }

        function updateBulkActionsBar() {
            const bar = document.getElementById('bulk-actions-bar');
            const countSpan = document.getElementById('selected-count');
            const selectAllCheckbox = document.getElementById('select-all-leads');

            if (selectedLeadIds.size > 0) {
                bar.style.display = 'block';
                countSpan.textContent = `${selectedLeadIds.size} selected`;
            } else {
                bar.style.display = 'none';
            }

            // Update select-all checkbox state
            const checkboxes = document.querySelectorAll('.lead-checkbox');
            if (checkboxes.length > 0 && selectedLeadIds.size === checkboxes.length) {
                selectAllCheckbox.checked = true;
                selectAllCheckbox.indeterminate = false;
            } else if (selectedLeadIds.size > 0) {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = true;
            } else {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = false;
            }
        }

        function clearSelection() {
            selectedLeadIds.clear();
            document.querySelectorAll('.lead-checkbox').forEach(cb => cb.checked = false);
            document.getElementById('select-all-leads').checked = false;
            updateBulkActionsBar();
        }

        async function deleteLead(leadId) {
            if (!confirm('Are you sure you want to delete this lead?')) return;

            try {
                const response = await fetch(`/api/leads/${leadId}`, { method: 'DELETE' });
                if (response.ok) {
                    selectedLeadIds.delete(leadId);
                    loadLeads();
                    loadLeadStats();
                } else {
                    alert('Failed to delete lead');
                }
            } catch (error) {
                console.error('Error deleting lead:', error);
                alert('Error deleting lead');
            }
        }

        async function deleteSelectedLeads() {
            if (selectedLeadIds.size === 0) return;

            if (!confirm(`Are you sure you want to delete ${selectedLeadIds.size} lead(s)?`)) return;

            try {
                const response = await fetch('/api/leads/bulk-delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lead_ids: Array.from(selectedLeadIds) })
                });

                if (response.ok) {
                    const data = await response.json();
                    selectedLeadIds.clear();
                    loadLeads();
                    loadLeadStats();
                    alert(`Deleted ${data.deleted_count} lead(s)`);
                } else {
                    alert('Failed to delete leads');
                }
            } catch (error) {
                console.error('Error deleting leads:', error);
                alert('Error deleting leads');
            }
        }

        // Lead Lists Functions
        let leadLists = [];

        async function loadLeadLists() {
            try {
                const response = await fetch('/api/lead-lists');
                leadLists = await response.json();
                return leadLists;
            } catch (error) {
                console.error('Error loading lists:', error);
                return [];
            }
        }

        function showListsModal() {
            document.getElementById('lists-modal').classList.add('active');
            renderListsModal();
        }

        function closeListsModal(event) {
            if (!event || event.target.classList.contains('modal-overlay')) {
                document.getElementById('lists-modal').classList.remove('active');
            }
        }

        async function renderListsModal() {
            const container = document.getElementById('lists-container');
            container.innerHTML = '<p style="color: #666; text-align: center;">Loading...</p>';

            const lists = await loadLeadLists();

            if (lists.length === 0) {
                container.innerHTML = '<p style="color: #666; text-align: center;">No lists yet. Create one above!</p>';
                return;
            }

            container.innerHTML = `
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333;">
                            <th style="padding: 8px; text-align: left; color: #888;">Name</th>
                            <th style="padding: 8px; text-align: left; color: #888;">Leads</th>
                            <th style="padding: 8px; text-align: left; color: #888;">Created</th>
                            <th style="padding: 8px; text-align: right; color: #888;">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${lists.map(list => `
                            <tr style="border-bottom: 1px solid #222;">
                                <td style="padding: 12px;">${list.name}</td>
                                <td style="padding: 12px; color: #888;">${list.lead_count || 0}</td>
                                <td style="padding: 12px; color: #888;">${formatDate(list.created_at)}</td>
                                <td style="padding: 12px; text-align: right;">
                                    <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="viewListLeads(${list.id}, '${list.name}')">View</button>
                                    <button class="btn" style="padding: 4px 8px; font-size: 11px; background: #dc3545;" onclick="deleteList(${list.id})">Delete</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        async function createNewList() {
            const name = document.getElementById('new-list-name').value.trim();
            if (!name) {
                alert('Please enter a list name');
                return;
            }

            try {
                const response = await fetch('/api/lead-lists', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });

                if (response.ok) {
                    document.getElementById('new-list-name').value = '';
                    renderListsModal();
                } else {
                    alert('Failed to create list');
                }
            } catch (error) {
                console.error('Error creating list:', error);
                alert('Error creating list');
            }
        }

        async function deleteList(listId) {
            if (!confirm('Are you sure you want to delete this list?')) return;

            try {
                const response = await fetch(`/api/lead-lists/${listId}`, { method: 'DELETE' });
                if (response.ok) {
                    renderListsModal();
                } else {
                    alert('Failed to delete list');
                }
            } catch (error) {
                console.error('Error deleting list:', error);
                alert('Error deleting list');
            }
        }

        async function viewListLeads(listId, listName) {
            // Filter leads by list - for now just show in a simple alert
            // Could extend to filter the main table
            try {
                const response = await fetch(`/api/lead-lists/${listId}/leads`);
                const leads = await response.json();
                alert(`List "${listName}" has ${leads.length} lead(s)`);
            } catch (error) {
                console.error('Error viewing list:', error);
            }
        }

        function addSelectedToList() {
            if (selectedLeadIds.size === 0) {
                alert('No leads selected');
                return;
            }
            document.getElementById('add-to-list-count').textContent = selectedLeadIds.size;
            document.getElementById('add-to-list-modal').classList.add('active');
            renderAddToListOptions();
        }

        function closeAddToListModal(event) {
            if (!event || event.target.classList.contains('modal-overlay')) {
                document.getElementById('add-to-list-modal').classList.remove('active');
            }
        }

        async function renderAddToListOptions() {
            const container = document.getElementById('list-options-container');
            container.innerHTML = '<p style="color: #666; text-align: center;">Loading...</p>';

            const lists = await loadLeadLists();

            if (lists.length === 0) {
                container.innerHTML = '<p style="color: #666; text-align: center;">No lists yet. Create one in the Lists manager first.</p>';
                return;
            }

            container.innerHTML = lists.map(list => `
                <button class="btn btn-secondary" style="width: 100%; margin-bottom: 8px; text-align: left;"
                        onclick="addLeadsToList(${list.id}, '${list.name}')">
                    ${list.name} <span style="color: #888;">(${list.lead_count || 0} leads)</span>
                </button>
            `).join('');
        }

        async function addLeadsToList(listId, listName) {
            try {
                const response = await fetch(`/api/lead-lists/${listId}/leads`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lead_ids: Array.from(selectedLeadIds) })
                });

                if (response.ok) {
                    closeAddToListModal();
                    alert(`Added ${selectedLeadIds.size} lead(s) to "${listName}"`);
                } else {
                    alert('Failed to add leads to list');
                }
            } catch (error) {
                console.error('Error adding leads to list:', error);
                alert('Error adding leads to list');
            }
        }

        function getStatusColor(status) {
            const colors = {
                'NEW': '#4a9eff',
                'CONTACTED': '#888',
                'ENGAGED': '#ffc107',
                'QUALIFIED': '#9c27b0',
                'MEETING_BOOKED': '#28a745',
                'WON': '#28a745',
                'LOST': '#dc3545'
            };
            return colors[status] || '#666';
        }

        function formatDate(dateStr) {
            if (!dateStr) return '-';
            const date = new Date(dateStr);
            const now = new Date();
            const diff = now - date;
            const hours = Math.floor(diff / 3600000);
            const days = Math.floor(diff / 86400000);

            if (hours < 1) return 'Just now';
            if (hours < 24) return `${hours}h ago`;
            if (days < 7) return `${days}d ago`;
            return date.toLocaleDateString();
        }

        function renderPagination(total, page, pageSize) {
            const totalPages = Math.ceil(total / pageSize);
            const container = document.getElementById('leads-pagination');

            if (totalPages <= 1) {
                container.innerHTML = '';
                return;
            }

            let html = '';
            if (page > 1) {
                html += `<button class="btn btn-secondary" style="padding: 8px 12px;" onclick="goToPage(${page - 1})">Prev</button>`;
            }
            html += `<span style="padding: 8px 16px; color: #888;">Page ${page} of ${totalPages}</span>`;
            if (page < totalPages) {
                html += `<button class="btn btn-secondary" style="padding: 8px 12px;" onclick="goToPage(${page + 1})">Next</button>`;
            }
            container.innerHTML = html;
        }

        function goToPage(page) {
            leadsCurrentPage = page;
            loadLeads();
        }

        // Lead Tab Switching
        function showLeadTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.lead-tab').forEach(tab => tab.style.display = 'none');
            // Remove active from all buttons
            document.querySelectorAll('.lead-tab-btn').forEach(btn => btn.classList.remove('active'));
            // Show selected tab
            document.getElementById('lead-tab-' + tabName).style.display = 'block';
            // Mark button as active
            document.querySelector('.lead-tab-btn[data-tab="' + tabName + '"]').classList.add('active');
        }

        // All lead field IDs for clearing/populating
        const leadFields = [
            'lead-first-name', 'lead-last-name', 'lead-email', 'lead-phone',
            'lead-phone-type', 'lead-linkedin-url', 'lead-status', 'lead-sentiment',
            'lead-company', 'lead-title', 'lead-industry', 'lead-website',
            'lead-company-linkedin', 'lead-company-size', 'lead-revenue',
            'lead-seniority', 'lead-department', 'lead-funding-stage', 'lead-technologies',
            'lead-address', 'lead-city', 'lead-state', 'lead-country', 'lead-timezone',
            'lead-icebreaker', 'lead-trigger-event', 'lead-pain-points', 'lead-notes',
            'lead-custom-1', 'lead-custom-2', 'lead-custom-3', 'lead-custom-4', 'lead-custom-5',
            'lead-source'
        ];

        // Map HTML IDs to database field names
        const fieldIdToDb = {
            'lead-first-name': 'first_name',
            'lead-last-name': 'last_name',
            'lead-email': 'email',
            'lead-phone': 'phone',
            'lead-phone-type': 'phone_type',
            'lead-linkedin-url': 'linkedin_url',
            'lead-status': 'status',
            'lead-sentiment': 'sentiment_status',
            'lead-company': 'company',
            'lead-title': 'title',
            'lead-industry': 'industry',
            'lead-website': 'website',
            'lead-company-linkedin': 'company_linkedin_url',
            'lead-company-size': 'company_size',
            'lead-revenue': 'revenue',
            'lead-seniority': 'seniority',
            'lead-department': 'department',
            'lead-funding-stage': 'funding_stage',
            'lead-technologies': 'technologies',
            'lead-address': 'address',
            'lead-city': 'city',
            'lead-state': 'state',
            'lead-country': 'country',
            'lead-timezone': 'timezone',
            'lead-icebreaker': 'icebreaker',
            'lead-trigger-event': 'trigger_event',
            'lead-pain-points': 'pain_points',
            'lead-notes': 'notes',
            'lead-custom-1': 'custom_1',
            'lead-custom-2': 'custom_2',
            'lead-custom-3': 'custom_3',
            'lead-custom-4': 'custom_4',
            'lead-custom-5': 'custom_5',
            'lead-source': 'source'
        };

        // Add/Edit Lead Modal
        function showAddLeadModal() {
            document.getElementById('lead-modal-title').textContent = 'Add Lead';
            document.getElementById('lead-id').value = '';
            // Clear all fields
            leadFields.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.value = '';
            });
            // Set defaults
            document.getElementById('lead-status').value = 'NEW';
            // Reset to first tab
            showLeadTab('basic');
            document.getElementById('lead-modal').classList.add('active');
        }

        async function editLead(leadId) {
            try {
                const response = await fetch(`/api/leads/${leadId}`);
                const lead = await response.json();

                document.getElementById('lead-modal-title').textContent = 'Edit Lead';
                document.getElementById('lead-id').value = lead.id;

                // Populate all fields
                leadFields.forEach(id => {
                    const el = document.getElementById(id);
                    const dbField = fieldIdToDb[id];
                    if (el && dbField) el.value = lead[dbField] || '';
                });

                // Reset to first tab
                showLeadTab('basic');
                document.getElementById('lead-modal').classList.add('active');
            } catch (error) {
                alert('Failed to load lead: ' + error);
            }
        }

        function closeLeadModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('lead-modal').classList.remove('active');
            }
        }

        async function saveLead() {
            const leadId = document.getElementById('lead-id').value;

            // Collect all fields
            const lead = {};
            leadFields.forEach(id => {
                const el = document.getElementById(id);
                const dbField = fieldIdToDb[id];
                if (el && dbField && el.value) {
                    lead[dbField] = el.value;
                }
            });

            if (!lead.phone && !lead.email) {
                alert('Either phone number or email is required');
                showLeadTab('basic');
                return;
            }

            try {
                const url = leadId ? `/api/leads/${leadId}` : '/api/leads';
                const method = leadId ? 'PUT' : 'POST';

                const response = await fetch(url, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(lead)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to save lead');
                }

                closeLeadModal();
                loadLeads();
                loadLeadStats();
            } catch (error) {
                alert('Failed to save lead: ' + error);
            }
        }

        // Lead Detail Modal
        async function showLeadDetails(leadId) {
            try {
                const [leadResponse, interactionsResponse] = await Promise.all([
                    fetch(`/api/leads/${leadId}`),
                    fetch(`/api/leads/${leadId}/interactions`)
                ]);

                const lead = await leadResponse.json();
                const interactions = await interactionsResponse.json();

                document.getElementById('lead-detail-title').textContent =
                    `${lead.first_name || ''} ${lead.last_name || ''}`.trim() || 'Lead Details';

                // Build lead info HTML
                let infoHtml = `
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                        <div><span style="color: #888; font-size: 12px;">Email</span><br>${lead.email || '-'}</div>
                        <div><span style="color: #888; font-size: 12px;">Phone</span><br>${lead.phone || '-'}</div>
                        <div><span style="color: #888; font-size: 12px;">Company</span><br>${lead.company || '-'}</div>
                        <div><span style="color: #888; font-size: 12px;">Title</span><br>${lead.title || '-'}</div>
                        <div><span style="color: #888; font-size: 12px;">Industry</span><br>${lead.industry || '-'}</div>
                        <div><span style="color: #888; font-size: 12px;">Status</span><br>
                            <span style="padding: 4px 8px; border-radius: 12px; font-size: 11px; background: ${getStatusColor(lead.status)}; color: #fff;">
                                ${lead.status || 'NEW'}
                            </span>
                        </div>
                    </div>
                `;
                if (lead.notes) {
                    infoHtml += `<div style="margin-top: 16px;"><span style="color: #888; font-size: 12px;">Notes</span><br>${lead.notes}</div>`;
                }
                document.getElementById('lead-detail-content').innerHTML = infoHtml;

                // Build interactions HTML
                const interactionsContainer = document.getElementById('lead-interactions');
                if (!interactions || interactions.length === 0) {
                    interactionsContainer.innerHTML = '<p style="color: #666;">No interactions yet</p>';
                } else {
                    interactionsContainer.innerHTML = interactions.map(i => `
                        <div style="padding: 12px; background: #0a0a0a; border-radius: 8px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #4a9eff;">${i.channel || 'call'}</span>
                                <span style="color: #888; font-size: 12px;">${formatDate(i.created_at)}</span>
                            </div>
                            ${i.summary ? `<div style="color: #ccc;">${i.summary}</div>` : ''}
                            ${i.outcome ? `<div style="margin-top: 8px; color: #888; font-size: 12px;">Outcome: ${i.outcome}</div>` : ''}
                        </div>
                    `).join('');
                }

                document.getElementById('lead-detail-modal').classList.add('active');
            } catch (error) {
                alert('Failed to load lead details: ' + error);
            }
        }

        function closeLeadDetailModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('lead-detail-modal').classList.remove('active');
            }
        }

        // Call a lead
        function callLead(leadId, phone) {
            if (!phone) {
                alert('No phone number for this lead');
                return;
            }
            // Switch to calls tab and populate phone
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.tab-btn').classList.add('active');
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById('tab-calls').classList.add('active');
            document.getElementById('phone').value = phone;
            document.getElementById('objective').focus();
        }

        // CSV Import
        function handleCsvImport(input) {
            const file = input.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                const text = e.target.result;
                parseCsv(text);
                showCsvModal();
            };
            reader.readAsText(file);
            input.value = ''; // Reset input
        }

        function parseCsv(text) {
            const lines = text.split('\\n').filter(line => line.trim());
            if (lines.length < 2) {
                alert('CSV must have headers and at least one data row');
                return;
            }

            // Parse headers (first line)
            csvHeaders = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

            // Parse data rows
            csvData = [];
            for (let i = 1; i < lines.length; i++) {
                const values = parseCSVLine(lines[i]);
                if (values.length === csvHeaders.length) {
                    const row = {};
                    csvHeaders.forEach((h, idx) => row[h] = values[idx]);
                    csvData.push(row);
                }
            }
        }

        function parseCSVLine(line) {
            const result = [];
            let current = '';
            let inQuotes = false;

            for (let char of line) {
                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === ',' && !inQuotes) {
                    result.push(current.trim());
                    current = '';
                } else {
                    current += char;
                }
            }
            result.push(current.trim());
            return result;
        }

        function showCsvModal() {
            const dbFields = [
                { value: '', label: '-- Skip --' },
                // Contact
                { value: 'first_name', label: 'First Name' },
                { value: 'last_name', label: 'Last Name' },
                { value: 'email', label: 'Email' },
                { value: 'phone', label: 'Phone *' },
                { value: 'phone_type', label: 'Phone Type' },
                { value: 'linkedin_url', label: 'LinkedIn URL' },
                // Company
                { value: 'company', label: 'Company' },
                { value: 'title', label: 'Title' },
                { value: 'industry', label: 'Industry' },
                { value: 'website', label: 'Website' },
                { value: 'company_linkedin_url', label: 'Company LinkedIn' },
                { value: 'company_size', label: 'Company Size' },
                { value: 'employee_count', label: 'Employee Count' },
                { value: 'revenue', label: 'Revenue' },
                { value: 'funding_stage', label: 'Funding Stage' },
                { value: 'seniority', label: 'Seniority' },
                { value: 'department', label: 'Department' },
                { value: 'technologies', label: 'Technologies' },
                // Location
                { value: 'address', label: 'Address' },
                { value: 'city', label: 'City' },
                { value: 'state', label: 'State' },
                { value: 'country', label: 'Country' },
                { value: 'timezone', label: 'Timezone' },
                // Personalization
                { value: 'icebreaker', label: 'Icebreaker' },
                { value: 'trigger_event', label: 'Trigger Event' },
                { value: 'pain_points', label: 'Pain Points' },
                { value: 'notes', label: 'Notes' },
                { value: 'source', label: 'Source' },
                // Custom
                { value: 'custom_1', label: 'Custom 1' },
                { value: 'custom_2', label: 'Custom 2' },
                { value: 'custom_3', label: 'Custom 3' },
                { value: 'custom_4', label: 'Custom 4' },
                { value: 'custom_5', label: 'Custom 5' }
            ];

            // Auto-detect field mapping based on header names
            function guessField(header) {
                const h = header.toLowerCase();
                if (h.includes('first') && h.includes('name')) return 'first_name';
                if (h.includes('last') && h.includes('name')) return 'last_name';
                if (h === 'name' || h === 'full name') return 'first_name';
                if (h.includes('email')) return 'email';
                if (h.includes('phone') || h.includes('mobile') || h.includes('cell')) return 'phone';
                if (h.includes('company') || h.includes('organization') || h.includes('org')) return 'company';
                if (h.includes('title') || h.includes('position') || h.includes('job')) return 'title';
                if (h.includes('industry') || h.includes('sector')) return 'industry';
                if (h.includes('linkedin') && h.includes('company')) return 'company_linkedin_url';
                if (h.includes('linkedin')) return 'linkedin_url';
                if (h.includes('website') || h.includes('url') || h.includes('domain')) return 'website';
                if (h.includes('employee') || h.includes('headcount')) return 'employee_count';
                if (h.includes('size')) return 'company_size';
                if (h.includes('revenue') || h.includes('arr')) return 'revenue';
                if (h.includes('funding') || h.includes('round')) return 'funding_stage';
                if (h.includes('seniority') || h.includes('level')) return 'seniority';
                if (h.includes('department') || h.includes('function') || h.includes('team')) return 'department';
                if (h.includes('tech')) return 'technologies';
                if (h.includes('address') || h.includes('street')) return 'address';
                if (h.includes('city') || h.includes('location')) return 'city';
                if (h.includes('state') || h.includes('province') || h.includes('region')) return 'state';
                if (h.includes('country')) return 'country';
                if (h.includes('timezone') || h.includes('time zone')) return 'timezone';
                if (h.includes('icebreaker') || h.includes('intro')) return 'icebreaker';
                if (h.includes('trigger') || h.includes('event')) return 'trigger_event';
                if (h.includes('pain') || h.includes('challenge') || h.includes('problem')) return 'pain_points';
                if (h.includes('source') || h.includes('origin')) return 'source';
                if (h.includes('note') || h.includes('comment')) return 'notes';
                return '';
            }

            // Build mapping UI
            let mappingHtml = '<div style="display: grid; gap: 12px;">';
            csvHeaders.forEach((header, idx) => {
                const guessed = guessField(header);
                mappingHtml += `
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="flex: 1; color: #888;">${header}</span>
                        <span>→</span>
                        <select id="csv-map-${idx}" style="flex: 1; padding: 8px; background: #333; border: 1px solid #444; border-radius: 6px; color: #fff;">
                            ${dbFields.map(f => `<option value="${f.value}" ${f.value === guessed ? 'selected' : ''}>${f.label}</option>`).join('')}
                        </select>
                    </div>
                `;
            });
            mappingHtml += '</div>';
            document.getElementById('csv-mapping').innerHTML = mappingHtml;

            // Preview
            document.getElementById('csv-preview').innerHTML = `
                <p style="color: #888; margin-bottom: 8px;">${csvData.length} rows found</p>
            `;

            document.getElementById('csv-modal').classList.add('active');
        }

        function closeCsvModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('csv-modal').classList.remove('active');
            }
        }

        async function importCsv() {
            // Build field mapping
            const mapping = {};
            csvHeaders.forEach((header, idx) => {
                const select = document.getElementById(`csv-map-${idx}`);
                if (select && select.value) {
                    mapping[header] = select.value;
                }
            });

            // Check phone is mapped
            if (!Object.values(mapping).includes('phone')) {
                alert('Phone field must be mapped');
                return;
            }

            try {
                const response = await fetch('/api/leads/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        data: csvData,
                        mapping: mapping
                    })
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.detail || 'Import failed');
                }

                alert(`Imported ${result.imported} leads. ${result.skipped} skipped.`);
                closeCsvModal();
                loadLeads();
                loadLeadStats();
            } catch (error) {
                alert('Import failed: ' + error);
            }
        }

        // ========================================
        // AGENTS MANAGEMENT
        // ========================================

        async function loadAgents() {
            try {
                const response = await fetch('/api/agents');
                const agents = await response.json();

                const container = document.getElementById('agents-list');
                if (!agents || agents.length === 0) {
                    container.innerHTML = '<div class="card" style="text-align: center; padding: 40px; color: #666;">No agents configured</div>';
                    return;
                }

                container.innerHTML = agents.map(agent => `
                    <div class="card" style="margin-bottom: 16px; cursor: pointer; transition: border-color 0.2s; border: 1px solid ${agent.enabled ? '#333' : '#444'};" onclick="editAgent('${agent.id}')" onmouseover="this.style.borderColor='#4a9eff'" onmouseout="this.style.borderColor='${agent.enabled ? '#333' : '#444'}'">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="display: flex; gap: 16px; align-items: flex-start;">
                                <div style="font-size: 36px; line-height: 1;">${agent.icon || '🤖'}</div>
                                <div>
                                    <h3 style="margin: 0 0 4px 0; display: flex; align-items: center; gap: 8px;">
                                        ${agent.name}
                                        ${!agent.enabled ? '<span style="font-size: 11px; color: #888; font-weight: normal;">(disabled)</span>' : ''}
                                    </h3>
                                    <p style="color: #888; margin: 0 0 8px 0; font-size: 13px;">${agent.objective || ''}</p>
                                    <div style="display: flex; gap: 12px; font-size: 12px;">
                                        <span style="padding: 4px 8px; background: #222; border-radius: 4px;">
                                            ${agent.model_tier === 'opus' ? 'Opus (Reasoning)' : agent.model_tier === 'sonnet' ? 'Sonnet (Smart)' : 'Haiku (Fast)'}
                                        </span>
                                        <span style="color: #666;">
                                            ${agent.tools.length} tools
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <button class="btn btn-secondary" style="padding: 8px 16px;" onclick="event.stopPropagation(); editAgent('${agent.id}')">Edit</button>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load agents:', error);
            }
        }

        function showAgentTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.agent-tab').forEach(tab => tab.style.display = 'none');
            // Remove active from all buttons
            document.querySelectorAll('.agent-tab-btn').forEach(btn => btn.classList.remove('active'));
            // Show selected tab
            document.getElementById('agent-tab-' + tabName).style.display = 'block';
            // Mark button as active
            document.querySelector('.agent-tab-btn[data-tab="' + tabName + '"]').classList.add('active');
        }

        async function editAgent(agentId) {
            try {
                const response = await fetch(`/api/agents/${agentId}`);
                const agent = await response.json();

                // Populate form
                document.getElementById('agent-id').value = agent.id;
                document.getElementById('agent-modal-title').textContent = `Edit ${agent.name}`;
                document.getElementById('agent-name').value = agent.name;
                document.getElementById('agent-icon').value = agent.icon || '';
                document.getElementById('agent-objective').value = agent.objective || '';
                document.getElementById('agent-model-tier').value = agent.model_tier;
                document.getElementById('agent-enabled').checked = agent.enabled;
                document.getElementById('agent-persona').value = agent.persona;

                // Load knowledge base content
                if (agent.knowledge_base) {
                    const kbResponse = await fetch(`/api/agents/${agentId}/knowledge`);
                    if (kbResponse.ok) {
                        const kbData = await kbResponse.json();
                        document.getElementById('agent-knowledge').value = kbData.content || '';
                    } else {
                        document.getElementById('agent-knowledge').value = '';
                    }
                } else {
                    document.getElementById('agent-knowledge').value = '';
                }

                // Set tool checkboxes
                const allTools = ['search_web', 'get_movie_showtimes', 'make_call', 'send_sms', 'search_contacts', 'book_calendar', 'check_calendar'];
                allTools.forEach(tool => {
                    const checkbox = document.getElementById('tool-' + tool);
                    if (checkbox) {
                        checkbox.checked = agent.tools.includes(tool);
                    }
                });

                // Reset to general tab
                showAgentTab('general');

                // Show modal
                document.getElementById('agent-modal').classList.add('active');
            } catch (error) {
                console.error('Failed to load agent:', error);
                alert('Failed to load agent: ' + error);
            }
        }

        function closeAgentModal(event) {
            if (!event || event.target === event.currentTarget) {
                document.getElementById('agent-modal').classList.remove('active');
            }
        }

        async function saveAgent() {
            const agentId = document.getElementById('agent-id').value;

            // Collect selected tools
            const tools = [];
            ['search_web', 'get_movie_showtimes', 'make_call', 'send_sms', 'search_contacts', 'book_calendar', 'check_calendar'].forEach(tool => {
                const checkbox = document.getElementById('tool-' + tool);
                if (checkbox && checkbox.checked) {
                    tools.push(tool);
                }
            });

            const data = {
                name: document.getElementById('agent-name').value,
                icon: document.getElementById('agent-icon').value,
                objective: document.getElementById('agent-objective').value,
                model_tier: document.getElementById('agent-model-tier').value,
                enabled: document.getElementById('agent-enabled').checked,
                persona: document.getElementById('agent-persona').value,
                tools: tools
            };

            try {
                // Save agent
                const response = await fetch(`/api/agents/${agentId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to save agent');
                }

                // Save knowledge base
                const kbContent = document.getElementById('agent-knowledge').value;
                await fetch(`/api/agents/${agentId}/knowledge`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: kbContent })
                });

                closeAgentModal();
                loadAgents();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        // Calendar settings toggle
        function toggleCalendarSettings() {
            const provider = document.getElementById('calendar-provider').value;
            document.getElementById('calcom-settings').style.display = provider === 'cal.com' ? 'block' : 'none';
            document.getElementById('calendly-settings').style.display = provider === 'calendly' ? 'block' : 'none';
        }

        // =============================================================================
        // Email Account Management
        // =============================================================================

        let emailPresets = {};

        async function loadEmailAccounts() {
            try {
                const res = await fetch('/api/email-accounts');
                const data = await res.json();

                // Update stats
                document.getElementById('stat-total-accounts').textContent = data.total_accounts || 0;
                document.getElementById('stat-active-accounts').textContent = data.active_accounts || 0;
                document.getElementById('stat-sent-today').textContent = data.emails_sent_today || 0;
                document.getElementById('stat-remaining-capacity').textContent = data.remaining_capacity_today || 0;

                // Render accounts list
                const container = document.getElementById('email-accounts-list');
                const accounts = data.accounts || [];

                if (accounts.length === 0) {
                    container.innerHTML = `
                        <div style="text-align: center; padding: 40px; color: #888;">
                            <p style="font-size: 48px; margin-bottom: 16px;">📧</p>
                            <p>No email accounts configured yet.</p>
                            <p style="margin-top: 8px;">Click "Add Account" to set up your first email account for campaigns.</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = accounts.map(account => `
                    <div class="email-account-card" style="background: #333; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="flex: 1;">
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                                    <span style="font-weight: bold;">${account.email}</span>
                                    ${getStatusBadge(account.status)}
                                </div>
                                <div style="color: #888; font-size: 13px;">
                                    ${account.display_name || 'No display name'} &bull; ${account.smtp_host}:${account.smtp_port}
                                </div>
                                <div style="display: flex; gap: 16px; margin-top: 12px; color: #888; font-size: 12px;">
                                    <span>Sent: ${account.emails_sent_today}/${account.daily_limit} today</span>
                                    <span>Health: ${account.health_score}%</span>
                                    ${account.warmup_enabled ? `<span>Warmup day ${account.warmup_day}</span>` : ''}
                                    ${account.last_error ? `<span style="color: #f87171;">${account.last_error.substring(0, 50)}...</span>` : ''}
                                </div>
                            </div>
                            <div style="display: flex; gap: 8px;">
                                <button class="btn btn-secondary" onclick="openTestEmailModal(${account.id})" style="padding: 8px 12px; font-size: 12px;">Send Test</button>
                                <button class="btn btn-secondary" onclick="editEmailAccount(${account.id})" style="padding: 8px 12px; font-size: 12px;">Edit</button>
                                <button class="btn btn-secondary" onclick="deleteEmailAccount(${account.id})" style="padding: 8px 12px; font-size: 12px; color: #f87171;">Delete</button>
                            </div>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Failed to load email accounts:', error);
                document.getElementById('email-accounts-list').innerHTML = `
                    <p style="color: #f87171; text-align: center; padding: 20px;">Failed to load email accounts</p>
                `;
            }
        }

        function getStatusBadge(status) {
            const badges = {
                'active': '<span style="background: #166534; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Active</span>',
                'pending': '<span style="background: #854d0e; color: #fbbf24; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Pending</span>',
                'warmup': '<span style="background: #9a3412; color: #fb923c; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Warmup</span>',
                'error': '<span style="background: #7f1d1d; color: #f87171; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Error</span>',
                'disabled': '<span style="background: #374151; color: #9ca3af; padding: 2px 8px; border-radius: 4px; font-size: 11px;">Disabled</span>'
            };
            return badges[status] || badges['pending'];
        }

        async function loadEmailPresets() {
            try {
                const res = await fetch('/api/email-accounts/presets');
                emailPresets = await res.json();
            } catch (error) {
                console.error('Failed to load email presets:', error);
            }
        }

        function openEmailAccountModal(accountId = null) {
            document.getElementById('email-modal').classList.add('active');
            document.getElementById('email-modal-title').textContent = accountId ? 'Edit Email Account' : 'Add Email Account';
            document.getElementById('email-account-id').value = accountId || '';

            // Reset form
            if (!accountId) {
                document.getElementById('email-preset').value = '';
                document.getElementById('email-email').value = '';
                document.getElementById('email-display-name').value = '';
                document.getElementById('email-smtp-host').value = '';
                document.getElementById('email-smtp-port').value = '587';
                document.getElementById('email-smtp-username').value = '';
                document.getElementById('email-smtp-password').value = '';
                document.getElementById('email-smtp-tls').checked = true;
                // IMAP fields
                document.getElementById('email-imap-host').value = '';
                document.getElementById('email-imap-port').value = '993';
                document.getElementById('email-imap-username').value = '';
                document.getElementById('email-imap-password').value = '';
                // Limits
                document.getElementById('email-daily-limit').value = '100';
                document.getElementById('email-hourly-limit').value = '20';
                document.getElementById('email-delay').value = '60';
                document.getElementById('email-warmup').checked = false;
                document.getElementById('email-signature-html').value = '';
                document.getElementById('preset-notes').textContent = '';
            }
        }

        function closeEmailModal(event) {
            if (!event || event.target.classList.contains('modal-overlay')) {
                document.getElementById('email-modal').classList.remove('active');
            }
        }

        function applyEmailPreset() {
            const preset = document.getElementById('email-preset').value;
            if (!preset || !emailPresets[preset]) {
                document.getElementById('preset-notes').textContent = '';
                return;
            }

            const config = emailPresets[preset];
            document.getElementById('email-smtp-host').value = config.smtp_host || '';
            document.getElementById('email-smtp-port').value = config.smtp_port || 587;
            document.getElementById('email-smtp-tls').checked = config.smtp_use_tls !== false;
            // IMAP from preset
            document.getElementById('email-imap-host').value = config.imap_host || '';
            document.getElementById('email-imap-port').value = config.imap_port || 993;
            document.getElementById('preset-notes').textContent = config.notes || '';
        }

        async function editEmailAccount(accountId) {
            try {
                const res = await fetch(`/api/email-accounts/${accountId}`);
                const account = await res.json();

                document.getElementById('email-account-id').value = account.id;
                document.getElementById('email-modal-title').textContent = 'Edit Email Account';
                document.getElementById('email-preset').value = '';
                document.getElementById('email-email').value = account.email;
                document.getElementById('email-display-name').value = account.display_name || '';
                document.getElementById('email-smtp-host').value = account.smtp_host;
                document.getElementById('email-smtp-port').value = account.smtp_port;
                document.getElementById('email-smtp-username').value = account.smtp_username;
                document.getElementById('email-smtp-password').value = ''; // Don't pre-fill password
                document.getElementById('email-smtp-tls').checked = account.smtp_use_tls;
                // IMAP fields
                document.getElementById('email-imap-host').value = account.imap_host || '';
                document.getElementById('email-imap-port').value = account.imap_port || 993;
                document.getElementById('email-imap-username').value = account.imap_username || '';
                document.getElementById('email-imap-password').value = ''; // Don't pre-fill password
                // Limits
                document.getElementById('email-daily-limit').value = account.daily_limit;
                document.getElementById('email-hourly-limit').value = account.hourly_limit;
                document.getElementById('email-delay').value = account.delay_between_emails_seconds;
                document.getElementById('email-warmup').checked = account.warmup_enabled;
                document.getElementById('email-signature-html').value = account.signature_html || '';

                document.getElementById('email-modal').classList.add('active');
            } catch (error) {
                alert('Failed to load account: ' + error.message);
            }
        }

        async function saveEmailAccount() {
            const accountId = document.getElementById('email-account-id').value;
            const email = document.getElementById('email-email').value;
            const data = {
                email: email,
                display_name: document.getElementById('email-display-name').value,
                smtp_host: document.getElementById('email-smtp-host').value,
                smtp_port: parseInt(document.getElementById('email-smtp-port').value) || 587,
                smtp_username: document.getElementById('email-smtp-username').value || email,
                smtp_password: document.getElementById('email-smtp-password').value,
                smtp_use_tls: document.getElementById('email-smtp-tls').checked,
                // IMAP fields
                imap_host: document.getElementById('email-imap-host').value,
                imap_port: parseInt(document.getElementById('email-imap-port').value) || 993,
                imap_username: document.getElementById('email-imap-username').value || email,
                imap_password: document.getElementById('email-imap-password').value || document.getElementById('email-smtp-password').value,
                // Limits
                daily_limit: parseInt(document.getElementById('email-daily-limit').value) || 100,
                hourly_limit: parseInt(document.getElementById('email-hourly-limit').value) || 20,
                delay_between_emails_seconds: parseInt(document.getElementById('email-delay').value) || 60,
                warmup_enabled: document.getElementById('email-warmup').checked,
                signature_html: document.getElementById('email-signature-html').value,
                preset: document.getElementById('email-preset').value
            };

            // Validation
            if (!data.email || !data.smtp_host || !data.smtp_password) {
                alert('Please fill in all required fields (email, SMTP host, password)');
                return;
            }

            try {
                const url = accountId ? `/api/email-accounts/${accountId}` : '/api/email-accounts';
                const method = accountId ? 'PUT' : 'POST';

                const res = await fetch(url, {
                    method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!res.ok) {
                    const error = await res.json();
                    throw new Error(error.detail || 'Failed to save account');
                }

                closeEmailModal();
                loadEmailAccounts();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function deleteEmailAccount(accountId) {
            if (!confirm('Are you sure you want to delete this email account?')) return;

            try {
                const res = await fetch(`/api/email-accounts/${accountId}`, { method: 'DELETE' });
                if (!res.ok) throw new Error('Failed to delete');
                loadEmailAccounts();
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function testEmailAccount() {
            const accountId = document.getElementById('email-account-id').value;
            if (!accountId) {
                alert('Please save the account first before testing');
                return;
            }

            const btn = document.getElementById('test-email-btn');
            btn.textContent = 'Testing...';
            btn.disabled = true;

            try {
                const res = await fetch(`/api/email-accounts/${accountId}/test`, { method: 'POST' });
                const result = await res.json();

                if (result.success) {
                    alert('Connection successful!');
                } else {
                    alert('Connection failed: ' + result.message);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.textContent = 'Test Connection';
                btn.disabled = false;
            }
        }

        function openTestEmailModal(accountId) {
            document.getElementById('test-email-account-id').value = accountId;
            document.getElementById('test-email-to').value = '';
            document.getElementById('test-email-modal').classList.add('active');
        }

        function closeTestEmailModal(event) {
            if (!event || event.target.classList.contains('modal-overlay')) {
                document.getElementById('test-email-modal').classList.remove('active');
            }
        }

        async function sendTestEmail() {
            const accountId = document.getElementById('test-email-account-id').value;
            const toEmail = document.getElementById('test-email-to').value;

            if (!toEmail) {
                alert('Please enter an email address');
                return;
            }

            try {
                const res = await fetch('/api/email/send-test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ account_id: parseInt(accountId), to_email: toEmail })
                });

                const result = await res.json();
                if (result.success) {
                    alert('Test email sent successfully! Check your inbox.');
                    closeTestEmailModal();
                } else {
                    alert('Failed to send: ' + (result.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        // Initialize
        connectWebSocket();
        loadCallAgents();
        loadHistory();
        loadSettings();
        loadApiKeys();
        loadIntegrations();
        loadModemStatus();
        loadSmsStatus();
        loadConversations();
        loadInbox();  // Unified inbox
        loadAgents();
        loadLeads();
        loadLeadStats();
        loadEmailAccounts();
        loadEmailPresets();
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
    global incoming_handler, incoming_listener_task, shared_modem

    call_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Stop incoming listener if running (modem can only do one thing at a time)
    was_listening = incoming_listener_task is not None and not incoming_listener_task.done()
    if was_listening:
        logger.info("Pausing incoming listener for outbound call")
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
        await asyncio.sleep(1)  # Give modem time to release

    # Merge saved settings with call context
    settings = load_settings()
    merged_context = {**settings, **request.context}  # Call context overrides settings

    # Add agent_id to context if provided
    if request.agent_id:
        merged_context["AGENT_ID"] = request.agent_id

    # Create agent with pre-loaded conversation engine (models already initialized at startup)
    # Pass shared_modem so we don't create a conflicting connection
    agent = PhoneAgentLocal(pre_initialize=False, conversation_engine=preloaded_conversation, modem=shared_modem)

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
        global incoming_handler, incoming_listener_task
        try:
            # Broadcast dialing status
            await broadcast({
                "type": "status",
                "status": "dialing"
            })

            # Models are pre-loaded at startup, so no initialization delay here
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

            # Restart incoming listener if it was running
            if was_listening:
                logger.info("Restarting incoming listener after outbound call")
                await asyncio.sleep(2)  # Give modem time to settle
                incoming_handler = IncomingCallHandler()
                incoming_listener_task = asyncio.create_task(incoming_handler.start_listening())
                await broadcast({
                    "type": "incoming_listener_status",
                    "listening": True
                })

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
    """Update user settings - MERGES with existing settings to preserve api_keys etc."""
    global sms_handler

    # Load existing settings first
    existing = load_settings()

    # Deep merge: update existing with new values
    def deep_merge(base: dict, updates: dict) -> dict:
        """Recursively merge updates into base dict"""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(existing, settings)
    save_settings(merged)

    # API keys are read fresh from settings on each API call - no reload needed

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


# ==========================================
# SMS COMMAND API ENDPOINTS
# ==========================================

@app.get("/api/sms/status")
async def get_sms_status():
    """Get SMS command monitor status"""
    global sms_handler, sms_monitor_task

    is_monitoring = (sms_monitor_task is not None and
                     not sms_monitor_task.done())

    settings = load_settings()
    primary_phone = settings.get("sms", {}).get("PRIMARY_PHONE", "")

    return {
        "monitoring": is_monitoring,
        "primary_phone": primary_phone,
        "pending_calls": len(sms_handler.pending_calls) if sms_handler else 0
    }


@app.post("/api/sms/start")
async def start_sms_monitor():
    """Start the SMS command monitor"""
    global sms_handler, sms_monitor_task

    # Check if already running
    if sms_monitor_task and not sms_monitor_task.done():
        return {"status": "already_running"}

    # Get primary phone from settings (try sms.PRIMARY_PHONE first, then fall back to CALLBACK_NUMBER)
    settings = load_settings()
    primary_phone = settings.get("sms", {}).get("PRIMARY_PHONE", "")

    if not primary_phone:
        # Fall back to CALLBACK_NUMBER
        import re
        callback = settings.get("CALLBACK_NUMBER") or settings.get("personal", {}).get("CALLBACK_NUMBER")
        if callback:
            primary_phone = re.sub(r'\D', '', callback)  # Strip non-digits

    if not primary_phone:
        raise HTTPException(400, "Primary phone number not configured in settings")

    # Create AI-powered handler (settings loaded dynamically via property)
    sms_handler = SMSAIHandler(primary_phone)

    # Pending auto-replies (phone -> (message, scheduled_time))
    pending_auto_replies = {}

    # Start monitor using push notifications (no polling!)
    def sms_monitor_loop_sync():
        global shared_modem, modem_status_cache
        from sim7600_modem import SIM7600Modem
        import time as time_module
        import random

        # Outer loop for automatic reconnection
        while True:
            modem = None
            try:
                logger.info("Attempting to connect to modem...")
                modem = SIM7600Modem()
                if not modem.connect():
                    logger.error("Failed to connect to modem, retrying in 10 seconds...")
                    modem_status_cache = {"connected": False, "signal_strength": 0}
                    time_module.sleep(10)
                    continue

                # Set the shared modem and update cache
                shared_modem = modem
                signal = modem.get_signal_strength()
                modem_status_cache = {"connected": True, "signal_strength": signal}
                logger.info(f"Modem connected successfully. Signal: {signal}")

                # Set up SMS sending callback for the handler (with auto-splitting)
                def send_sms_callback(phone: str, message: str) -> bool:
                    try:
                        chunks = split_sms(message)
                        for chunk in chunks:
                            if not modem.send_sms(phone, chunk):
                                return False
                            if len(chunks) > 1:
                                time_module.sleep(0.5)
                        return True
                    except Exception as e:
                        logger.error(f"SMS send error: {e}")
                        return False

                sms_handler.on_send_sms(send_sms_callback)

                # SMS callback - AI processes all messages
                def on_sms_received(sender: str, text: str):
                    nonlocal modem
                    logger.info(f"SMS received from {sender}: {text}")

                    # Get current settings (handler fetches fresh settings via property)
                    current_settings = load_settings()
                    my_phone = current_settings.get("CALLBACK_NUMBER", "")
                    try:
                        database.save_message({
                            "channel": "sms",
                            "direction": "inbound",
                            "from_address": sender,
                            "to_address": my_phone,
                            "body": text,
                            "status": "received",
                            "provider": "modem"
                        })
                    except Exception as e:
                        logger.error(f"Failed to save incoming SMS to database: {e}")

                    # Process with AI - handles both main user and others
                    try:
                        response = sms_handler.process_message(sender, text)

                        if response:
                            is_main = sms_handler.is_main_user(sender)

                            if is_main:
                                # Send immediately to main user (split if too long)
                                chunks = split_sms(response)
                                for chunk in chunks:
                                    modem.send_sms(sender, chunk)
                                    if len(chunks) > 1:
                                        time_module.sleep(0.5)  # Brief pause between messages
                                logger.info(f"Sent {len(chunks)} message(s) to main user")

                                # Save outbound message to database for conversation history
                                try:
                                    database.save_message({
                                        "channel": "sms",
                                        "direction": "outbound",
                                        "from_address": my_phone,
                                        "to_address": sender,
                                        "body": response,
                                        "status": "sent",
                                        "provider": "modem",
                                        "sent_at": datetime.now().isoformat()
                                    })
                                except Exception as e:
                                    logger.error(f"Failed to save outbound SMS: {e}")
                            else:
                                # Schedule with delay for non-main users (more human-like)
                                autopilot = current_settings.get("autopilot", {})
                                delay_min = autopilot.get("REPLY_DELAY_MIN", 30)
                                delay_max = autopilot.get("REPLY_DELAY_MAX", 120)
                                delay = random.randint(delay_min, delay_max)

                                scheduled_time = time_module.time() + delay
                                pending_auto_replies[sender] = (response, scheduled_time)
                                logger.info(f"Scheduled reply to {sender} in {delay} seconds")

                    except Exception as e:
                        logger.error(f"Error processing SMS with AI: {e}")
                        # Send error to main user only
                        if sms_handler.is_main_user(sender):
                            try:
                                modem.send_sms(sender, f"Error: {str(e)[:100]}")
                            except:
                                pass

                # Register SMS callback - modem will call this on +CMTI
                modem.on_sms(on_sms_received)

                logger.info(f"SMS monitor started. Listening for commands from {primary_phone}")

                # Inner loop - handles auto-replies and signal updates
                # Breaks out on disconnect to trigger reconnection
                signal_check_counter = 0
                health_check_counter = 0
                consecutive_errors = 0
                while True:
                    try:
                        # Quick health check every 10 seconds
                        health_check_counter += 1
                        if health_check_counter >= 10:
                            health_check_counter = 0
                            if not modem.is_connected:
                                logger.warning("Modem health check failed, attempting reconnect...")
                                if modem.reconnect():
                                    logger.info("Modem reconnected successfully via health check")
                                    consecutive_errors = 0
                                else:
                                    consecutive_errors += 1
                                    if consecutive_errors >= 3:
                                        logger.error("Modem reconnection failed 3 times, breaking to restart...")
                                        modem_status_cache = {"connected": False, "signal_strength": 0}
                                        raise Exception("Modem reconnection failed")

                        # Update signal strength every 60 seconds
                        signal_check_counter += 1
                        if signal_check_counter >= 60:
                            signal_check_counter = 0
                            try:
                                with shared_modem_lock:
                                    signal = modem.get_signal_strength()
                                    modem_status_cache["signal_strength"] = signal
                                    consecutive_errors = 0  # Reset on success
                            except Exception as e:
                                consecutive_errors += 1
                                if consecutive_errors >= 5:
                                    logger.error(f"Modem appears disconnected after {consecutive_errors} errors, reconnecting...")
                                    modem_status_cache = {"connected": False, "signal_strength": 0}
                                    raise Exception("Modem disconnected")

                        # Check for pending auto-replies that are due
                        current_time = time_module.time()
                        for phone, (reply_msg, scheduled_time) in list(pending_auto_replies.items()):
                            if current_time >= scheduled_time:
                                try:
                                    logger.info(f"Sending delayed auto-reply to {phone}")
                                    chunks = split_sms(reply_msg)
                                    with shared_modem_lock:
                                        for chunk in chunks:
                                            modem.send_sms(phone, chunk)
                                            if len(chunks) > 1:
                                                time_module.sleep(0.5)
                                    # Save outbound message to database
                                    my_phone = load_settings().get("CALLBACK_NUMBER", "")
                                    database.save_message({
                                        "channel": "sms",
                                        "direction": "outbound",
                                        "from_address": my_phone,
                                        "to_address": phone,
                                        "body": reply_msg,
                                        "status": "sent",
                                        "provider": "modem",
                                        "sent_at": datetime.now().isoformat()
                                    })
                                except Exception as e:
                                    logger.error(f"Failed to send auto-reply: {e}")
                                del pending_auto_replies[phone]

                        time_module.sleep(1)  # Check auto-replies every second

                        # Check for pending calls and execute them
                        while sms_handler.has_pending_calls():
                            pending = sms_handler.get_pending_call()
                            if pending:
                                logger.info(f"Executing pending call: {pending}")

                                # Make the call
                                try:
                                    # Use shared modem and pre-loaded conversation engine (no model re-loading)
                                    agent = PhoneAgentLocal(
                                        modem=modem,
                                        pre_initialize=False,
                                        conversation_engine=preloaded_conversation
                                    )

                                    # Use the stored main event loop for broadcasting
                                    global main_event_loop

                                    # Register transcript callback to broadcast to dashboard
                                    def on_sms_call_transcript(role, text):
                                        if main_event_loop and main_event_loop.is_running():
                                            asyncio.run_coroutine_threadsafe(
                                                broadcast({
                                                    "type": "transcript",
                                                    "role": role,
                                                    "text": text,
                                                    "source": "sms_call",
                                                    "phone": pending.get('phone', '')
                                                }),
                                                main_event_loop
                                            )

                                    def on_sms_call_state(state):
                                        status_map = {
                                            "idle": "idle",
                                            "listening": "connected",
                                            "processing": "speaking",
                                            "speaking": "speaking",
                                            "completed": "ended",
                                            "failed": "failed"
                                        }
                                        status = status_map.get(state.value, state.value)
                                        if main_event_loop and main_event_loop.is_running():
                                            asyncio.run_coroutine_threadsafe(
                                                broadcast({
                                                    "type": "status",
                                                    "status": status,
                                                    "source": "sms_call",
                                                    "phone": pending.get('phone', '')
                                                }),
                                                main_event_loop
                                            )

                                    agent.on_transcript(on_sms_call_transcript)
                                    agent.on_state_change(on_sms_call_state)

                                    # Broadcast that a call is starting
                                    if main_event_loop and main_event_loop.is_running():
                                        asyncio.run_coroutine_threadsafe(
                                            broadcast({
                                                "type": "sms_call_started",
                                                "phone": pending.get('phone', ''),
                                                "contact_name": pending.get('contact_name', ''),
                                                "objective": pending.get('objective', '')
                                            }),
                                            main_event_loop
                                        )

                                    call_settings = load_settings()

                                    # Build context from settings (top-level keys)
                                    context = {}
                                    for key in ['MY_NAME', 'ADDRESS', 'CITY', 'STATE', 'ZIP', 'CALLBACK_NUMBER',
                                                'EMAIL', 'COMPANY', 'CARD_NUMBER', 'CARD_EXP', 'CARD_CVV', 'BILLING_ZIP',
                                                'VEHICLE_YEAR', 'VEHICLE_MAKE', 'VEHICLE_MODEL', 'VEHICLE_COLOR']:
                                        if call_settings.get(key):
                                            context[key] = call_settings[key]

                                    # Add agent_id to context if specified
                                    if pending.get('agent_id'):
                                        context['AGENT_ID'] = pending['agent_id']

                                    # Add lead info to context if available
                                    if pending.get('lead_id'):
                                        lead = database.get_lead(pending['lead_id'])
                                        if lead:
                                            context['LEAD_ID'] = lead['id']
                                            context['LEAD_FIRST_NAME'] = lead.get('first_name', '')
                                            context['LEAD_LAST_NAME'] = lead.get('last_name', '')
                                            context['LEAD_NAME'] = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
                                            context['LEAD_COMPANY'] = lead.get('company', '')
                                            context['LEAD_PHONE'] = lead.get('phone', '')
                                            context['LEAD_EMAIL'] = lead.get('email', '')

                                    request = CallRequest(
                                        phone=pending['phone'],
                                        objective=pending['objective'],
                                        context=context,
                                        enable_tools=True  # SMS-initiated calls have full tool access
                                    )

                                    # call() is async, so run it in event loop
                                    result = asyncio.run(agent.call(request))

                                    # Update lead status if we have lead_id
                                    if pending.get('lead_id'):
                                        database.log_interaction(pending['lead_id'], {
                                            "channel": "call",
                                            "direction": "outbound",
                                            "duration_seconds": result.duration_seconds,
                                            "summary": result.summary,
                                            "outcome": "completed" if result.success else "failed",
                                            "objective": pending['objective']
                                        })

                                    # SMS summary is already sent by agent_local._send_sms_summary()

                                    # Broadcast call result to dashboard
                                    if main_event_loop and main_event_loop.is_running():
                                        asyncio.run_coroutine_threadsafe(
                                            broadcast({
                                                "type": "result",
                                                "success": result.success,
                                                "summary": result.summary,
                                                "source": "sms_call",
                                                "phone": pending.get('phone', ''),
                                                "contact_name": pending.get('contact_name', '')
                                            }),
                                            main_event_loop
                                        )

                                    # After call completes, verify modem is still connected
                                    if not modem.is_connected:
                                        logger.warning("Modem disconnected after call, triggering reconnect...")
                                        raise Exception("Modem disconnected after call")

                                except Exception as e:
                                    logger.error(f"Error making SMS-initiated call: {e}")
                                    try:
                                        modem.send_sms(primary_phone, f"Call failed: {str(e)[:100]}")
                                    except:
                                        pass

                                    # Check if modem is still connected after error
                                    if not modem.is_connected:
                                        logger.warning("Modem disconnected during call error handling")
                                        raise Exception("Modem disconnected")

                    except Exception as e:
                        error_str = str(e).lower()
                        if "no such device" in error_str or "disconnected" in error_str:
                            logger.error(f"Modem disconnected: {e}")
                            modem_status_cache = {"connected": False, "signal_strength": 0}
                            break  # Break inner loop to reconnect
                        else:
                            logger.error(f"SMS monitor error: {e}")
                            time_module.sleep(5)

            except Exception as e:
                logger.error(f"Modem connection error: {e}")
                modem_status_cache = {"connected": False, "signal_strength": 0}

            # Cleanup before reconnect attempt
            if modem:
                try:
                    modem.disconnect()
                except:
                    pass
                modem = None
                shared_modem = None

            logger.info("Waiting 5 seconds before reconnecting to modem...")
            time_module.sleep(5)

    # Start the monitor in a background thread
    import threading
    sms_thread = threading.Thread(target=sms_monitor_loop_sync, daemon=True)
    sms_thread.start()

    return {"status": "started"}


@app.post("/api/sms/stop")
async def stop_sms_monitor():
    """Stop the SMS command monitor"""
    global sms_handler, sms_monitor_task

    if sms_monitor_task:
        sms_monitor_task.cancel()
        try:
            await sms_monitor_task
        except asyncio.CancelledError:
            pass
        sms_monitor_task = None

    sms_handler = None

    await broadcast({
        "type": "sms_monitor_status",
        "monitoring": False
    })

    return {"status": "stopped"}


# ==========================================
# MODEM STATUS API ENDPOINT
# ==========================================

@app.get("/api/modem/status")
async def get_modem_status():
    """Get modem connection and signal status from cache"""
    global modem_status_cache

    # Return cached status (updated by SMS monitor thread)
    return modem_status_cache


# ==========================================
# SMS MESSAGES API ENDPOINTS
# ==========================================

@app.get("/api/sms/messages")
async def get_sms_messages():
    """Get recent SMS messages - returns empty since SMS monitor handles this"""
    # The SMS monitor thread handles reading SMS messages
    # We don't read directly here to avoid blocking the event loop
    # and conflicting with the monitor's modem access
    return []


# ==========================================
# CONVERSATIONS / INBOX API ENDPOINTS
# ==========================================

@app.get("/api/conversations")
async def get_conversations(limit: int = 50, offset: int = 0):
    """Get all conversations for the inbox view"""
    conversations = database.get_conversations(limit=limit, offset=offset)
    return conversations


@app.get("/api/conversations/{contact_address}/messages")
async def get_conversation_messages(contact_address: str, limit: int = 100, offset: int = 0):
    """Get messages for a specific conversation"""
    messages = database.get_conversation_messages(contact_address, limit=limit, offset=offset)
    return messages


@app.post("/api/conversations/{contact_address}/read")
async def mark_conversation_read(contact_address: str):
    """Mark a conversation as read"""
    database.mark_conversation_read(contact_address)
    return {"status": "ok"}


@app.get("/api/conversations/unread")
async def get_unread_count():
    """Get total unread message count"""
    count = database.get_unread_count()
    return {"unread": count}


@app.get("/api/conversations/{contact_address}/autopilot")
async def get_thread_autopilot(contact_address: str):
    """Get autopilot status for a specific thread"""
    # Check if this thread has autopilot disabled
    disabled = database.is_thread_autopilot_disabled(contact_address)
    return {"enabled": not disabled}


@app.put("/api/conversations/{contact_address}/autopilot")
async def set_thread_autopilot(contact_address: str, data: dict):
    """Set autopilot status for a specific thread"""
    enabled = data.get("enabled", True)
    database.set_thread_autopilot(contact_address, enabled)
    return {"status": "ok", "enabled": enabled}


# ==========================================
# UNIFIED INBOX API ENDPOINTS
# ==========================================

@app.get("/api/inbox")
async def get_unified_inbox(
    channel: Optional[str] = None,
    direction: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get unified inbox conversations with filters.

    Query params:
    - channel: Filter by 'sms', 'email', or 'call'
    - direction: Filter by 'inbound' or 'outbound'
    - search: Full-text search query
    - limit: Max results (default 50)
    - offset: Pagination offset
    """
    conversations, total = database.get_unified_inbox(
        channel=channel,
        direction=direction,
        search=search,
        limit=limit,
        offset=offset
    )
    return {
        "conversations": conversations,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/inbox/{contact_address}/messages")
async def get_contact_messages(
    contact_address: str,
    channel: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get all messages for a contact across all channels"""
    messages = database.get_contact_messages(
        contact_address=contact_address,
        channel=channel,
        limit=limit,
        offset=offset
    )
    return messages


class InboxSearchRequest(BaseModel):
    query: str
    limit: int = 50
    offset: int = 0


@app.post("/api/inbox/search")
async def search_inbox(request: InboxSearchRequest):
    """Full-text search across all messages"""
    results = database.search_messages_fts(
        query=request.query,
        limit=request.limit,
        offset=request.offset
    )
    return {"results": results, "query": request.query}


# ==========================================
# AUTOPILOT QUEUE API ENDPOINTS
# ==========================================

@app.get("/api/autopilot/queue")
async def get_autopilot_queue(limit: int = 50):
    """Get all pending autopilot responses"""
    pending = database.get_pending_autopilot_responses(limit=limit)
    return {"pending": pending, "count": len(pending)}


@app.get("/api/autopilot/queue/{queue_id}")
async def get_autopilot_response(queue_id: int):
    """Get a single autopilot queue entry"""
    entry = database.get_autopilot_response(queue_id)
    if not entry:
        raise HTTPException(404, "Autopilot response not found")
    return entry


class AutopilotUpdateRequest(BaseModel):
    proposed_message: Optional[str] = None
    scheduled_send_at: Optional[str] = None


@app.put("/api/autopilot/queue/{queue_id}")
async def update_autopilot_response(queue_id: int, request: AutopilotUpdateRequest):
    """Update a pending autopilot response (e.g., edit the message)"""
    data = {}
    if request.proposed_message is not None:
        data["proposed_message"] = request.proposed_message
    if request.scheduled_send_at is not None:
        data["scheduled_send_at"] = request.scheduled_send_at

    if not data:
        raise HTTPException(400, "No fields to update")

    if not database.update_autopilot_response(queue_id, data):
        raise HTTPException(404, "Autopilot response not found or already processed")
    return {"status": "updated"}


@app.post("/api/autopilot/queue/{queue_id}/approve")
async def approve_autopilot_response(queue_id: int):
    """Approve a pending autopilot response for immediate sending"""
    entry = database.approve_autopilot_response(queue_id)
    if not entry:
        raise HTTPException(404, "Autopilot response not found or already processed")

    # TODO: Actually send the message via modem/email
    # For now, just mark as approved - background processor will handle sending

    await broadcast({
        "type": "autopilot_approved",
        "queue_id": queue_id,
        "contact_address": entry.get("contact_address")
    })

    return {"status": "approved", "entry": entry}


@app.post("/api/autopilot/queue/{queue_id}/cancel")
async def cancel_autopilot_response(queue_id: int):
    """Cancel a pending autopilot response"""
    if not database.cancel_autopilot_response(queue_id):
        raise HTTPException(404, "Autopilot response not found or already processed")

    await broadcast({
        "type": "autopilot_cancelled",
        "queue_id": queue_id
    })

    return {"status": "cancelled"}


class SmsRequest(BaseModel):
    phone: str
    message: str


@app.post("/api/sms/send")
async def send_sms_message(request: SmsRequest):
    """Send an SMS message using the shared modem"""
    global shared_modem

    def send_sms_sync():
        """Run SMS send in thread to avoid blocking"""
        if shared_modem:
            with shared_modem_lock:
                return shared_modem.send_sms(request.phone, request.message)
        else:
            from sim7600_modem import SIM7600Modem
            modem = SIM7600Modem()
            if not modem.connect():
                return False
            success = modem.send_sms(request.phone, request.message)
            modem.disconnect()
            return success

    try:
        # Run in thread pool to avoid blocking async event loop
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, send_sms_sync)

        if not success:
            raise HTTPException(500, "Failed to send SMS")

        # Save outbound message to database
        settings = load_settings()
        my_phone = settings.get("CALLBACK_NUMBER", "")
        try:
            database.save_message({
                "channel": "sms",
                "direction": "outbound",
                "from_address": my_phone,
                "to_address": request.phone,
                "body": request.message,
                "status": "sent",
                "provider": "modem",
                "sent_at": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to save outbound SMS to database: {e}")

        return {"status": "sent", "phone": request.phone}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error sending SMS: {str(e)}")


# ==========================================
# KNOWLEDGE BASE API ENDPOINTS
# ==========================================

@app.get("/api/knowledge")
async def list_knowledge_bases():
    """List all knowledge bases"""
    return knowledge_base.list_knowledge_bases()


@app.post("/api/knowledge")
async def create_knowledge_base_endpoint(data: dict):
    """Create a new knowledge base"""
    try:
        kb_id = knowledge_base.create_knowledge_base(
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            objective_keywords=data.get("objective_keywords", [])
        )
        return {"id": kb_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/knowledge/{kb_id}")
async def get_knowledge_base_endpoint(kb_id: str):
    """Get a knowledge base by ID"""
    kb = knowledge_base.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@app.put("/api/knowledge/{kb_id}")
async def update_knowledge_base_endpoint(kb_id: str, data: dict):
    """Update a knowledge base"""
    if not knowledge_base.update_knowledge_base(kb_id, data):
        raise HTTPException(404, "Knowledge base not found")
    return {"status": "updated"}


@app.delete("/api/knowledge/{kb_id}")
async def delete_knowledge_base_endpoint(kb_id: str):
    """Delete a knowledge base"""
    if not knowledge_base.delete_knowledge_base(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return {"status": "deleted"}


@app.get("/api/knowledge/{kb_id}/documents")
async def list_documents_endpoint(kb_id: str, doc_type: Optional[str] = None):
    """List documents in a knowledge base"""
    return knowledge_base.list_documents(kb_id, doc_type=doc_type)


@app.post("/api/knowledge/{kb_id}/documents")
async def add_document_endpoint(kb_id: str, data: dict):
    """Add a document to a knowledge base"""
    try:
        doc_id = knowledge_base.add_document(
            kb_id=kb_id,
            title=data.get("title", ""),
            content=data.get("content", ""),
            doc_type=data.get("doc_type", "text"),
            tags=data.get("tags", [])
        )
        return {"id": doc_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/knowledge/{kb_id}/documents/{doc_id}")
async def get_document_endpoint(kb_id: str, doc_id: str):
    """Get a document by ID"""
    doc = knowledge_base.get_document(kb_id, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.delete("/api/knowledge/{kb_id}/documents/{doc_id}")
async def delete_document_endpoint(kb_id: str, doc_id: str):
    """Delete a document"""
    if not knowledge_base.delete_document(kb_id, doc_id):
        raise HTTPException(404, "Document not found")
    return {"status": "deleted"}


@app.get("/api/knowledge/search")
async def search_knowledge_endpoint(query: str, kb_ids: Optional[str] = None, limit: int = 5):
    """Search for relevant documents"""
    kb_list = kb_ids.split(",") if kb_ids else None
    return knowledge_base.search_documents(query, kb_list, limit)


# ==========================================
# LEADS API ENDPOINTS
# ==========================================

@app.get("/api/leads")
async def list_leads(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    status: Optional[str] = None
):
    """List leads with pagination and filtering"""
    filters = {}
    if status:
        filters["status"] = status

    offset = (page - 1) * page_size
    leads, total = database.search_leads(
        filters,
        search=search,
        offset=offset,
        limit=page_size
    )

    return {
        "leads": leads,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@app.get("/api/leads/stats")
async def get_lead_stats():
    """Get lead statistics"""
    return database.get_lead_stats()


@app.post("/api/leads")
async def create_lead(lead: dict):
    """Create a new lead"""
    if not lead.get("phone") and not lead.get("email"):
        raise HTTPException(400, "Either phone number or email is required")

    try:
        lead_id = database.create_lead(lead)
        return {"id": lead_id, "status": "created"}
    except Exception as e:
        raise HTTPException(500, f"Failed to create lead: {str(e)}")


@app.get("/api/leads/{lead_id}")
async def get_lead(lead_id: int):
    """Get a single lead by ID"""
    lead = database.get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "Lead not found")
    return lead


@app.put("/api/leads/{lead_id}")
async def update_lead(lead_id: int, lead: dict):
    """Update a lead"""
    existing = database.get_lead(lead_id)
    if not existing:
        raise HTTPException(404, "Lead not found")

    try:
        database.update_lead(lead_id, lead)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(500, f"Failed to update lead: {str(e)}")


@app.delete("/api/leads/{lead_id}")
async def delete_lead(lead_id: int):
    """Delete a lead"""
    existing = database.get_lead(lead_id)
    if not existing:
        raise HTTPException(404, "Lead not found")

    try:
        database.delete_lead(lead_id)
        return {"status": "deleted"}
    except Exception as e:
        import traceback
        logger.error(f"Failed to delete lead {lead_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Failed to delete lead: {str(e)}")


@app.post("/api/leads/bulk-delete")
async def bulk_delete_leads(data: dict):
    """Delete multiple leads at once"""
    lead_ids = data.get("lead_ids", [])
    if not lead_ids:
        raise HTTPException(400, "No lead IDs provided")

    try:
        deleted_count = database.delete_leads_bulk(lead_ids)
        return {"status": "deleted", "deleted_count": deleted_count}
    except Exception as e:
        import traceback
        logger.error(f"Failed to bulk delete leads {lead_ids}: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Failed to delete leads: {str(e)}")


# =============================================================================
# Lead Lists API
# =============================================================================

@app.get("/api/lead-lists")
async def get_lead_lists():
    """Get all lead lists"""
    return database.get_lead_lists()


@app.post("/api/lead-lists")
async def create_lead_list(data: dict):
    """Create a new lead list"""
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(400, "List name is required")

    description = data.get("description", "")

    try:
        list_id = database.create_lead_list(name, description)
        return {"status": "created", "id": list_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to create list: {str(e)}")


@app.delete("/api/lead-lists/{list_id}")
async def delete_lead_list(list_id: int):
    """Delete a lead list"""
    try:
        database.delete_lead_list(list_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete list: {str(e)}")


@app.get("/api/lead-lists/{list_id}/leads")
async def get_list_leads(list_id: int):
    """Get leads in a list"""
    return database.get_list_leads(list_id)


@app.post("/api/lead-lists/{list_id}/leads")
async def add_leads_to_list(list_id: int, data: dict):
    """Add leads to a list"""
    lead_ids = data.get("lead_ids", [])
    if not lead_ids:
        raise HTTPException(400, "No lead IDs provided")

    try:
        database.add_leads_to_list(list_id, lead_ids)
        return {"status": "added", "count": len(lead_ids)}
    except Exception as e:
        raise HTTPException(500, f"Failed to add leads to list: {str(e)}")


@app.get("/api/leads/{lead_id}/interactions")
async def get_lead_interactions(lead_id: int):
    """Get interactions for a lead"""
    lead = database.get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "Lead not found")

    return database.get_lead_interactions(lead_id)


@app.post("/api/leads/import")
async def import_leads(data: dict):
    """Import leads from CSV data"""
    rows = data.get("data", [])
    mapping = data.get("mapping", {})

    if not rows:
        raise HTTPException(400, "No data to import")
    if "phone" not in mapping.values():
        raise HTTPException(400, "Phone field must be mapped")

    imported = 0
    skipped = 0
    errors = []

    for row in rows:
        try:
            # Map CSV columns to lead fields
            lead_data = {}
            for csv_col, db_field in mapping.items():
                if csv_col in row and row[csv_col]:
                    lead_data[db_field] = row[csv_col]

            # Skip if no phone
            if not lead_data.get("phone"):
                skipped += 1
                continue

            # Set source
            lead_data["source"] = "csv_import"

            database.create_lead(lead_data)
            imported += 1
        except Exception as e:
            skipped += 1
            errors.append(str(e))

    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors[:10]  # Limit error messages
    }


# ==========================================
# AGENTS API ENDPOINTS
# ==========================================

from agents import get_agent_manager, AgentManager
from pathlib import Path

# Agent manager instance
_agent_manager: AgentManager = None

def get_agents_manager():
    """Get or create agent manager"""
    global _agent_manager
    if _agent_manager is None:
        settings = load_settings()
        _agent_manager = get_agent_manager(settings)
    return _agent_manager


@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    manager = get_agents_manager()
    return manager.list_agents()


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get a single agent"""
    manager = get_agents_manager()
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    return agent.to_dict()


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, data: dict):
    """Update an agent"""
    manager = get_agents_manager()
    agent = manager.update_agent(agent_id, data)
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Save to settings
    settings = load_settings()
    if "agents" not in settings:
        settings["agents"] = {}

    settings["agents"][agent_id] = {
        "name": agent.name,
        "model_tier": agent.model_tier.value,
        "objective": agent.objective,
        "persona": agent.persona,
        "enabled": agent.enabled,
        "tools": agent.tools,
        "icon": agent.icon,
        "knowledge_base": agent.knowledge_base,
    }
    save_settings(settings)

    # Reload agent manager
    global _agent_manager
    _agent_manager = get_agent_manager(settings)

    return agent.to_dict()


@app.get("/api/agents/{agent_id}/knowledge")
async def get_agent_knowledge(agent_id: str):
    """Get agent's knowledge base content"""
    manager = get_agents_manager()
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    # Read knowledge base file
    kb_path = agent.knowledge_base
    if not kb_path:
        return {"content": ""}

    full_path = Path(__file__).parent / kb_path
    if full_path.exists():
        content = full_path.read_text()
        return {"content": content}

    return {"content": ""}


@app.put("/api/agents/{agent_id}/knowledge")
async def update_agent_knowledge(agent_id: str, data: dict):
    """Update agent's knowledge base content"""
    manager = get_agents_manager()
    agent = manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")

    content = data.get("content", "")

    # Ensure knowledge directory exists
    kb_dir = Path(__file__).parent / "knowledge"
    kb_dir.mkdir(exist_ok=True)

    # Set knowledge base path if not set
    if not agent.knowledge_base:
        agent.knowledge_base = f"knowledge/{agent_id}.md"

        # Save to settings
        settings = load_settings()
        if "agents" not in settings:
            settings["agents"] = {}
        if agent_id not in settings["agents"]:
            settings["agents"][agent_id] = {}
        settings["agents"][agent_id]["knowledge_base"] = agent.knowledge_base
        save_settings(settings)

    # Write content
    full_path = Path(__file__).parent / agent.knowledge_base
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)

    return {"status": "saved", "path": agent.knowledge_base}


# =============================================================================
# Email Account API Endpoints
# =============================================================================

@app.get("/api/email-accounts")
async def get_email_accounts():
    """List all email accounts"""
    from email_sender import get_email_manager
    manager = get_email_manager()
    return manager.get_stats()


@app.post("/api/email-accounts")
async def create_email_account(data: dict):
    """Create a new email account"""
    from email_sender import get_email_manager, EmailAccount, SMTP_PRESETS

    manager = get_email_manager()

    # Check if email already exists
    for existing in manager.accounts:
        if existing.email.lower() == data.get("email", "").lower():
            raise HTTPException(400, "Email account already exists")

    # Apply preset if provided
    preset = data.get("preset")
    if preset and preset in SMTP_PRESETS:
        preset_config = SMTP_PRESETS[preset]
        for key, value in preset_config.items():
            if key != "notes" and key not in data:
                data[key] = value

    # Create account
    account = EmailAccount(
        email=data.get("email", ""),
        display_name=data.get("display_name", ""),
        smtp_host=data.get("smtp_host", ""),
        smtp_port=data.get("smtp_port", 587),
        smtp_username=data.get("smtp_username", data.get("email", "")),
        smtp_password=data.get("smtp_password", ""),
        smtp_use_tls=data.get("smtp_use_tls", True),
        smtp_use_ssl=data.get("smtp_use_ssl", False),
        imap_host=data.get("imap_host", ""),
        imap_port=data.get("imap_port", 993),
        imap_username=data.get("imap_username", data.get("email", "")),
        imap_password=data.get("imap_password", data.get("smtp_password", "")),
        imap_use_ssl=data.get("imap_use_ssl", True),
        daily_limit=data.get("daily_limit", 100),
        hourly_limit=data.get("hourly_limit", 20),
        delay_between_emails_seconds=data.get("delay_between_emails_seconds", 60),
        warmup_enabled=data.get("warmup_enabled", False),
        warmup_target_volume=data.get("warmup_target_volume", 10),
        signature_html=data.get("signature_html", ""),
        signature_text=data.get("signature_text", ""),
    )

    account_id = manager.save_account(account)
    return {"id": account_id, "status": "created"}


@app.get("/api/email-accounts/presets")
async def get_email_presets():
    """Get available SMTP presets (Gmail, Outlook, etc.)"""
    from email_sender import SMTP_PRESETS
    return SMTP_PRESETS


@app.get("/api/email-accounts/{account_id}")
async def get_email_account(account_id: int):
    """Get a specific email account"""
    from email_sender import get_email_manager
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")
    return account.to_dict()


@app.put("/api/email-accounts/{account_id}")
async def update_email_account(account_id: int, data: dict):
    """Update an email account"""
    from email_sender import get_email_manager, EmailAccountStatus
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")

    # Update fields
    if "display_name" in data:
        account.display_name = data["display_name"]
    if "smtp_host" in data:
        account.smtp_host = data["smtp_host"]
    if "smtp_port" in data:
        account.smtp_port = data["smtp_port"]
    if "smtp_username" in data:
        account.smtp_username = data["smtp_username"]
    if "smtp_password" in data and data["smtp_password"] != "***":
        account.smtp_password = data["smtp_password"]
    if "smtp_use_tls" in data:
        account.smtp_use_tls = data["smtp_use_tls"]
    if "smtp_use_ssl" in data:
        account.smtp_use_ssl = data["smtp_use_ssl"]
    if "imap_host" in data:
        account.imap_host = data["imap_host"]
    if "imap_port" in data:
        account.imap_port = data["imap_port"]
    if "imap_username" in data:
        account.imap_username = data["imap_username"]
    if "imap_password" in data and data["imap_password"] != "***":
        account.imap_password = data["imap_password"]
    if "imap_use_ssl" in data:
        account.imap_use_ssl = data["imap_use_ssl"]
    if "daily_limit" in data:
        account.daily_limit = data["daily_limit"]
    if "hourly_limit" in data:
        account.hourly_limit = data["hourly_limit"]
    if "delay_between_emails_seconds" in data:
        account.delay_between_emails_seconds = data["delay_between_emails_seconds"]
    if "status" in data:
        account.status = EmailAccountStatus(data["status"])
    if "warmup_enabled" in data:
        account.warmup_enabled = data["warmup_enabled"]
    if "warmup_target_volume" in data:
        account.warmup_target_volume = data["warmup_target_volume"]
    if "signature_html" in data:
        account.signature_html = data["signature_html"]
    if "signature_text" in data:
        account.signature_text = data["signature_text"]

    manager.save_account(account)
    return {"status": "updated"}


@app.delete("/api/email-accounts/{account_id}")
async def delete_email_account(account_id: int):
    """Delete an email account"""
    from email_sender import get_email_manager
    manager = get_email_manager()
    if manager.delete_account(account_id):
        return {"status": "deleted"}
    raise HTTPException(404, "Email account not found")


@app.post("/api/email-accounts/{account_id}/test")
async def test_email_account(account_id: int):
    """Test SMTP connection for an email account"""
    from email_sender import get_email_manager
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")

    result = manager.test_connection(account)

    # Update account status based on test result
    if result["success"]:
        from email_sender import EmailAccountStatus
        if account.status == EmailAccountStatus.PENDING:
            account.status = EmailAccountStatus.ACTIVE
        elif account.status == EmailAccountStatus.ERROR:
            account.status = EmailAccountStatus.ACTIVE
        account.last_error = ""
    else:
        account.last_error = result.get("message", "Test failed")

    manager.save_account(account)
    return result


@app.post("/api/email-accounts/{account_id}/enable")
async def enable_email_account(account_id: int):
    """Re-enable a disabled email account"""
    from email_sender import get_email_manager, EmailAccountStatus
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")

    account.status = EmailAccountStatus.ACTIVE
    account.auto_disabled_at = None
    account.auto_disabled_reason = ""
    manager.save_account(account)
    return {"status": "enabled"}


@app.post("/api/email-accounts/{account_id}/warmup/start")
async def start_email_warmup(account_id: int):
    """Start warmup mode for an email account"""
    from email_sender import get_email_manager, EmailAccountStatus
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")

    account.warmup_enabled = True
    account.warmup_day = 0
    account.warmup_target_volume = 10
    account.status = EmailAccountStatus.WARMUP
    manager.save_account(account)
    return {"status": "warmup_started", "target_volume": 10}


@app.post("/api/email-accounts/{account_id}/warmup/stop")
async def stop_email_warmup(account_id: int):
    """Stop warmup mode for an email account"""
    from email_sender import get_email_manager, EmailAccountStatus
    manager = get_email_manager()
    account = manager.get_account(account_id)
    if not account:
        raise HTTPException(404, "Email account not found")

    account.warmup_enabled = False
    account.status = EmailAccountStatus.ACTIVE
    manager.save_account(account)
    return {"status": "warmup_stopped"}


@app.post("/api/email/send")
async def send_email(data: dict):
    """
    Send an email. Plain text by default, HTML optional.

    Required:
    - to_email: Recipient email address
    - subject: Email subject
    - body: Plain text body

    Optional:
    - body_html: HTML body (if provided, sends multipart)
    - variables: Dict of variables to substitute (e.g., {first_name: "John"})
    - lead_id: Associated lead ID
    - campaign_id: Associated campaign ID
    - account_id: Force use of specific email account
    - inject_tracking: Enable open/click tracking (default: false)
    """
    from email_sender import get_email_manager

    manager = get_email_manager()

    if not manager.accounts:
        raise HTTPException(400, "No email accounts configured")

    result = manager.send(
        to_email=data.get("to_email"),
        subject=data.get("subject"),
        body=data.get("body", ""),
        body_html=data.get("body_html", ""),
        variables=data.get("variables", {}),
        lead_id=data.get("lead_id"),
        campaign_id=data.get("campaign_id"),
        campaign_step_id=data.get("campaign_step_id"),
        account_id=data.get("account_id"),
        inject_tracking=data.get("inject_tracking", False),
        tracking_base_url=data.get("tracking_base_url")
    )

    if not result["success"]:
        raise HTTPException(400, result.get("error", "Failed to send email"))

    return result


@app.post("/api/email/send-test")
async def send_test_email(data: dict):
    """Send a test email to verify account is working (plain text)"""
    from email_sender import get_email_manager

    manager = get_email_manager()

    account_id = data.get("account_id")
    to_email = data.get("to_email")

    if not to_email:
        raise HTTPException(400, "to_email is required")

    result = manager.send(
        to_email=to_email,
        subject="Test Email from VersaBox",
        body=f"Test Email\n\nThis is a test email from your VersaBox AI Phone Agent.\n\nIf you received this, your email configuration is working correctly!\n\nSent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        account_id=account_id,
        inject_tracking=False,
        allow_pending=True  # Allow sending from pending accounts for testing
    )

    return result


# =============================================================================
# Email Tracking Endpoints (Public - no auth)
# =============================================================================

@app.get("/t/o/{tracking_id}")
async def track_email_open(tracking_id: str):
    """Track email open via 1x1 pixel"""
    from fastapi.responses import Response
    import database as db

    # Log the open event
    try:
        conn = db.get_db()
        cursor = conn.cursor()

        # Find the message by tracking_id
        cursor.execute("""
            UPDATE messages SET
                status = 'opened',
                read_at = datetime('now')
            WHERE external_id = ? AND status IN ('sent', 'delivered')
        """, (tracking_id,))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to track email open: {e}")

    # Return 1x1 transparent GIF
    gif_bytes = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
    return Response(content=gif_bytes, media_type="image/gif")


@app.get("/t/c/{tracking_id}/{url:path}")
async def track_email_click(tracking_id: str, url: str):
    """Track email click and redirect"""
    from fastapi.responses import RedirectResponse
    import urllib.parse
    import database as db

    # Decode URL
    decoded_url = urllib.parse.unquote(url)

    # Log the click event
    try:
        conn = db.get_db()
        cursor = conn.cursor()

        # Update message status to clicked
        cursor.execute("""
            UPDATE messages SET status = 'clicked'
            WHERE external_id = ? AND status IN ('sent', 'delivered', 'opened')
        """, (tracking_id,))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to track email click: {e}")

    return RedirectResponse(url=decoded_url)


@app.get("/t/unsub/{tracking_id}")
async def handle_unsubscribe(tracking_id: str):
    """Handle email unsubscribe"""
    from fastapi.responses import HTMLResponse
    import database as db

    try:
        conn = db.get_db()
        cursor = conn.cursor()

        # Find the message and get lead_id
        cursor.execute("""
            SELECT lead_id FROM messages WHERE external_id = ?
        """, (tracking_id,))
        row = cursor.fetchone()

        if row and row[0]:
            lead_id = row[0]
            # Mark lead as do_not_contact
            cursor.execute("""
                UPDATE leads SET
                    do_not_contact = 1,
                    status = 'UNSUBSCRIBED',
                    dnc_detected_at = datetime('now')
                WHERE id = ?
            """, (lead_id,))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to process unsubscribe: {e}")

    return HTMLResponse("""
    <html>
    <head><title>Unsubscribed</title></head>
    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
        <h1>You've been unsubscribed</h1>
        <p>You will no longer receive emails from us.</p>
    </body>
    </html>
    """)


def main():
    """Run the web server"""
    print("\n" + "=" * 60)
    print("AI Phone Agent - Web UI")
    print("=" * 60)
    print("\nOpen http://localhost in your browser")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=80)


if __name__ == "__main__":
    main()
