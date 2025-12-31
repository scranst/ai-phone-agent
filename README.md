# AI Phone Agent

A general-purpose AI agent that makes phone calls on your behalf using your Mac + iPhone.

## Features

- **Voice Conversations**: Uses OpenAI Realtime API for natural, low-latency conversations
- **Any Objective**: Order pizza, schedule appointments, call customer service, etc.
- **Web UI**: Simple browser interface to configure and monitor calls
- **Live Transcript**: See the conversation as it happens
- **Call Recording**: Saves recordings and logs of all calls
- **Hold/Transfer Handling**: Waits through hold music and re-introduces when transferred

## Prerequisites

1. **macOS** with iPhone Continuity enabled
2. **iPhone** connected to same Apple ID (for making calls)
3. **OpenAI API Key** with Realtime API access
4. **BlackHole** audio driver (free)

## Quick Start

```bash
# 1. Install BlackHole (requires password)
brew install blackhole-2ch

# 2. Set up the project
cd ai-phone-agent
./setup.sh

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 4. Activate virtual environment
source venv/bin/activate

# 5. Start the web UI
python web.py

# 6. Open http://localhost:8000
```

## Audio Setup (IMPORTANT)

For the AI to talk on phone calls, you need to route audio through BlackHole:

1. Open **Audio MIDI Setup** app (search in Spotlight)
2. Click **+** → **Create Multi-Output Device**
3. Check both:
   - BlackHole 2ch
   - Your headphones/speakers
4. In **FaceTime** preferences, set audio output to "BlackHole 2ch"

This routes FaceTime audio through BlackHole so the AI can hear and speak on calls.

## Usage

### Web UI (Recommended)

1. Run `python web.py`
2. Open http://localhost:8000
3. Enter:
   - **Phone number** to call
   - **Objective** (what you want to accomplish)
   - **Context** (any info the AI needs - name, address, etc.)
4. Click **Start Call**
5. Watch the live transcript

### CLI

```bash
python agent.py "775-555-1234" "Order a large pepperoni pizza" \
    -c name "Scott" \
    -c address "123 Main St, Reno NV"
```

### Python API

```python
from agent import PhoneAgent, CallRequest

agent = PhoneAgent()

result = await agent.call(CallRequest(
    phone="775-555-1234",
    objective="Schedule an electrician for outlet installation",
    context={
        "name": "Scott Stevenson",
        "business": "My Store",
        "address": "123 Commerce Dr, Reno NV",
        "scope": "Install 4 new 220V outlets and electrical dropdown"
    }
))

print(result.summary)
print(result.collected_info)
```

## Project Structure

```
ai-phone-agent/
├── agent.py              # Main orchestrator
├── call_controller.py    # FaceTime/Continuity control
├── audio_router.py       # BlackHole audio routing
├── conversation.py       # OpenAI Realtime API
├── web.py                # Web UI server
├── config.py             # Configuration
├── calls/                # Call recordings & logs
├── contacts/             # Contact lists
└── prompts/              # System prompts
```

## Troubleshooting

### "BlackHole not found"
Install with: `brew install blackhole-2ch`
May require restart after installation.

### "Unauthenticated" from OpenAI
Make sure your `.env` file contains a valid `OPENAI_API_KEY`.

### Call doesn't connect
- Ensure iPhone is on same WiFi as Mac
- Ensure iPhone Continuity is enabled in FaceTime settings
- Try making a manual FaceTime call first

### AI can't hear caller / Caller can't hear AI
- Check Audio MIDI Setup configuration
- Ensure FaceTime is using BlackHole for audio
- Check volume levels on BlackHole device

## Included Contacts

`contacts/reno_electricians.json` - List of electrical contractors in Reno, NV with phone numbers.
