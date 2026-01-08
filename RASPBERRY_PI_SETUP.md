# Raspberry Pi 5 Setup Guide

Complete guide to running the AI Phone Agent on a Raspberry Pi 5.

## Hardware Required

- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- Official Raspberry Pi 27W USB-C Power Supply
- MicroSD card (32GB+ recommended, Class 10 or faster)
- SIM7600 USB Modem
- USB Audio Device (for modem audio routing)
- SIM card with voice/SMS plan
- Ethernet cable OR WiFi credentials

## 1. Initial Pi Setup

### Flash the OS (on your Mac)

Download Raspberry Pi Imager from https://www.raspberrypi.com/software/
Or install via Homebrew:
```bash
brew install --cask raspberry-pi-imager
```

In the Imager:
1. Choose Device: **Raspberry Pi 5**
2. Choose OS: **Raspberry Pi OS (64-bit)** - Lite version is fine
3. Choose Storage: Your SD card
4. Click the gear icon (settings) to pre-configure:
   - Enable SSH (Use password authentication)
   - Set username: `pi`
   - Set password: (choose something secure)
   - Configure WiFi (SSID and password)
   - Set hostname: `phoneagent`
   - Set timezone

Flash the SD card and wait for completion.

### First Boot

1. Insert SD card into Pi
2. Connect ethernet (optional but recommended for first setup)
3. Connect power - Pi will boot automatically
4. Wait 2-3 minutes for first boot to complete

### Connect via SSH

From your Mac terminal:
```bash
ssh pi@phoneagent.local
```

If `.local` doesn't resolve, find the IP:
- Check your router's admin page, or
- Use: `ping phoneagent.local`

## 2. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y python3-pip python3-venv python3-dev \
    libportaudio2 portaudio19-dev ffmpeg \
    libsndfile1 libopenblas-dev git

# For USB serial (SIM7600)
sudo apt install -y libusb-1.0-0-dev

# Reboot after updates
sudo reboot
```

## 3. Transfer Code from Mac

On your Mac:
```bash
# Create archive (excluding venv and cache files)
cd ~/ai-phone-agent
tar -czf ../ai-phone-agent.tar.gz \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='calls/*.wav' \
    .

# Copy to Pi
scp ../ai-phone-agent.tar.gz pi@phoneagent.local:~/
```

On the Pi:
```bash
# Extract
mkdir ~/ai-phone-agent
cd ~/ai-phone-agent
tar -xzf ../ai-phone-agent.tar.gz

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Some packages may take a while to compile on Pi.

## 4. Connect Hardware

### SIM7600 Modem
- Connect USB cable from SIM7600 to any Pi USB 3.0 port (blue ports)
- Insert SIM card into modem
- Attach antennas to modem

### USB Audio Device
- Connect to another USB port

### Verify Connections

```bash
# Check USB devices
lsusb
# Should see: SimTech, Incorporated
# Should see: USB Audio Device

# Check serial ports
ls /dev/ttyUSB*
# Should see: /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyUSB2

# Check audio devices
arecord -l
aplay -l
```

## 5. Configure Environment

```bash
cd ~/ai-phone-agent

# Set your Anthropic API key
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# Or create a .env file if your app uses python-dotenv
echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' > .env
```

## 6. Configure Audio Device

Check your audio device numbers:
```bash
arecord -l
aplay -l
```

If device IDs differ from Mac, update `audio_router_sim7600.py`:
```python
# Find the line that sets the device and update the index
```

## 7. Optimize for Pi (Optional)

For faster startup, use smaller Whisper model. Edit `conversation_local.py`:
```python
self.stt = SpeechToText(
    model_size="tiny.en",  # Changed from small.en
    device="cpu",
    compute_type="int8"
)
```

## 8. Test the Setup

```bash
cd ~/ai-phone-agent
source venv/bin/activate

# Test modem connection
python -c "from sim7600_modem import SIM7600Modem; m = SIM7600Modem(); print('Connected!' if m.connect() else 'Failed')"

# Run the web server
python web.py
```

Access from any device on your network: `http://phoneagent.local:8765`

## 9. Run as a Service (Auto-start on Boot)

Create systemd service:
```bash
sudo nano /etc/systemd/system/phoneagent.service
```

Paste this content:
```ini
[Unit]
Description=AI Phone Agent
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ai-phone-agent
Environment=PATH=/home/pi/ai-phone-agent/venv/bin
Environment=ANTHROPIC_API_KEY=sk-ant-your-key-here
ExecStart=/home/pi/ai-phone-agent/venv/bin/python web.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable phoneagent
sudo systemctl start phoneagent

# Check status
sudo systemctl status phoneagent

# View logs
journalctl -u phoneagent -f
```

## 10. Useful Commands

```bash
# SSH into Pi
ssh pi@phoneagent.local

# Check if service is running
sudo systemctl status phoneagent

# Restart service
sudo systemctl restart phoneagent

# View live logs
journalctl -u phoneagent -f

# Stop service
sudo systemctl stop phoneagent

# Check disk space
df -h

# Check memory
free -h

# Check CPU temperature
vcgencmd measure_temp
```

## Troubleshooting

### Can't connect via SSH
- Make sure Pi is on the same network
- Try using IP address instead of hostname
- Check if SSH is enabled in Pi Imager settings

### Modem not detected
- Check USB connection
- Try different USB port
- Run `dmesg | tail -20` to see USB events

### Audio issues
- Run `arecord -l` and `aplay -l` to verify device detection
- Check device indices in audio_router_sim7600.py

### Slow performance
- Use `tiny.en` Whisper model instead of `small.en`
- Ensure good ventilation for Pi (consider a heatsink/fan)
- Check CPU temperature: `vcgencmd measure_temp`

### Service won't start
- Check logs: `journalctl -u phoneagent -n 50`
- Verify paths in service file
- Make sure venv exists and has all packages
