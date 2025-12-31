#!/bin/bash
# AI Phone Agent - Setup Script

set -e

echo "========================================"
echo "AI Phone Agent - Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Check for BlackHole
echo ""
echo "Checking for BlackHole audio driver..."
if system_profiler SPAudioDataType | grep -q "BlackHole"; then
    echo "✓ BlackHole is installed"
else
    echo "✗ BlackHole not found"
    echo ""
    echo "Installing BlackHole..."
    brew install blackhole-2ch || {
        echo ""
        echo "Please install BlackHole manually:"
        echo "  brew install blackhole-2ch"
        echo ""
        echo "After installation, you may need to:"
        echo "1. Open System Settings > Privacy & Security"
        echo "2. Allow the BlackHole kernel extension"
        echo "3. Restart your Mac"
    }
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key"
fi

# Create calls directory
mkdir -p calls

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Make sure iPhone Continuity is enabled"
echo "3. Run: source venv/bin/activate"
echo "4. Run: python web.py"
echo "5. Open http://localhost:8000"
echo ""
echo "Audio Setup (IMPORTANT):"
echo "For calls to work, you need to configure audio routing:"
echo "1. Open 'Audio MIDI Setup' app"
echo "2. Create a Multi-Output Device with:"
echo "   - BlackHole 2ch"
echo "   - Your speakers/headphones"
echo "3. Set FaceTime to use BlackHole for audio"
echo ""
