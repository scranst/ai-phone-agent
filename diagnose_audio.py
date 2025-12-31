#!/usr/bin/env python3
"""
Audio diagnostic - run this during a call to see what's flowing
"""
import sounddevice as sd
import numpy as np
import time
import sys

def find_device(name):
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d['name'].lower():
            return i
    return None

bh2 = find_device("BlackHole 2ch")
bh16 = find_device("BlackHole 16ch")

print("=" * 50)
print("AUDIO DIAGNOSTIC - Run during a call")
print("=" * 50)
print(f"\nBlackHole 2ch:  index {bh2}")
print(f"BlackHole 16ch: index {bh16}")
print(f"\nSystem Input:  {sd.query_devices(kind='input')['name']}")
print(f"System Output: {sd.query_devices(kind='output')['name']}")
print("\nMonitoring audio levels (Ctrl+C to stop)...")
print("-" * 50)
print("BH16 = Call audio TO AI (should see levels when caller speaks)")
print("BH2  = AI voice TO caller (should see levels when AI speaks)")
print("-" * 50)

try:
    while True:
        # Sample from both devices
        bh16_audio = sd.rec(int(48000 * 0.1), samplerate=48000, channels=1,
                            device=bh16, dtype=np.float32)
        sd.wait()
        bh16_level = np.max(np.abs(bh16_audio))

        bh2_audio = sd.rec(int(48000 * 0.1), samplerate=48000, channels=1,
                           device=bh2, dtype=np.float32)
        sd.wait()
        bh2_level = np.max(np.abs(bh2_audio))

        bh16_bar = "█" * int(bh16_level * 50)
        bh2_bar = "█" * int(bh2_level * 50)

        print(f"\rBH16: {bh16_level:5.3f} |{bh16_bar:<25}| BH2: {bh2_level:5.3f} |{bh2_bar:<25}|", end="", flush=True)

except KeyboardInterrupt:
    print("\n\nDone.")
