"""
Call Controller - Manages phone calls via macOS FaceTime/Continuity

Uses AppleScript to:
- Initiate calls through iPhone via Continuity
- Monitor call state
- End calls
"""

import subprocess
import asyncio
import logging
import re
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class CallState(Enum):
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    CONNECTED = "connected"
    ON_HOLD = "on_hold"
    ENDED = "ended"
    FAILED = "failed"


@dataclass
class CallInfo:
    phone_number: str
    state: CallState
    start_time: Optional[float] = None
    connect_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None


class CallController:
    def __init__(self):
        self.current_call: Optional[CallInfo] = None
        self._state_callbacks: list[Callable[[CallState], None]] = []
        self._monitor_task: Optional[asyncio.Task] = None

    def on_state_change(self, callback: Callable[[CallState], None]):
        """Register a callback for call state changes"""
        self._state_callbacks.append(callback)

    def _notify_state_change(self, new_state: CallState):
        """Notify all callbacks of state change"""
        if self.current_call:
            self.current_call.state = new_state
        for callback in self._state_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _run_applescript(self, script: str) -> tuple[bool, str]:
        """Run AppleScript and return (success, output)"""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    async def start_call(self, phone_number: str) -> bool:
        """
        Initiate a phone call via FaceTime/Continuity

        Args:
            phone_number: The phone number to call (will be cleaned)

        Returns:
            True if call was initiated successfully
        """
        # Clean the phone number
        clean_number = "".join(c for c in phone_number if c.isdigit() or c == "+")

        logger.info(f"Starting call to {clean_number}")

        self.current_call = CallInfo(
            phone_number=clean_number,
            state=CallState.DIALING,
            start_time=time.time()
        )
        self._notify_state_change(CallState.DIALING)

        # AppleScript to initiate call via FaceTime and auto-click Call button
        script = f'''
        tell application "FaceTime"
            activate
        end tell
        delay 0.5
        open location "tel:{clean_number}"
        delay 1.5

        -- Auto-click the Call button
        tell application "System Events"
            tell process "FaceTime"
                -- Look for the Call button and click it
                try
                    click button "Call" of window 1
                on error
                    -- Try clicking the first button if "Call" not found by name
                    try
                        click button 1 of window 1
                    end try
                end try
            end tell
        end tell
        '''

        success, output = self._run_applescript(script)

        if not success:
            logger.error(f"Failed to initiate call: {output}")
            self._notify_state_change(CallState.FAILED)
            return False

        # Start monitoring call state
        self._monitor_task = asyncio.create_task(self._monitor_call_state())

        return True

    async def _monitor_call_state(self):
        """Monitor the call state by checking FaceTime status"""
        # Give FaceTime time to start the call
        await asyncio.sleep(2)
        self._notify_state_change(CallState.RINGING)

        # Poll for call connection
        # Note: Detecting exact call state from FaceTime is tricky
        # We'll use a simplified approach and rely on audio detection
        check_interval = 1.0
        max_ring_time = 60  # Max time to wait for answer
        ring_start = time.time()

        while self.current_call and self.current_call.state == CallState.RINGING:
            await asyncio.sleep(check_interval)

            # Check if call was answered using multiple heuristics
            # Look for indicators of an active call vs still ringing
            script = '''
            tell application "System Events"
                tell process "FaceTime"
                    if exists (window 1) then
                        set windowTitle to name of window 1
                        set buttonNames to ""
                        try
                            set buttonNames to name of every button of window 1
                        end try
                        -- Check for static text that might show duration
                        set staticTexts to ""
                        try
                            set staticTexts to value of every static text of window 1
                        end try
                        return windowTitle & "|BUTTONS|" & buttonNames & "|TEXTS|" & staticTexts
                    end if
                end tell
            end tell
            return "none"
            '''
            success, window_info = self._run_applescript(script)

            if success and window_info != "none":
                # Log window info for debugging
                logger.debug(f"FaceTime window: {window_info}")

                # Multiple ways to detect connected state:
                # 1. Window title contains the phone number (original check)
                # 2. There's an "End" button visible
                # 3. There's a timer showing (like "0:05")
                # 4. The "Call" button is gone
                window_lower = window_info.lower()

                is_connected = False

                # Check if phone number in title
                if self.current_call.phone_number[-4:] in window_info:
                    is_connected = True
                    logger.debug("Connected: phone number in window title")

                # Check for "End" button (appears during active call)
                if "end" in window_lower:
                    is_connected = True
                    logger.debug("Connected: End button found")

                # Check for timer format (like "0:05" or "1:23")
                if re.search(r'\d+:\d{2}', window_info):
                    is_connected = True
                    logger.debug("Connected: Timer visible")

                # Check that Call button is NOT present (it disappears when connected)
                if "|BUTTONS|" in window_info:
                    buttons_part = window_info.split("|BUTTONS|")[1].split("|TEXTS|")[0]
                    if "call" not in buttons_part.lower() and buttons_part.strip():
                        is_connected = True
                        logger.debug("Connected: Call button gone")

                if is_connected:
                    self.current_call.connect_time = time.time()
                    self._notify_state_change(CallState.CONNECTED)
                    logger.info("Call connected")
                    break

            # FALLBACK: If ringing for 15+ seconds, assume connected
            # This handles cases where FaceTime window detection fails
            ring_duration = time.time() - ring_start
            if ring_duration > 15:
                logger.info(f"Assuming connected after {ring_duration:.1f}s of ringing (fallback)")
                self.current_call.connect_time = time.time()
                self._notify_state_change(CallState.CONNECTED)
                logger.info("Call connected (fallback)")
                break

            # Timeout check
            if time.time() - ring_start > max_ring_time:
                logger.warning("Call ring timeout")
                self._notify_state_change(CallState.FAILED)
                break

        # Continue monitoring while connected
        # Give the call some time before checking if it ended
        await asyncio.sleep(5)  # Don't check for end immediately

        consecutive_ended_checks = 0
        while self.current_call and self.current_call.state == CallState.CONNECTED:
            await asyncio.sleep(check_interval)

            # Check if FaceTime process is still running
            script = '''
            tell application "System Events"
                if exists process "FaceTime" then
                    return "running"
                end if
            end tell
            return "not_running"
            '''
            success, status = self._run_applescript(script)

            # Only end if FaceTime process is completely gone (not just window)
            # Require multiple consecutive "not running" checks to avoid false positives
            if not success or status == "not_running":
                consecutive_ended_checks += 1
                if consecutive_ended_checks >= 3:  # 3 consecutive checks = really ended
                    logger.info("FaceTime process ended - call terminated")
                    self._end_call_internal()
                    break
            else:
                consecutive_ended_checks = 0  # Reset if FaceTime is running

    def _end_call_internal(self):
        """Internal call ending logic"""
        if self.current_call:
            self.current_call.end_time = time.time()
            if self.current_call.connect_time:
                self.current_call.duration = (
                    self.current_call.end_time - self.current_call.connect_time
                )
            self._notify_state_change(CallState.ENDED)
            logger.info(f"Call ended. Duration: {self.current_call.duration:.1f}s"
                       if self.current_call.duration else "Call ended")

    async def end_call(self):
        """End the current call"""
        if not self.current_call:
            return

        logger.info("Ending call")

        # Cancel monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # AppleScript to end the call
        script = '''
        tell application "FaceTime"
            -- Click the end call button
        end tell
        tell application "System Events"
            tell process "FaceTime"
                -- Try to find and click end button
                if exists (button 1 of window 1) then
                    click button 1 of window 1
                end if
            end tell
        end tell
        -- Fallback: just quit FaceTime
        tell application "FaceTime" to quit
        '''

        self._run_applescript(script)
        self._end_call_internal()

    def set_on_hold(self, on_hold: bool = True):
        """Mark the call as on hold (detected externally via audio)"""
        if self.current_call and self.current_call.state == CallState.CONNECTED:
            if on_hold:
                self._notify_state_change(CallState.ON_HOLD)
            else:
                self._notify_state_change(CallState.CONNECTED)

    def get_call_info(self) -> Optional[CallInfo]:
        """Get current call information"""
        return self.current_call


# Test function
async def test_call_controller():
    controller = CallController()

    def on_state(state):
        print(f"Call state: {state.value}")

    controller.on_state_change(on_state)

    # Don't actually make a call in test - just verify it initializes
    print("Call controller initialized successfully")
    print("To test: await controller.start_call('555-123-4567')")


if __name__ == "__main__":
    asyncio.run(test_call_controller())
