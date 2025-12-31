"""
SIM7600 Modem Controller - Direct USB communication for voice calls

Uses pyusb to communicate with the SIM7600G-H modem via USB,
bypassing the need for serial port drivers on macOS.

Supports:
- SIM7600G-H USB dongles (various PIDs)
- SIM7600G-H 4G HAT (B) with audio (Waveshare)
"""

import usb.core
import usb.util
import time
import logging
import threading
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CallState(Enum):
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"  # Outgoing call, waiting for answer
    INCOMING = "incoming"  # Incoming call
    CONNECTED = "connected"
    ENDED = "ended"
    FAILED = "failed"


@dataclass
class CallInfo:
    phone_number: str
    state: CallState
    start_time: Optional[float] = None
    connect_time: Optional[float] = None
    end_time: Optional[float] = None
    direction: str = "outgoing"  # or "incoming"


class SIM7600Modem:
    """Controller for SIM7600G-H USB modem"""

    # USB IDs for SIM7600 (SIMCOM vendor)
    VENDOR_ID = 0x1e0e

    # Known Product IDs:
    # 0x9001 - Standard mode (no audio)
    # 0x9011 - RNDIS + audio mode
    # 0x9025 - ECM mode
    PRODUCT_IDS = [0x9001, 0x9011, 0x9025]

    # Default endpoints for PID 0x9001
    # Interface 2 is typically the AT command port
    AT_INTERFACE = 2
    AT_EP_IN = 0x84
    AT_EP_OUT = 0x03

    # For PID 0x9011 (with audio), AT is on interface 4
    AT_INTERFACE_9011 = 4
    AT_EP_IN_9011 = 0x86
    AT_EP_OUT_9011 = 0x04

    # Audio interface (not used - audio is via 3.5mm jack on HAT)
    AUDIO_INTERFACE = 5
    AUDIO_EP_IN = 0x8a
    AUDIO_EP_OUT = 0x06

    def __init__(self):
        self.dev: Optional[usb.core.Device] = None
        self.current_call: Optional[CallInfo] = None
        self._state_callbacks: list[Callable[[CallState], None]] = []
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._product_id: Optional[int] = None
        self._at_interface: int = self.AT_INTERFACE
        self._at_ep_in: int = self.AT_EP_IN
        self._at_ep_out: int = self.AT_EP_OUT

    def connect(self) -> bool:
        """Connect to the SIM7600 modem"""
        # Try each known Product ID
        for pid in self.PRODUCT_IDS:
            self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=pid)
            if self.dev is not None:
                self._product_id = pid
                break

        if self.dev is None:
            logger.error("SIM7600 modem not found")
            logger.info("Make sure the modem is connected and powered")
            return False

        logger.info(f"Found SIM7600 (PID 0x{self._product_id:04x}): {self.dev.manufacturer} - {self.dev.product}")

        # Select correct endpoints based on PID
        if self._product_id == 0x9011:
            self._at_interface = self.AT_INTERFACE_9011
            self._at_ep_in = self.AT_EP_IN_9011
            self._at_ep_out = self.AT_EP_OUT_9011
            logger.info("Using RNDIS+Audio mode endpoints")
        else:
            self._at_interface = self.AT_INTERFACE
            self._at_ep_in = self.AT_EP_IN
            self._at_ep_out = self.AT_EP_OUT

        try:
            # Detach kernel drivers if any
            for i in range(8):
                try:
                    if self.dev.is_kernel_driver_active(i):
                        self.dev.detach_kernel_driver(i)
                        logger.debug(f"Detached kernel driver from interface {i}")
                except:
                    pass

            # Set configuration
            self.dev.set_configuration()

            # Claim AT command interface
            usb.util.claim_interface(self.dev, self._at_interface)
            logger.info(f"Connected to SIM7600 modem (interface {self._at_interface})")

            # Check if SIM is ready
            response = self._send_at("AT+CPIN?")
            if "READY" not in response:
                logger.error(f"SIM not ready: {response}")
                return False

            # Get signal strength
            response = self._send_at("AT+CSQ")
            logger.info(f"Signal: {response.strip()}")

            # Start monitoring thread
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to modem: {e}")
            return False

    def disconnect(self):
        """Disconnect from the modem"""
        self._running = False

        # Stop monitor thread
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
            self._monitor_thread = None

        # Reset call state
        self.current_call = None

        if self.dev:
            try:
                # Try to hang up any active call
                self._send_at("AT+CHUP", timeout=1000)
            except:
                pass

            try:
                usb.util.release_interface(self.dev, self._at_interface)
            except:
                pass

            try:
                usb.util.dispose_resources(self.dev)
            except:
                pass

            try:
                # Reset the device
                self.dev.reset()
            except:
                pass

            self.dev = None

        # Give USB subsystem time to fully release
        time.sleep(1.5)
        logger.info("Disconnected from modem")

    def _send_at(self, cmd: str, timeout: int = 2000) -> str:
        """Send AT command and return response"""
        if not self.dev:
            return "ERROR: Not connected"

        with self._lock:
            try:
                cmd_bytes = (cmd + "\r\n").encode()
                self.dev.write(self._at_ep_out, cmd_bytes, timeout)
                time.sleep(0.1)

                response = bytes()
                for _ in range(20):
                    try:
                        data = self.dev.read(self._at_ep_in, 512, timeout=200)
                        response += bytes(data)
                        if b"OK" in response or b"ERROR" in response:
                            break
                    except usb.core.USBTimeoutError:
                        break

                return response.decode('utf-8', errors='replace')

            except Exception as e:
                logger.error(f"AT command error: {e}")
                return f"ERROR: {e}"

    def on_state_change(self, callback: Callable[[CallState], None]):
        """Register callback for call state changes"""
        self._state_callbacks.append(callback)

    def on_audio(self, callback: Callable[[bytes], None]):
        """Register callback for incoming audio data"""
        self._audio_callback = callback

    def _notify_state(self, state: CallState):
        """Notify all callbacks of state change"""
        if self.current_call:
            self.current_call.state = state
        for cb in self._state_callbacks:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def dial(self, phone_number: str) -> bool:
        """Initiate a voice call"""
        # Clean phone number
        clean_number = "".join(c for c in phone_number if c.isdigit() or c == "+")

        logger.info(f"Dialing {clean_number}")

        self.current_call = CallInfo(
            phone_number=clean_number,
            state=CallState.DIALING,
            start_time=time.time(),
            direction="outgoing"
        )
        self._notify_state(CallState.DIALING)

        # Enable audio channel (headphone mode for 3.5mm jack)
        response = self._send_at("AT+CSDVC=1")
        logger.info(f"Audio routing to headset: {response.strip()}")

        # Set volume to 2 (avoid clipping - max was causing distortion)
        response = self._send_at("AT+CLVL=1")
        logger.info(f"Volume set: {response.strip()}")

        # Enable echo suppression for better audio quality
        response = self._send_at("AT+CECM=1")
        logger.info(f"Echo suppression: {response.strip()}")

        # TDD noise suppression
        self._send_at("AT^PWRCTL=0,1,3")

        # Dial
        response = self._send_at(f"ATD{clean_number};", timeout=5000)

        if "OK" in response:
            logger.info("Call initiated")
            self._notify_state(CallState.RINGING)
            return True
        else:
            logger.error(f"Dial failed: {response}")
            self._notify_state(CallState.FAILED)
            return False

    def answer(self) -> bool:
        """Answer an incoming call"""
        response = self._send_at("ATA")
        if "OK" in response:
            if self.current_call:
                self.current_call.connect_time = time.time()
            self._notify_state(CallState.CONNECTED)
            return True
        return False

    def hangup(self) -> bool:
        """End the current call"""
        response = self._send_at("AT+CHUP")
        if self.current_call:
            self.current_call.end_time = time.time()
        self._notify_state(CallState.ENDED)
        return "OK" in response

    def send_dtmf(self, digit: str):
        """Send DTMF tone"""
        self._send_at(f"AT+VTS={digit}")

    def hold_call(self) -> bool:
        """Put current call on hold"""
        # Check current call state first
        clcc = self._send_at("AT+CLCC")
        logger.info(f"Current calls before hold: {clcc.strip()}")

        # Try standard CHLD command first
        response = self._send_at("AT+CHLD=2")
        logger.info(f"Hold response (raw): [{response}]")

        if "OK" in response:
            logger.info("Call placed on hold")
            return True

        # Some modems need CHLD=2,0 format
        response = self._send_at("AT+CHLD=2,0")
        logger.info(f"Hold (alt) response (raw): [{response}]")

        if "OK" in response:
            logger.info("Call placed on hold (alt command)")
            return True

        # Try AT+CHLD=? to see supported options
        supported = self._send_at("AT+CHLD=?")
        logger.info(f"Supported CHLD options: {supported.strip()}")

        logger.error(f"Failed to hold call")
        return False

    def resume_call(self) -> bool:
        """Resume held call"""
        response = self._send_at("AT+CHLD=2")
        if "OK" in response:
            logger.info("Call resumed")
            return True
        return False

    def conference_calls(self) -> bool:
        """Join held call with active call (3-way conference)"""
        # Try standard conference command
        response = self._send_at("AT+CHLD=3")
        logger.info(f"Conference response: {response.strip()}")

        if "OK" in response and "END" not in response:
            logger.info("Calls conferenced together")
            return True

        # Try alternative: CHLD=3,0 (join all calls)
        response = self._send_at("AT+CHLD=3,0")
        logger.info(f"Conference (alt1) response: {response.strip()}")

        if "OK" in response and "END" not in response:
            logger.info("Calls conferenced (alt1)")
            return True

        # Try multiparty call command
        response = self._send_at("AT+CHLD=4")
        logger.info(f"Conference (alt2 CHLD=4) response: {response.strip()}")

        if "OK" in response and "END" not in response:
            logger.info("Calls conferenced (multiparty)")
            return True

        # Check if any response contains "END" which means call dropped
        if "END" in response:
            logger.error("Conference failed - call ended")
            return False

        logger.error(f"Failed to conference calls")
        return False

    def check_supplementary_services(self) -> dict:
        """Check what supplementary services are supported by the network"""
        services = {}

        # Check call waiting
        resp = self._send_at("AT+CCWA?")
        services['call_waiting'] = "OK" in resp
        logger.info(f"Call waiting: {resp.strip()}")

        # Check call hold
        resp = self._send_at("AT+CHLD=?")
        services['call_hold'] = resp
        logger.info(f"Call hold options: {resp.strip()}")

        # Check ECT (Explicit Call Transfer) support
        resp = self._send_at("AT+CHLD=4")  # ECT is sometimes CHLD=4
        logger.info(f"ECT via CHLD=4: {resp.strip()}")

        # Check if AT+CTFR is supported
        resp = self._send_at("AT+CTFR=?")
        services['ctfr'] = "OK" in resp or "ERROR" not in resp
        logger.info(f"CTFR support: {resp.strip()}")

        # Check supplementary service notifications
        resp = self._send_at("AT+CSSN=1,1")
        logger.info(f"CSSN: {resp.strip()}")

        return services

    def explicit_call_transfer(self, phone_number: str) -> bool:
        """
        Try Explicit Call Transfer (ECT) - direct transfer without conference.
        This is the 3GPP standard way to transfer calls.
        """
        clean_number = "".join(c for c in phone_number if c.isdigit() or c == "+")

        # Method 1: AT+CTFR (Call Transfer)
        response = self._send_at(f"AT+CTFR=\"{clean_number}\"")
        logger.info(f"CTFR response: {response.strip()}")
        if "OK" in response:
            logger.info("ECT via CTFR successful")
            return True

        # Method 2: AT+CTFR without quotes
        response = self._send_at(f"AT+CTFR={clean_number}")
        logger.info(f"CTFR (no quotes) response: {response.strip()}")
        if "OK" in response:
            logger.info("ECT via CTFR successful")
            return True

        # Method 3: Blind transfer via ATD with special prefix
        # Some modems support ATD>number for transfer
        response = self._send_at(f"ATD>{clean_number};", timeout=3000)
        logger.info(f"ATD> transfer response: {response.strip()}")
        if "OK" in response:
            logger.info("Blind transfer via ATD> successful")
            return True

        return False

    def transfer_to(self, phone_number: str, announce_message: str = None) -> bool:
        """
        Transfer current call to another number.

        Tries multiple methods:
        1. Explicit Call Transfer (ECT) - cleanest method
        2. Hold + Dial + Conference (3-way calling)

        Args:
            phone_number: Number to transfer to
            announce_message: Optional message to speak before transfer

        Returns:
            True if transfer successful
        """
        clean_number = "".join(c for c in phone_number if c.isdigit() or c == "+")
        logger.info(f"Initiating transfer to {clean_number}")

        # First, check what services are available
        self.check_supplementary_services()

        # Enable call waiting and supplementary service notifications
        self._send_at("AT+CCWA=1")
        self._send_at("AT+CSSN=1,1")  # Enable SS notifications
        time.sleep(0.2)

        # Method 1: Try ECT first (cleanest transfer)
        logger.info("Attempting Explicit Call Transfer (ECT)...")
        if self.explicit_call_transfer(clean_number):
            return True

        logger.info("ECT not supported, trying 3-way calling method...")

        # Method 2: 3-way calling (no call waiting required)
        # On carriers like Tello that support 3-way but NOT call waiting,
        # we need to dial the second number directly without holding first.

        # Try to dial second number while first call is active
        # The modem should automatically manage the first call
        logger.info("Dialing second number for 3-way call (no hold)...")

        # Step 1: Dial transfer target directly (skip hold)
        response = self._send_at(f"ATD{clean_number};", timeout=5000)
        if "OK" not in response:
            logger.error(f"Transfer failed: Could not dial {clean_number}")
            self.resume_call()  # Try to resume original call
            return False

        logger.info(f"Dialing transfer target {clean_number}...")

        # Step 2: Wait for answer (up to 30 seconds)
        answered = False
        for _ in range(60):
            time.sleep(0.5)
            response = self._send_at("AT+CLCC")
            logger.debug(f"CLCC during transfer: {response.strip()}")

            # Look for two calls - one held (stat=1) and one active (stat=0)
            if response.count("+CLCC:") >= 2:
                # Check if second call is answered
                lines = response.split("+CLCC:")
                for line in lines[1:]:
                    fields = line.split(",")
                    if len(fields) >= 3:
                        stat = int(fields[2].strip())
                        if stat == 0:  # Active call found
                            answered = True
                            break
                if answered:
                    break

        if not answered:
            logger.error("Transfer failed: Target did not answer")
            self.resume_call()  # Resume original call
            return False

        logger.info("Transfer target answered")
        time.sleep(1.0)  # Let call stabilize

        # Step 3: Merge calls (3-way conference)
        conference_methods = [
            ("AT+CHLD=3", "standard conference"),
            ("AT+CHLD=3,0", "conference variant"),
            ("AT+CHLD=4", "multiparty/ECT"),
            ("AT+CHLD=2", "swap then conference"),
        ]

        for cmd, desc in conference_methods:
            response = self._send_at(cmd)
            logger.info(f"{desc} ({cmd}): {response.strip()}")

            if "OK" in response and "END" not in response:
                # Verify conference worked
                time.sleep(0.5)
                clcc = self._send_at("AT+CLCC")
                logger.info(f"Call state after {desc}: {clcc.strip()}")

                # Check for multiparty flag (mpty=1)
                if "1,\"" in clcc or ",1\n" in clcc:
                    logger.info(f"Transfer complete via {desc}")
                    return True

        # Last resort: Try to merge calls by dialing again while connected
        logger.info("Trying last resort: merge via redial...")
        response = self._send_at("AT+CHLD=1")  # Release held call
        logger.info(f"CHLD=1 response: {response.strip()}")

        logger.error("All transfer methods failed")
        return False

    def get_call_info(self) -> Optional[CallInfo]:
        """Get current call information"""
        return self.current_call

    def _monitor_loop(self):
        """Background thread to monitor call state"""
        while self._running:
            try:
                # Check call status
                response = self._send_at("AT+CLCC")

                if self.current_call:
                    if "+CLCC:" in response:
                        # Parse call state from CLCC response
                        # Format: +CLCC: id,dir,stat,mode,mpty[,number,type]
                        # stat: 0=active, 1=held, 2=dialing, 3=alerting, 4=incoming, 5=waiting
                        try:
                            # Extract the part after +CLCC: and parse fields
                            clcc_data = response.split("+CLCC:")[1].split("\n")[0]
                            fields = clcc_data.split(",")
                            if len(fields) >= 3:
                                stat = int(fields[2].strip())
                                if stat == 0:  # Active call
                                    if self.current_call.state != CallState.CONNECTED:
                                        self.current_call.connect_time = time.time()
                                        logger.info("Call answered (CLCC stat=0)")
                                        self._notify_state(CallState.CONNECTED)
                                elif stat == 2:  # Dialing
                                    pass
                                elif stat == 3:  # Alerting (ringing on other end)
                                    if self.current_call.state != CallState.RINGING:
                                        self._notify_state(CallState.RINGING)
                                elif stat == 4:  # Incoming
                                    self._notify_state(CallState.INCOMING)
                        except (IndexError, ValueError) as e:
                            logger.debug(f"CLCC parse error: {e}")
                    elif self.current_call.state == CallState.CONNECTED:
                        # No call info but we were connected - call ended
                        self.current_call.end_time = time.time()
                        self._notify_state(CallState.ENDED)
                    # Note: Don't mark as ended if RINGING - CLCC may not return data
                    # until the call is actually answered

                # Check for incoming calls when idle
                if not self.current_call or self.current_call.state == CallState.IDLE:
                    if "RING" in response or "+CLIP:" in response:
                        # Extract caller ID if available
                        number = "Unknown"
                        if "+CLIP:" in response:
                            try:
                                number = response.split('"')[1]
                            except:
                                pass
                        self.current_call = CallInfo(
                            phone_number=number,
                            state=CallState.INCOMING,
                            start_time=time.time(),
                            direction="incoming"
                        )
                        self._notify_state(CallState.INCOMING)

            except Exception as e:
                logger.debug(f"Monitor error: {e}")

            time.sleep(0.5)

    def read_audio(self) -> Optional[bytes]:
        """Read audio data from modem (if USB audio is enabled)"""
        if not self.dev:
            return None
        try:
            # Try to read from audio endpoint
            data = self.dev.read(self.AUDIO_EP_IN, 320, timeout=100)
            return bytes(data)
        except:
            return None

    def write_audio(self, audio_data: bytes):
        """Write audio data to modem (if USB audio is enabled)"""
        if not self.dev:
            return
        try:
            self.dev.write(self.AUDIO_EP_OUT, audio_data, timeout=100)
        except Exception as e:
            logger.debug(f"Audio write error: {e}")


# Test function
def test_modem():
    logging.basicConfig(level=logging.INFO)

    modem = SIM7600Modem()

    if not modem.connect():
        print("Failed to connect to modem")
        return

    print("\nModem connected successfully!")
    print("-" * 40)

    # Get some info
    print("Network:", modem._send_at("AT+COPS?").strip())
    print("Signal:", modem._send_at("AT+CSQ").strip())

    # Don't actually make a call in test
    print("\nModem is ready to make calls!")
    print("Use: modem.dial('1234567890')")
    print("     modem.hangup()")

    modem.disconnect()


if __name__ == "__main__":
    test_modem()
