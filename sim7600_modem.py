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
import usb.backend.libusb1 as libusb1
import time
import logging
import threading
import platform
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# On macOS, we need to explicitly set the libusb path
_usb_backend = None
if platform.system() == 'Darwin':
    # Try common homebrew locations
    for libpath in ['/opt/homebrew/lib/libusb-1.0.dylib', '/usr/local/lib/libusb-1.0.dylib']:
        try:
            _usb_backend = libusb1.get_backend(find_library=lambda x, p=libpath: p)
            if _usb_backend:
                logger.debug(f"Using libusb from: {libpath}")
                break
        except:
            pass


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
        self._sms_callbacks: list[Callable[[str, str], None]] = []  # (sender, message)
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._reconnect_lock = threading.Lock()  # Separate lock for reconnection
        self._sms_in_progress = False  # Flag to pause monitor during SMS operations
        self._product_id: Optional[int] = None
        self._at_interface: int = self.AT_INTERFACE
        self._at_ep_in: int = self.AT_EP_IN
        self._at_ep_out: int = self.AT_EP_OUT
        self._last_successful_at: float = 0  # Track last successful AT command
        self._reconnecting: bool = False  # Flag to prevent concurrent reconnects

    def connect(self, retries: int = 3) -> bool:
        """Connect to the SIM7600 modem with retry logic"""
        for attempt in range(retries):
            # Try each known Product ID
            self.dev = None
            for pid in self.PRODUCT_IDS:
                self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=pid, backend=_usb_backend)
                if self.dev is not None:
                    self._product_id = pid
                    break

            if self.dev is None:
                if attempt < retries - 1:
                    logger.debug(f"Modem not found, retrying in 1s (attempt {attempt + 1}/{retries})")
                    time.sleep(1)
                    continue
                logger.error("SIM7600 modem not found")
                logger.info("Make sure the modem is connected and powered")
                return False

            logger.info(f"Found SIM7600 (PID 0x{self._product_id:04x}): {self.dev.manufacturer} - {self.dev.product}")

            # Try to connect - if it fails, retry after a delay
            if self._do_connect():
                return True

            if attempt < retries - 1:
                logger.warning(f"Connection failed, retrying in 2s (attempt {attempt + 1}/{retries})")
                # Dispose resources and reset device reference
                try:
                    usb.util.dispose_resources(self.dev)
                except:
                    pass
                self.dev = None
                time.sleep(2)

        return False

    def _do_connect(self) -> bool:
        """Internal method to actually connect to the modem"""
        if self.dev is None:
            return False

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
            # Try USB reset to clear any stale state
            try:
                self.dev.reset()
                time.sleep(0.5)  # Give device time to re-enumerate
            except Exception as e:
                logger.debug(f"USB reset skipped: {e}")

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

            # Check if SIM is ready (retry a few times as modem may need time)
            sim_ready = False
            for attempt in range(3):
                time.sleep(0.5)  # Give modem time between attempts
                response = self._send_at("AT+CPIN?", timeout=3000)
                if "READY" in response:
                    sim_ready = True
                    break
                elif "ERROR" in response:
                    logger.error(f"SIM error: {response}")
                    break
                # If we just got "OK" without READY, the SIM might still be initializing
                logger.info(f"SIM check attempt {attempt + 1}: {response.strip()}")

            if not sim_ready:
                # Last resort: try to make a simple AT command to verify modem works
                test_response = self._send_at("AT")
                if "OK" in test_response:
                    logger.warning("SIM status unclear but modem responds - continuing anyway")
                    sim_ready = True
                else:
                    logger.error(f"SIM not ready after retries")
                    return False

            # Get signal strength
            response = self._send_at("AT+CSQ")
            logger.info(f"Signal: {response.strip()}")

            # Enable new SMS indications (sends +CMTI when SMS arrives)
            # Mode 2,1 = buffer URCs during AT commands, show +CMTI for new SMS
            self._send_at("AT+CNMI=2,1,0,0,0")
            # Set text mode for SMS
            self._send_at("AT+CMGF=1")
            logger.info("SMS notifications enabled (+CMTI)")

            # Start monitoring thread
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to modem: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if modem is connected and responsive"""
        if not self.dev:
            return False
        # Check if we've had a successful AT command recently (within 30 seconds)
        if self._last_successful_at > 0 and (time.time() - self._last_successful_at) < 30:
            return True
        # Otherwise do a quick check
        try:
            with self._lock:
                self.dev.write(self._at_ep_out, b"AT\r\n", 1000)
                time.sleep(0.1)
                data = self.dev.read(self._at_ep_in, 512, timeout=500)
                if b"OK" in bytes(data):
                    self._last_successful_at = time.time()
                    return True
        except:
            pass
        return False

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the modem after a disconnection.
        Thread-safe - only one reconnection attempt at a time.
        """
        # Use a separate lock to prevent concurrent reconnection attempts
        if not self._reconnect_lock.acquire(blocking=False):
            logger.debug("Reconnection already in progress")
            return False

        try:
            if self._reconnecting:
                return False

            self._reconnecting = True
            logger.warning("Modem disconnected - attempting to reconnect...")

            # Stop the monitor thread temporarily
            was_running = self._running
            self._running = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2)

            # Clean up old connection
            if self.dev:
                try:
                    usb.util.release_interface(self.dev, self._at_interface)
                except:
                    pass
                try:
                    usb.util.dispose_resources(self.dev)
                except:
                    pass
                self.dev = None

            # Wait for USB to settle
            time.sleep(2)

            # Try to reconnect
            for attempt in range(5):
                logger.info(f"Reconnection attempt {attempt + 1}/5...")
                if self.connect(retries=1):
                    logger.info("Modem reconnected successfully!")
                    self._reconnecting = False
                    return True
                time.sleep(2)

            logger.error("Failed to reconnect to modem after 5 attempts")
            self._reconnecting = False
            return False

        finally:
            self._reconnect_lock.release()

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

    def _send_at(self, cmd: str, timeout: int = 2000, auto_reconnect: bool = True) -> str:
        """Send AT command and return response. Auto-reconnects on USB errors."""
        if not self.dev:
            if auto_reconnect and not self._reconnecting:
                if self.reconnect():
                    return self._send_at(cmd, timeout, auto_reconnect=False)
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

                # Track successful communication
                self._last_successful_at = time.time()
                return response.decode('utf-8', errors='replace')

            except usb.core.USBError as e:
                # USB error - device likely disconnected
                error_msg = str(e)
                logger.error(f"AT command USB error: {e}")

                # Check for disconnection errors
                if "No such device" in error_msg or "errno 19" in error_msg.lower():
                    self.dev = None  # Mark as disconnected
                    if auto_reconnect and not self._reconnecting:
                        # Try to reconnect in background (don't block)
                        threading.Thread(target=self.reconnect, daemon=True).start()

                return f"ERROR: {e}"

            except Exception as e:
                logger.error(f"AT command error: {e}")
                return f"ERROR: {e}"

    def on_state_change(self, callback: Callable[[CallState], None]):
        """Register callback for call state changes"""
        self._state_callbacks.append(callback)

    def on_audio(self, callback: Callable[[bytes], None]):
        """Register callback for incoming audio data"""
        self._audio_callback = callback

    def on_sms(self, callback: Callable[[str, str], None]):
        """Register callback for incoming SMS (sender, message)"""
        self._sms_callbacks.append(callback)

    def _decode_ucs2_hex(self, hex_string: str) -> str:
        """
        Decode UCS2 hex-encoded SMS message.
        When SMS contains special characters (smart quotes, emojis, etc.),
        the modem returns the message as UCS2 hex-encoded.
        Example: "00430061006C006C" -> "Call"
        """
        try:
            # Check if this looks like UCS2 hex (all hex chars, length multiple of 4)
            if not hex_string or len(hex_string) % 4 != 0:
                return hex_string
            if not all(c in '0123456789ABCDEFabcdef' for c in hex_string):
                return hex_string

            # Decode UCS2 (UTF-16 big-endian)
            result = []
            for i in range(0, len(hex_string), 4):
                code_point = int(hex_string[i:i+4], 16)
                result.append(chr(code_point))
            return ''.join(result)
        except Exception as e:
            logger.debug(f"UCS2 decode failed: {e}")
            return hex_string

    def _notify_sms(self, sender: str, message: str):
        """Notify all SMS callbacks"""
        # Try to decode UCS2 if message looks hex-encoded
        decoded_message = self._decode_ucs2_hex(message)
        for cb in self._sms_callbacks:
            try:
                cb(sender, decoded_message)
            except Exception as e:
                logger.error(f"SMS callback error: {e}")

    def _read_sms_by_index(self, index: str):
        """Read a specific SMS by index, notify callbacks, then delete it"""
        try:
            # Read the specific message
            response = self._send_at(f"AT+CMGR={index}", timeout=3000)

            if "+CMGR:" in response:
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('+CMGR:'):
                        try:
                            # Parse sender from +CMGR line
                            parts = line.split(',')
                            sender = parts[1].strip('" ')

                            # Message is on next line
                            if i + 1 < len(lines):
                                message = lines[i + 1].strip()
                                if message and message != 'OK':
                                    logger.info(f"SMS from {sender}: {message[:50]}...")
                                    self._notify_sms(sender, message)

                            # Delete the message
                            self._send_at(f"AT+CMGD={index}", timeout=2000)
                            break
                        except Exception as e:
                            logger.error(f"Failed to parse SMS: {e}")
        except Exception as e:
            logger.error(f"Failed to read SMS at index {index}: {e}")

    def _notify_state(self, state: CallState):
        """Notify all callbacks of state change"""
        if self.current_call:
            self.current_call.state = state
        for cb in self._state_callbacks:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def get_signal_strength(self) -> int:
        """Get signal strength (0-31, or 99 for unknown)"""
        response = self._send_at("AT+CSQ")
        # Response format: AT+CSQ\r\r\n+CSQ: 27,99\r\n\r\nOK\r\n
        try:
            if "+CSQ:" in response:
                # Extract the part after +CSQ:
                csq_part = response.split("+CSQ:")[1].strip()
                # Get first number before comma
                signal = int(csq_part.split(",")[0].strip())
                return signal
        except Exception as e:
            logger.debug(f"Failed to parse signal: {e}")
        return 0

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
        # Enable audio channel before answering
        self._send_at("AT+CSDVC=1")
        self._send_at("AT+CLVL=1")
        self._send_at("AT+CECM=1")

        response = self._send_at("ATA")
        if "OK" in response:
            if self.current_call:
                self.current_call.connect_time = time.time()
            self._notify_state(CallState.CONNECTED)
            return True
        return False

    def wait_for_incoming_call(self, timeout: float = None) -> Optional[str]:
        """
        Block until an incoming call is detected.

        Args:
            timeout: Maximum seconds to wait, or None for indefinite

        Returns:
            Caller ID phone number, or "Unknown" if not available, or None if timeout
        """
        # Enable caller ID (CLIP)
        self._send_at("AT+CLIP=1")

        start = time.time()
        while True:
            if timeout and (time.time() - start) > timeout:
                return None

            # Check for incoming call
            if self.current_call and self.current_call.state == CallState.INCOMING:
                return self.current_call.phone_number

            # Manual check for RING
            with self._lock:
                try:
                    # Try to read unsolicited responses
                    data = self.dev.read(self._at_ep_in, 512, timeout=500)
                    response = bytes(data).decode('utf-8', errors='replace')

                    if "RING" in response:
                        # Extract caller ID if available
                        number = "Unknown"
                        if "+CLIP:" in response:
                            try:
                                number = response.split('"')[1]
                            except:
                                pass
                        else:
                            # +CLIP: may come in a separate packet - wait for it
                            for _ in range(5):  # Try up to 5 times (2.5 seconds)
                                try:
                                    data2 = self.dev.read(self._at_ep_in, 512, timeout=500)
                                    response2 = bytes(data2).decode('utf-8', errors='replace')
                                    if "+CLIP:" in response2:
                                        try:
                                            number = response2.split('"')[1]
                                        except:
                                            pass
                                        break
                                except:
                                    break

                        self.current_call = CallInfo(
                            phone_number=number,
                            state=CallState.INCOMING,
                            start_time=time.time(),
                            direction="incoming"
                        )
                        self._notify_state(CallState.INCOMING)
                        logger.info(f"Incoming call from: {number}")
                        return number

                except usb.core.USBTimeoutError:
                    pass
                except Exception as e:
                    logger.debug(f"Wait for call error: {e}")

            time.sleep(0.2)

    def is_ringing(self) -> bool:
        """Check if there's currently an incoming call"""
        return (self.current_call is not None and
                self.current_call.state == CallState.INCOMING)

    def reject_call(self) -> bool:
        """Reject an incoming call"""
        response = self._send_at("AT+CHUP")
        if self.current_call:
            self.current_call.end_time = time.time()
            self.current_call.state = CallState.ENDED
        return "OK" in response

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
        time.sleep(2.0)  # Let call stabilize longer

        # Check current call state before merging
        clcc_before = self._send_at("AT+CLCC")
        logger.info(f"Calls before merge: {clcc_before.strip()}")

        # Step 3: Merge calls with AT+CHLD=3 (standard 3-way conference)
        logger.info("Sending merge command (AT+CHLD=3)...")
        response = self._send_at("AT+CHLD=3", timeout=5000)
        logger.info(f"Merge response: {response.strip()}")

        # Check for explicit failure
        if "+CME ERROR" in response:
            logger.error(f"Merge failed - network error: {response.strip()}")
            return False

        if "ERROR" in response:
            logger.error("Merge command returned ERROR")
            return False

        if "VOICE CALL: END" in response:
            logger.error("Merge failed - call ended")
            return False

        # If we got here without ERROR, assume the merge worked!
        # The 3-way call is now active. Don't check CLCC immediately
        # as it may return empty during the merge transition.
        logger.info("3-way conference initiated - calls should be merged")
        logger.info("Transfer complete! AI agent stepping back, human taking over.")
        return True

    def get_call_info(self) -> Optional[CallInfo]:
        """Get current call information"""
        return self.current_call

    def _monitor_loop(self):
        """Background thread to monitor call state"""
        while self._running:
            # Pause monitoring if SMS operation is in progress
            if self._sms_in_progress:
                time.sleep(0.1)
                continue

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

                # Check for new SMS notification (+CMTI)
                if "+CMTI:" in response:
                    try:
                        # Format: +CMTI: "SM",<index>
                        cmti_part = response.split("+CMTI:")[1].split("\n")[0]
                        index = cmti_part.split(",")[1].strip()
                        logger.info(f"New SMS notification at index {index}")
                        # Read the SMS
                        self._read_sms_by_index(index)
                    except Exception as e:
                        logger.error(f"Failed to parse CMTI: {e}")

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

    def send_sms(self, phone_number: str, message: str) -> bool:
        """
        Send an SMS message.

        Args:
            phone_number: Destination phone number
            message: Text message to send (max ~160 chars for single SMS)

        Returns:
            True if SMS was sent successfully
        """
        if not self.dev:
            logger.error("Cannot send SMS: Not connected to modem")
            return False

        # Clean phone number
        clean_number = "".join(c for c in phone_number if c.isdigit() or c == "+")

        logger.info(f"Sending SMS to {clean_number}: {message[:50]}...")

        # Signal monitor loop to pause (prevents lock contention)
        self._sms_in_progress = True
        time.sleep(0.6)  # Wait for monitor loop to finish current iteration

        try:
            with self._lock:
                # Clear any pending data from modem
                for _ in range(5):
                    try:
                        self.dev.read(self._at_ep_in, 512, timeout=100)
                    except:
                        break

                # Send a simple AT to make sure modem is responsive
                cmd = "AT\r\n"
                self.dev.write(self._at_ep_out, cmd.encode(), 2000)
                time.sleep(0.3)
                try:
                    response = self.dev.read(self._at_ep_in, 512, timeout=500)
                    if b"OK" not in bytes(response):
                        logger.warning("Modem not responding to AT, retrying...")
                        time.sleep(1)
                except:
                    pass

                # Set SMS text mode
                cmd = "AT+CMGF=1\r\n"
                self.dev.write(self._at_ep_out, cmd.encode(), 2000)
                time.sleep(0.3)

                # Clear response
                try:
                    self.dev.read(self._at_ep_in, 512, timeout=300)
                except:
                    pass

                # Start SMS send command
                cmd = f'AT+CMGS="{clean_number}"\r\n'
                self.dev.write(self._at_ep_out, cmd.encode(), 2000)
                time.sleep(0.5)

                # Wait for ">" prompt (may need multiple reads)
                prompt_found = False
                for _ in range(5):
                    try:
                        response = self.dev.read(self._at_ep_in, 512, timeout=1000)
                        response_str = bytes(response).decode('utf-8', errors='replace')
                        if ">" in response_str:
                            prompt_found = True
                            break
                    except usb.core.USBTimeoutError:
                        continue

                if not prompt_found:
                    logger.error("SMS failed: No prompt received")
                    # Send escape to cancel
                    self.dev.write(self._at_ep_out, b"\x1b", 1000)
                    return False

                # Send message content followed by Ctrl+Z (0x1A)
                msg_bytes = (message + chr(26)).encode()
                self.dev.write(self._at_ep_out, msg_bytes, 5000)

                # Wait for response (can take a few seconds)
                time.sleep(3)
                response = bytes()
                for _ in range(10):
                    try:
                        data = self.dev.read(self._at_ep_in, 512, timeout=1000)
                        response += bytes(data)
                        if b"OK" in response or b"ERROR" in response:
                            break
                    except usb.core.USBTimeoutError:
                        continue

                response_str = response.decode('utf-8', errors='replace')

                if "OK" in response_str:
                    logger.info(f"SMS sent successfully to {clean_number}")
                    return True
                else:
                    logger.error(f"SMS failed: {response_str}")
                    return False

        except Exception as e:
            logger.error(f"SMS error: {e}")
            return False
        finally:
            # Always resume monitor loop
            self._sms_in_progress = False

    def read_sms(self, delete_after_read: bool = True) -> list:
        """
        Read all unread SMS messages.

        Args:
            delete_after_read: Whether to delete messages after reading

        Returns:
            List of dicts with 'sender' and 'message' keys
        """
        if not self.dev:
            logger.error("Cannot read SMS: Not connected to modem")
            return []

        messages = []

        try:
            with self._lock:
                # Set SMS text mode
                self._send_at("AT+CMGF=1", timeout=2000)
                time.sleep(0.2)

                # List all messages (both read and unread)
                response = self._send_at("AT+CMGL=\"ALL\"", timeout=5000)

                # Parse messages
                # Format: +CMGL: <index>,"<status>","<sender>",,"<timestamp>"\r\n<message>\r\n
                lines = response.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('+CMGL:'):
                        try:
                            # Parse header
                            parts = line.split(',')
                            index = parts[0].split(':')[1].strip()
                            sender = parts[2].strip('"')

                            # Next line is the message
                            if i + 1 < len(lines):
                                message_text = lines[i + 1].strip()
                                if message_text and not message_text.startswith('+CMGL:') and message_text != 'OK':
                                    messages.append({
                                        'index': index,
                                        'sender': sender,
                                        'message': message_text
                                    })
                                    logger.info(f"Read SMS from {sender}: {message_text[:50]}...")
                        except Exception as e:
                            logger.debug(f"Error parsing SMS: {e}")
                    i += 1

                # Delete read messages if requested
                if delete_after_read and messages:
                    for msg in messages:
                        try:
                            self._send_at(f"AT+CMGD={msg['index']}", timeout=2000)
                        except:
                            pass

        except Exception as e:
            logger.error(f"Error reading SMS: {e}")

        return messages

    def check_new_sms(self) -> list:
        """
        Check for new (unread) SMS messages only.

        Returns:
            List of dicts with 'sender' and 'message' keys
        """
        if not self.dev:
            return []

        messages = []

        try:
            with self._lock:
                # Set SMS text mode
                self._send_at("AT+CMGF=1", timeout=2000)
                time.sleep(0.2)

                # List only unread messages
                response = self._send_at("AT+CMGL=\"REC UNREAD\"", timeout=5000)

                # Parse messages (same format as read_sms)
                lines = response.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('+CMGL:'):
                        try:
                            parts = line.split(',')
                            index = parts[0].split(':')[1].strip()
                            sender = parts[2].strip('"')

                            if i + 1 < len(lines):
                                message_text = lines[i + 1].strip()
                                if message_text and not message_text.startswith('+CMGL:') and message_text != 'OK':
                                    messages.append({
                                        'index': index,
                                        'sender': sender,
                                        'message': message_text
                                    })
                        except:
                            pass
                    i += 1

                # Delete after reading
                for msg in messages:
                    try:
                        self._send_at(f"AT+CMGD={msg['index']}", timeout=2000)
                    except:
                        pass

        except Exception as e:
            logger.debug(f"Error checking SMS: {e}")

        return messages


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
