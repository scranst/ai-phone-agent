"""
Calendar Integration Module

Provides unified interface for calendar APIs:
- Cal.com
- Calendly

Supports:
- Checking availability
- Booking appointments
"""

import logging
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """Represents an available time slot"""
    start: datetime
    end: datetime

    def __str__(self):
        return f"{self.start.strftime('%I:%M %p')} - {self.end.strftime('%I:%M %p')}"


@dataclass
class BookingResult:
    """Result of a booking attempt"""
    success: bool
    confirmation_id: Optional[str] = None
    message: str = ""
    booking_time: Optional[datetime] = None


class CalendarIntegration(ABC):
    """Base class for calendar integrations"""

    @abstractmethod
    def check_availability(self, target_date: date) -> list[TimeSlot]:
        """
        Check available time slots for a given date.

        Args:
            target_date: The date to check

        Returns:
            List of available TimeSlot objects
        """
        pass

    @abstractmethod
    def book_appointment(
        self,
        slot: TimeSlot,
        name: str,
        email: str,
        phone: str = "",
        notes: str = ""
    ) -> BookingResult:
        """
        Book an appointment in a given time slot.

        Args:
            slot: The time slot to book
            name: Customer name
            email: Customer email
            phone: Customer phone (optional)
            notes: Additional notes (optional)

        Returns:
            BookingResult with success status and details
        """
        pass


class CalComIntegration(CalendarIntegration):
    """Cal.com calendar integration"""

    BASE_URL = "https://api.cal.com/v1"

    def __init__(self, api_key: str, event_type_id: str):
        """
        Initialize Cal.com integration.

        Args:
            api_key: Cal.com API key
            event_type_id: The event type ID to use for bookings
        """
        self.api_key = api_key
        self.event_type_id = event_type_id
        # Cal.com uses query parameter for API key, not Bearer token
        self.headers = {
            "Content-Type": "application/json"
        }

    def check_availability(self, target_date: date) -> list[TimeSlot]:
        """Check available slots for a date on Cal.com"""
        try:
            # Cal.com slots endpoint
            url = f"{self.BASE_URL}/slots"

            # Get slots for the whole day
            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = start_time + timedelta(days=1)

            params = {
                "apiKey": self.api_key,
                "eventTypeId": self.event_type_id,
                "startTime": start_time.isoformat() + "Z",
                "endTime": end_time.isoformat() + "Z",
                "timeZone": "America/Los_Angeles"  # TODO: Make configurable
            }

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Cal.com API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            slots = []

            # Parse available slots - Cal.com returns {"slots": {"2026-01-03": [{"time": "..."}]}}
            slots_data = data.get("slots", {})
            if isinstance(slots_data, dict):
                # Slots grouped by date
                for date_key, day_slots in slots_data.items():
                    for slot_data in day_slots:
                        try:
                            start = datetime.fromisoformat(slot_data["time"].replace("Z", "+00:00"))
                            end = start + timedelta(minutes=30)
                            slots.append(TimeSlot(start=start, end=end))
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse slot: {e}")
            else:
                # Slots as flat list
                for slot_data in slots_data:
                    try:
                        start = datetime.fromisoformat(slot_data["time"].replace("Z", "+00:00"))
                        end = start + timedelta(minutes=30)
                        slots.append(TimeSlot(start=start, end=end))
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse slot: {e}")

            logger.info(f"Found {len(slots)} available slots on {target_date}")
            return slots

        except Exception as e:
            logger.error(f"Failed to check Cal.com availability: {e}")
            return []

    def book_appointment(
        self,
        slot: TimeSlot,
        name: str,
        email: str,
        phone: str = "",
        notes: str = ""
    ) -> BookingResult:
        """Book an appointment on Cal.com"""
        try:
            url = f"{self.BASE_URL}/bookings"

            # Format time with timezone offset for Cal.com
            # Cal.com requires the exact format from their slots API
            start_time = slot.start.strftime("%Y-%m-%dT%H:%M:%S") + "-08:00"  # PST

            payload = {
                "eventTypeId": int(self.event_type_id),
                "start": start_time,
                "responses": {
                    "name": name,
                    "email": email
                },
                "timeZone": "America/Los_Angeles",
                "language": "en",
                "metadata": {}
            }

            if notes:
                payload["metadata"]["notes"] = notes

            logger.info(f"Attempting to book: {name} at {start_time}")

            response = requests.post(
                url,
                headers=self.headers,
                params={"apiKey": self.api_key},
                json=payload,
                timeout=30
            )

            if response.status_code in [200, 201]:
                data = response.json()
                return BookingResult(
                    success=True,
                    confirmation_id=str(data.get("id", "")),
                    message="Appointment booked successfully",
                    booking_time=slot.start
                )
            else:
                logger.error(f"Cal.com booking failed: {response.status_code} - {response.text}")
                return BookingResult(
                    success=False,
                    message=f"Booking failed: {response.text}"
                )

        except Exception as e:
            logger.error(f"Failed to book Cal.com appointment: {e}")
            return BookingResult(
                success=False,
                message=str(e)
            )


class CalendlyIntegration(CalendarIntegration):
    """Calendly calendar integration"""

    BASE_URL = "https://api.calendly.com"

    def __init__(self, api_key: str, user_uri: str):
        """
        Initialize Calendly integration.

        Args:
            api_key: Calendly API key (Personal Access Token)
            user_uri: Calendly user URI (e.g., https://api.calendly.com/users/XXXXX)
        """
        self.api_key = api_key
        self.user_uri = user_uri
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._event_type_uri: Optional[str] = None

    def _get_event_type(self) -> Optional[str]:
        """Get the first active event type for the user"""
        if self._event_type_uri:
            return self._event_type_uri

        try:
            url = f"{self.BASE_URL}/event_types"
            params = {"user": self.user_uri, "active": True}

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                event_types = data.get("collection", [])
                if event_types:
                    self._event_type_uri = event_types[0]["uri"]
                    return self._event_type_uri

            logger.error(f"Failed to get Calendly event types: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Failed to get Calendly event type: {e}")
            return None

    def check_availability(self, target_date: date) -> list[TimeSlot]:
        """Check available slots for a date on Calendly"""
        try:
            event_type_uri = self._get_event_type()
            if not event_type_uri:
                logger.error("No event type found for Calendly user")
                return []

            url = f"{self.BASE_URL}/event_type_available_times"

            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = start_time + timedelta(days=1)

            params = {
                "event_type": event_type_uri,
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z"
            }

            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Calendly API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            slots = []

            for slot_data in data.get("collection", []):
                try:
                    start = datetime.fromisoformat(
                        slot_data["start_time"].replace("Z", "+00:00")
                    )
                    # Get scheduling link for this slot
                    slots.append(TimeSlot(start=start, end=start + timedelta(minutes=30)))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse Calendly slot: {e}")

            logger.info(f"Found {len(slots)} available Calendly slots on {target_date}")
            return slots

        except Exception as e:
            logger.error(f"Failed to check Calendly availability: {e}")
            return []

    def book_appointment(
        self,
        slot: TimeSlot,
        name: str,
        email: str,
        phone: str = "",
        notes: str = ""
    ) -> BookingResult:
        """
        Book an appointment on Calendly.

        Note: Calendly doesn't support direct API booking for most plans.
        This returns a scheduling link instead.
        """
        try:
            event_type_uri = self._get_event_type()
            if not event_type_uri:
                return BookingResult(
                    success=False,
                    message="No event type configured"
                )

            # For Calendly, we need to use their scheduling page
            # Direct booking is only available on enterprise plans
            # Return a helpful message with the scheduling link

            # Get the scheduling URL from the event type
            response = requests.get(
                event_type_uri,
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                scheduling_url = data.get("resource", {}).get("scheduling_url", "")

                return BookingResult(
                    success=True,
                    message=f"Please complete booking at: {scheduling_url}",
                    booking_time=slot.start
                )

            return BookingResult(
                success=False,
                message="Failed to get scheduling link"
            )

        except Exception as e:
            logger.error(f"Failed to book Calendly appointment: {e}")
            return BookingResult(
                success=False,
                message=str(e)
            )


def get_calendar_integration() -> Optional[CalendarIntegration]:
    """
    Get the configured calendar integration based on settings.

    Returns:
        CalendarIntegration instance or None if not configured
    """
    settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
    if not os.path.exists(settings_path):
        return None

    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)

        integrations = settings.get("integrations", {})
        provider = integrations.get("CALENDAR_PROVIDER", "")

        if provider == "cal.com":
            api_key = integrations.get("CAL_COM_API_KEY", "")
            event_type_id = integrations.get("CAL_COM_EVENT_TYPE_ID", "")
            if api_key and event_type_id:
                return CalComIntegration(api_key, event_type_id)
            else:
                logger.warning("Cal.com credentials not configured")

        elif provider == "calendly":
            api_key = integrations.get("CALENDLY_API_KEY", "")
            user_uri = integrations.get("CALENDLY_USER_URI", "")
            if api_key and user_uri:
                return CalendlyIntegration(api_key, user_uri)
            else:
                logger.warning("Calendly credentials not configured")

        return None

    except Exception as e:
        logger.error(f"Failed to load calendar integration: {e}")
        return None


# Functions for LLM tool use
def check_calendar_availability(date_str: str) -> str:
    """
    Check calendar availability for a given date.

    Args:
        date_str: Date string in format YYYY-MM-DD or natural language like "tomorrow"

    Returns:
        String describing available time slots
    """
    calendar = get_calendar_integration()
    if not calendar:
        return "Calendar not configured. Please set up calendar integration in settings."

    try:
        # Parse date
        target_date = _parse_date(date_str)
        if not target_date:
            return f"Could not parse date: {date_str}"

        slots = calendar.check_availability(target_date)

        if not slots:
            return f"No available time slots on {target_date.strftime('%B %d, %Y')}"

        # Format response
        lines = [f"Available times on {target_date.strftime('%B %d, %Y')}:"]
        for slot in slots:
            lines.append(f"  - {slot}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return f"Error checking availability: {str(e)}"


def book_calendar_appointment(
    date_str: str,
    time_str: str,
    name: str,
    email: str,
    phone: str = "",
    notes: str = ""
) -> str:
    """
    Book a calendar appointment.

    Args:
        date_str: Date in format YYYY-MM-DD or natural language
        time_str: Time in format HH:MM or natural language like "2pm"
        name: Customer name
        email: Customer email
        phone: Customer phone (optional)
        notes: Additional notes (optional)

    Returns:
        String describing booking result
    """
    calendar = get_calendar_integration()
    if not calendar:
        return "Calendar not configured. Please set up calendar integration in settings."

    try:
        # Parse date and time
        target_date = _parse_date(date_str)
        if not target_date:
            return f"Could not parse date: {date_str}"

        target_time = _parse_time(time_str)
        if not target_time:
            return f"Could not parse time: {time_str}"

        # Combine date and time
        target_datetime = datetime.combine(target_date, target_time)

        # Check if this time is available
        slots = calendar.check_availability(target_date)

        # Find matching slot
        matching_slot = None
        for slot in slots:
            # Check if requested time matches a slot start
            if abs((slot.start.replace(tzinfo=None) - target_datetime).total_seconds()) < 300:  # 5 min tolerance
                matching_slot = slot
                break

        if not matching_slot:
            # Try to find the closest available slot
            available_times = [slot.start.strftime('%I:%M %p') for slot in slots[:5]]
            if available_times:
                return f"The requested time ({time_str}) is not available. Available times: {', '.join(available_times)}"
            return f"No available time slots on {target_date.strftime('%B %d, %Y')}"

        # Book the appointment
        result = calendar.book_appointment(
            slot=matching_slot,
            name=name,
            email=email,
            phone=phone,
            notes=notes
        )

        if result.success:
            return f"Appointment booked successfully for {matching_slot.start.strftime('%B %d, %Y at %I:%M %p')}. {result.message}"
        else:
            return f"Failed to book appointment: {result.message}"

    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return f"Error booking appointment: {str(e)}"


def _parse_date(date_str: str) -> Optional[date]:
    """Parse a date string into a date object"""
    date_str = date_str.lower().strip()

    today = date.today()

    # Natural language dates
    if date_str == "today":
        return today
    elif date_str == "tomorrow":
        return today + timedelta(days=1)
    elif date_str == "day after tomorrow":
        return today + timedelta(days=2)

    # Try common formats
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d", "%B %d", "%b %d"]:
        try:
            parsed = datetime.strptime(date_str, fmt).date()
            # If no year in format, use current year
            if parsed.year == 1900:
                parsed = parsed.replace(year=today.year)
            return parsed
        except ValueError:
            continue

    return None


def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse a time string into a time object"""
    time_str = time_str.lower().strip()

    # Try common formats
    for fmt in ["%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M"]:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue

    return None


# Test function
def test_calendar():
    """Test calendar integration"""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Calendar Integration Test")
    print("=" * 60)

    calendar = get_calendar_integration()
    if not calendar:
        print("No calendar integration configured.")
        print("Please configure Cal.com or Calendly in settings.")
        return

    print(f"Using: {calendar.__class__.__name__}")

    # Test availability check
    tomorrow = date.today() + timedelta(days=1)
    print(f"\nChecking availability for {tomorrow}...")

    slots = calendar.check_availability(tomorrow)
    if slots:
        print(f"Found {len(slots)} available slots:")
        for slot in slots[:5]:
            print(f"  - {slot}")
    else:
        print("No available slots found")


if __name__ == "__main__":
    test_calendar()
