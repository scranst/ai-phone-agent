"""
SMS Command Handler

Allows the primary user to control the AI via text messages.

Commands:
- "call [contact] and [objective]" - Make a call to a contact with an objective
- "book [contact] for [date/time]" - Book an appointment with a contact
- "text [contact] [message]" - Send an SMS to a contact
- "status" - Get current system status
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
import json
import os

import database
from calendar_integration import get_calendar_integration, TimeSlot

logger = logging.getLogger(__name__)


class SMSCommandHandler:
    """Handles SMS commands from the primary user"""

    def __init__(self, primary_number: str):
        """
        Initialize with the primary user's phone number.
        Only commands from this number will be processed.
        """
        self.primary_number = self._normalize_phone(primary_number)
        self.pending_calls = []  # Queue of calls to make

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to digits only"""
        if not phone:
            return ""
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            digits = "1" + digits
        return digits

    def is_authorized(self, sender: str) -> bool:
        """Check if the sender is the primary user"""
        normalized = self._normalize_phone(sender)
        return normalized == self.primary_number

    def process_command(self, sender: str, message: str) -> Tuple[bool, str]:
        """
        Process an incoming SMS command.

        Args:
            sender: Phone number that sent the message
            message: The SMS content

        Returns:
            (success, response_message)
        """
        if not self.is_authorized(sender):
            logger.warning(f"Unauthorized SMS command from {sender}")
            return False, "Unauthorized"

        message = message.strip().lower()
        logger.info(f"Processing SMS command: {message}")

        # Parse command
        if message.startswith("call "):
            return self._handle_call_command(message[5:])
        elif message.startswith("book "):
            return self._handle_book_command(message[5:])
        elif message.startswith("text ") or message.startswith("sms "):
            return self._handle_text_command(message.split(" ", 1)[1] if " " in message else "")
        elif message == "status":
            return self._handle_status_command()
        elif message == "help":
            return self._handle_help_command()
        else:
            # Try to parse as natural language
            return self._handle_natural_command(message)

    def _handle_call_command(self, args: str) -> Tuple[bool, str]:
        """
        Handle: call [contact] and [objective]
        Example: "call john and remind him about the meeting tomorrow"
        """
        # Parse "contact and objective" or "contact to objective"
        match = re.match(r'(.+?)\s+(?:and|to)\s+(.+)', args, re.IGNORECASE)

        if not match:
            # Try just contact name
            contact_query = args.strip()
            objective = None
        else:
            contact_query = match.group(1).strip()
            objective = match.group(2).strip()

        # Find the contact
        lead = self._find_contact(contact_query)
        if not lead:
            return False, f"Contact '{contact_query}' not found"

        if not lead.get('phone'):
            return False, f"No phone number for {lead.get('first_name', contact_query)}"

        # Build call request
        contact_name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
        call_objective = objective or f"Follow up call with {contact_name}"

        # Queue the call
        self.pending_calls.append({
            'phone': lead['phone'],
            'objective': call_objective,
            'lead_id': lead['id'],
            'contact_name': contact_name
        })

        return True, f"Calling {contact_name} at {lead['phone']}. Objective: {call_objective}"

    def _handle_book_command(self, args: str) -> Tuple[bool, str]:
        """
        Handle: book [contact] for [date/time]
        Example: "book john for tuesday at 4pm"
        """
        # Parse "contact for date/time"
        match = re.match(r'(.+?)\s+(?:for|on|at)\s+(.+)', args, re.IGNORECASE)

        if not match:
            return False, "Format: book [contact] for [date/time]"

        contact_query = match.group(1).strip()
        datetime_str = match.group(2).strip()

        # Find the contact
        lead = self._find_contact(contact_query)
        if not lead:
            return False, f"Contact '{contact_query}' not found"

        # Parse the date/time
        parsed_dt = self._parse_datetime(datetime_str)
        if not parsed_dt:
            return False, f"Could not understand date/time: {datetime_str}"

        # Get calendar and book
        calendar = get_calendar_integration()
        if not calendar:
            return False, "Calendar not configured"

        contact_name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
        slot = TimeSlot(start=parsed_dt, end=parsed_dt + timedelta(minutes=30))

        result = calendar.book_appointment(
            slot=slot,
            name=contact_name,
            email=lead.get('email', ''),
            phone=lead.get('phone', ''),
            notes=f"Booked via SMS command"
        )

        if result.success:
            formatted_time = parsed_dt.strftime("%A %m/%d at %I:%M %p")
            return True, f"Booked {contact_name} for {formatted_time}"
        else:
            return False, f"Booking failed: {result.message}"

    def _handle_text_command(self, args: str) -> Tuple[bool, str]:
        """
        Handle: text [contact] [message]
        Example: "text john hey, just following up on our conversation"
        """
        # Parse "contact message"
        parts = args.split(" ", 1)
        if len(parts) < 2:
            return False, "Format: text [contact] [message]"

        contact_query = parts[0].strip()
        message = parts[1].strip()

        # Find the contact
        lead = self._find_contact(contact_query)
        if not lead:
            return False, f"Contact '{contact_query}' not found"

        if not lead.get('phone'):
            return False, f"No phone number for {lead.get('first_name', contact_query)}"

        contact_name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()

        # Return the SMS details - actual sending will be done by caller
        return True, f"SEND_SMS:{lead['phone']}:{message}"

    def _handle_status_command(self) -> Tuple[bool, str]:
        """Handle: status"""
        stats = database.get_lead_stats()
        pending = len(self.pending_calls)

        status = f"Leads: {stats['total']}"
        if pending:
            status += f", Pending calls: {pending}"

        return True, status

    def _handle_help_command(self) -> Tuple[bool, str]:
        """Handle: help"""
        help_text = (
            "Commands:\n"
            "call [name] and [task]\n"
            "book [name] for [time]\n"
            "text [name] [msg]\n"
            "status"
        )
        return True, help_text

    def _handle_natural_command(self, message: str) -> Tuple[bool, str]:
        """Try to parse natural language commands"""
        # Simple pattern matching for common phrases

        # "remind john about..." -> call john and remind about...
        match = re.match(r'remind\s+(\w+)\s+(?:about\s+)?(.+)', message, re.IGNORECASE)
        if match:
            return self._handle_call_command(f"{match.group(1)} and remind them about {match.group(2)}")

        # "schedule a call with john" -> call john
        match = re.match(r'(?:schedule|make)\s+(?:a\s+)?call\s+(?:with|to)\s+(\w+)', message, re.IGNORECASE)
        if match:
            return self._handle_call_command(match.group(1))

        # "set up a meeting with john for tuesday"
        match = re.match(r'(?:set up|schedule)\s+(?:a\s+)?meeting\s+with\s+(\w+)\s+(?:for|on)\s+(.+)', message, re.IGNORECASE)
        if match:
            return self._handle_book_command(f"{match.group(1)} for {match.group(2)}")

        return False, "Unknown command. Text 'help' for options."

    def _find_contact(self, query: str) -> Optional[dict]:
        """Find a contact by name (first name, last name, or full name)"""
        query = query.lower().strip()

        # Search leads
        leads, _ = database.search_leads(search=query, limit=10)

        if not leads:
            return None

        # Try exact first name match first
        for lead in leads:
            first = (lead.get('first_name') or '').lower()
            if first == query:
                return lead

        # Try full name match
        for lead in leads:
            full = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".lower().strip()
            if full == query:
                return lead

        # Return first result as fallback
        return leads[0]

    def _parse_datetime(self, text: str) -> Optional[datetime]:
        """Parse natural language date/time"""
        text = text.lower().strip()
        now = datetime.now()

        # Handle relative days
        day_offsets = {
            'today': 0,
            'tomorrow': 1,
            'monday': None,
            'tuesday': None,
            'wednesday': None,
            'thursday': None,
            'friday': None,
            'saturday': None,
            'sunday': None
        }

        # Find day
        target_date = None
        for day_name, offset in day_offsets.items():
            if day_name in text:
                if offset is not None:
                    target_date = now.date() + timedelta(days=offset)
                else:
                    # Find next occurrence of this weekday
                    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    target_weekday = days.index(day_name)
                    days_ahead = target_weekday - now.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = now.date() + timedelta(days=days_ahead)
                break

        if not target_date:
            target_date = now.date()

        # Find time
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = (time_match.group(3) or '').lower()

            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
        else:
            # Default to 10am
            hour = 10
            minute = 0

        return datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))

    def get_pending_call(self) -> Optional[dict]:
        """Get and remove the next pending call from the queue"""
        if self.pending_calls:
            return self.pending_calls.pop(0)
        return None

    def has_pending_calls(self) -> bool:
        """Check if there are pending calls"""
        return len(self.pending_calls) > 0
