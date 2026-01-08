"""
AI Tools - Actions the AI can perform when acting as personal assistant

These tools are available when the main user interacts with the AI.
"""

import logging
import json
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (for Claude API)
# =============================================================================

ASSISTANT_TOOLS = [
    {
        "name": "search_contacts",
        "description": "Search your contacts/leads database for a person or company. Use this to find phone numbers, emails, and info about people you know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name of person or company to search for"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for any information - businesses, phone numbers, events, news, facts, etc. Returns search results with titles and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'pizza delivery Las Vegas', 'weather today')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_movie_showtimes",
        "description": "Get movie showtimes for a specific location. Use this when asked about movies playing, showtimes, or what's in theaters. Returns actual showtimes from theaters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or zip code (e.g., 'Las Vegas' or '89122')"
                },
                "movie": {
                    "type": "string",
                    "description": "Optional: specific movie name to search for (e.g., 'Avatar'). Leave empty to get all movies playing."
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "make_call",
        "description": "Make a phone call to a number with a specific objective. Choose the right agent based on the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Phone number to call (e.g., '555-123-4567')"
                },
                "objective": {
                    "type": "string",
                    "description": "What you want to accomplish on the call (e.g., 'Order a large pepperoni pizza for delivery', 'Make a reservation for 2 at 7pm', 'Schedule a meeting')"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Which agent makes the call: 'personal_assistant' for personal errands needing your info (reservations with credit card, doctor appointments, utilities, orders); 'receptionist' for scheduling meetings with leads/contacts on your calendar; 'sales_rep' for sales demos/pitches to prospects.",
                    "enum": ["personal_assistant", "receptionist", "sales_rep"],
                    "default": "personal_assistant"
                }
            },
            "required": ["phone_number", "objective"]
        }
    },
    {
        "name": "send_sms",
        "description": "Send a text message to a phone number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Phone number to text"
                },
                "message": {
                    "type": "string",
                    "description": "The message to send"
                }
            },
            "required": ["phone_number", "message"]
        }
    },
    {
        "name": "check_calendar_availability",
        "description": "Check calendar availability for a specific date. ALWAYS call this before suggesting meeting times to ensure you only offer times that are actually available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to check in YYYY-MM-DD format (e.g., '2026-01-06')"
                }
            },
            "required": ["date"]
        }
    },
    {
        "name": "book_calendar_appointment",
        "description": "Book a calendar appointment. Only call this AFTER the customer confirms the time. Returns success/failure so you know if it actually worked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format"
                },
                "time": {
                    "type": "string",
                    "description": "Time in HH:MM 24-hour format (e.g., '14:00' for 2 PM)"
                },
                "name": {
                    "type": "string",
                    "description": "Customer's full name"
                },
                "phone": {
                    "type": "string",
                    "description": "Customer's phone number"
                },
                "email": {
                    "type": "string",
                    "description": "Customer's email (optional)"
                }
            },
            "required": ["date", "time", "name", "phone"]
        }
    }
]


# =============================================================================
# Tool Implementations
# =============================================================================

def search_contacts(query: str) -> dict:
    """Search leads database for a contact"""
    try:
        import database

        leads, total = database.search_leads(search=query, limit=5)

        if not leads:
            return {
                "found": False,
                "message": f"No contacts found matching '{query}'"
            }

        results = []
        for lead in leads:
            name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
            if not name:
                name = lead.get('company', 'Unknown')

            results.append({
                "name": name,
                "company": lead.get('company'),
                "phone": lead.get('phone'),
                "email": lead.get('email'),
                "title": lead.get('title'),
                "notes": lead.get('notes')
            })

        return {
            "found": True,
            "count": len(results),
            "contacts": results
        }

    except Exception as e:
        logger.error(f"Error searching contacts: {e}")
        return {
            "found": False,
            "error": str(e)
        }


def get_movie_showtimes(location: str, movie: str = None) -> dict:
    """Get movie showtimes using Google search results"""
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import quote
    import json
    from pathlib import Path

    try:
        # Try Google Custom Search first (returns structured data)
        try:
            settings_file = Path(__file__).parent / "settings.json"
            if settings_file.exists():
                settings = json.loads(settings_file.read_text())
                api_keys = settings.get("api_keys", {})
                google_api_key = api_keys.get("GOOGLE_API_KEY")
                google_cse_id = api_keys.get("GOOGLE_CSE_ID")

                if google_api_key and google_cse_id:
                    # Search for specific movie showtimes
                    query = f"{movie or 'movies'} showtimes {location} tonight"
                    response = requests.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={
                            "key": google_api_key,
                            "cx": google_cse_id,
                            "q": query,
                            "num": 5
                        },
                        timeout=10
                    )

                    if response.status_code == 200:
                        data = response.json()
                        results = []

                        for item in data.get("items", []):
                            snippet = item.get("snippet", "")
                            title = item.get("title", "")

                            # Look for time patterns in snippets (e.g., "7:30pm", "10:00 AM")
                            time_pattern = r'\b(1[0-2]|[1-9]):([0-5][0-9])\s*(am|pm|AM|PM|a\.m\.|p\.m\.)\b'
                            times_in_snippet = re.findall(time_pattern, snippet)

                            if times_in_snippet:
                                showtimes = [f"{t[0]}:{t[1]} {t[2].upper().replace('.','')}" for t in times_in_snippet]
                                results.append({
                                    "source": title[:50],
                                    "showtimes": showtimes,
                                    "details": snippet[:200]
                                })
                            elif movie and movie.lower() in title.lower():
                                results.append({
                                    "source": title[:50],
                                    "details": snippet[:200]
                                })

                        if results:
                            return {
                                "found": True,
                                "location": location,
                                "movie": movie,
                                "results": results,
                                "note": "For exact times, call the theater directly"
                            }
        except Exception as e:
            logger.debug(f"Google search failed: {e}")

        # Fallback: Search for theater info and suggest calling
        return {
            "found": False,
            "location": location,
            "movie": movie,
            "message": f"Could not find specific showtimes for {movie or 'movies'} in {location}.",
            "action": "Use search_web to find the theater phone number, then use make_call to call them directly for accurate showtimes."
        }

    except Exception as e:
        logger.error(f"Error getting movie showtimes: {e}")
        return {
            "found": False,
            "error": str(e),
            "action": "Use make_call to call a local theater for showtimes"
        }


def search_web(query: str, settings: dict = None) -> dict:
    """Search the web - uses Google Custom Search if configured, else DuckDuckGo"""
    import requests

    # Try to load settings if not provided
    if settings is None:
        try:
            import json
            from pathlib import Path
            settings_file = Path(__file__).parent / "settings.json"
            if settings_file.exists():
                settings = json.loads(settings_file.read_text())
        except:
            settings = {}

    # Check for Google Custom Search API
    api_keys = settings.get("api_keys", {})
    google_api_key = api_keys.get("GOOGLE_API_KEY")
    google_cse_id = api_keys.get("GOOGLE_CSE_ID")

    if google_api_key and google_cse_id:
        # Use Google Custom Search
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": google_api_key,
                    "cx": google_cse_id,
                    "q": query,
                    "num": 5
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                results = []
                phones = []

                for item in data.get("items", []):
                    result = {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", "")
                    }
                    results.append(result)

                    # Extract phone numbers from snippet
                    phone_pattern = r'[\(]?\d{3}[\)]?[-.\s]?\d{3}[-.\s]?\d{4}'
                    snippet_phones = re.findall(phone_pattern, item.get("snippet", ""))
                    phones.extend(snippet_phones)

                unique_phones = list(dict.fromkeys(phones[:5]))

                return {
                    "found": len(results) > 0,
                    "phone_numbers": unique_phones,
                    "results": results,
                    "query": query,
                    "source": "google"
                }
        except Exception as e:
            logger.error(f"Google search failed, falling back to DuckDuckGo: {e}")

    # Fallback to DuckDuckGo
    try:
        search_url = "https://html.duckduckgo.com/html/"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.post(
            search_url,
            data={"q": query},
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "found": False,
                "error": f"Search failed with status {response.status_code}"
            }

        html = response.text

        # Extract phone numbers from results
        phone_pattern = r'[\(]?\d{3}[\)]?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, html)

        # Extract result snippets using BeautifulSoup if available, else regex
        results = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            for result_div in soup.select('.result')[:5]:
                title_elem = result_div.select_one('.result__a')
                snippet_elem = result_div.select_one('.result__snippet')

                if title_elem:
                    result = {"title": title_elem.get_text(strip=True)}
                    if snippet_elem:
                        result["snippet"] = snippet_elem.get_text(strip=True)
                    results.append(result)
        except ImportError:
            # Fallback to regex if BeautifulSoup not available
            result_pattern = r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'

            titles = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html, re.DOTALL)

            for i, title in enumerate(titles[:5]):
                result = {"title": title.strip()}
                if i < len(snippets):
                    # Strip HTML tags from snippet
                    snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
                    result["snippet"] = snippet
                results.append(result)

        # Deduplicate and clean phone numbers
        unique_phones = list(dict.fromkeys(phones[:5]))

        return {
            "found": len(unique_phones) > 0 or len(results) > 0,
            "phone_numbers": unique_phones,
            "results": results,
            "query": query,
            "source": "duckduckgo"
        }

    except Exception as e:
        logger.error(f"Error searching web: {e}")
        return {
            "found": False,
            "error": str(e)
        }


def search_web_google(query: str, settings: dict = None) -> dict:
    """Search using Google Custom Search API (if configured)"""
    # This is a placeholder - would need API key configuration
    # Fall back to DuckDuckGo
    return search_web(query)


def make_call(phone_number: str, objective: str, agent_id: str = "personal_assistant") -> dict:
    """Initiate a phone call via the local API"""
    import requests

    try:
        # Clean phone number
        clean_phone = re.sub(r'\D', '', phone_number)
        if len(clean_phone) == 10:
            clean_phone = "1" + clean_phone

        # Call the local API to start the call
        response = requests.post(
            "http://localhost:8765/api/call",
            json={
                "phone": clean_phone,
                "objective": objective,
                "agent_id": agent_id,
                "context": {}
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "message": f"Call started to {phone_number}",
                "phone": clean_phone,
                "objective": objective,
                "call_id": result.get("call_id")
            }
        else:
            return {
                "success": False,
                "error": f"API returned {response.status_code}: {response.text}"
            }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Could not connect to call API. Is the server running?"
        }
    except Exception as e:
        logger.error(f"Error making call: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def send_sms(phone_number: str, message: str) -> dict:
    """Send an SMS message"""
    try:
        from sim7600_modem import SIM7600Modem

        # Clean phone number
        clean_phone = re.sub(r'\D', '', phone_number)
        if len(clean_phone) == 10:
            clean_phone = "1" + clean_phone

        modem = SIM7600Modem()
        if not modem.connect():
            return {
                "success": False,
                "error": "Could not connect to modem"
            }

        success = modem.send_sms(clean_phone, message)
        modem.disconnect()

        if success:
            return {
                "success": True,
                "message": f"SMS sent to {phone_number}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to send SMS"
            }

    except Exception as e:
        logger.error(f"Error sending SMS: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def check_calendar_availability(date_str: str) -> dict:
    """Check calendar availability for a specific date"""
    try:
        from calendar_integration import get_calendar_integration
        from datetime import datetime

        calendar = get_calendar_integration()
        if not calendar:
            return {
                "success": False,
                "error": "Calendar not configured"
            }

        # Parse date
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid date format: {date_str}. Use YYYY-MM-DD"
            }

        slots = calendar.check_availability(target_date)

        if not slots:
            return {
                "success": True,
                "date": date_str,
                "available": False,
                "message": f"No available times on {date_str}",
                "times": []
            }

        # Sort and format times
        sorted_slots = sorted(slots, key=lambda s: s.start)
        times = [slot.start.strftime("%I:%M %p").lstrip("0") for slot in sorted_slots]

        return {
            "success": True,
            "date": date_str,
            "available": True,
            "times": times,
            "message": f"Available times on {date_str}: {', '.join(times)}"
        }

    except Exception as e:
        logger.error(f"Error checking calendar: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def book_calendar_appointment(date_str: str, time_str: str, name: str, phone: str, email: str = "") -> dict:
    """Book a calendar appointment"""
    try:
        from calendar_integration import get_calendar_integration, TimeSlot
        from datetime import datetime, timedelta

        calendar = get_calendar_integration()
        if not calendar:
            return {
                "success": False,
                "error": "Calendar not configured"
            }

        # Parse date and time
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            target_time = datetime.strptime(time_str, "%H:%M").time()
            target_datetime = datetime.combine(target_date, target_time)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid date/time format: {e}. Use YYYY-MM-DD and HH:MM"
            }

        # Check if time is actually available
        slots = calendar.check_availability(target_date)
        matching_slot = None

        for slot in slots:
            slot_time = slot.start.replace(tzinfo=None) if slot.start.tzinfo else slot.start
            if abs((slot_time - target_datetime).total_seconds()) < 300:  # 5 min tolerance
                matching_slot = slot
                break

        if not matching_slot:
            # Time not available - return available times
            available_times = [s.start.strftime("%I:%M %p").lstrip("0") for s in sorted(slots, key=lambda x: x.start)]
            return {
                "success": False,
                "error": f"Time {time_str} is not available on {date_str}",
                "available_times": available_times,
                "message": f"That time isn't available. Available times: {', '.join(available_times)}" if available_times else "No times available on that date"
            }

        # Book the appointment
        result = calendar.book_appointment(
            slot=matching_slot,
            name=name,
            email=email or f"{phone}@phone.booking",
            phone=phone,
            notes="Booked via AI phone agent"
        )

        if result.success:
            formatted_time = target_datetime.strftime("%I:%M %p").lstrip("0")
            formatted_date = target_date.strftime("%A, %B %d")
            return {
                "success": True,
                "confirmation_id": result.confirmation_id,
                "booked_date": date_str,
                "booked_time": time_str,
                "formatted": f"{formatted_date} at {formatted_time}",
                "name": name,
                "message": f"Successfully booked appointment for {name} on {formatted_date} at {formatted_time}"
            }
        else:
            return {
                "success": False,
                "error": result.message
            }

    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# =============================================================================
# Tool Dispatcher
# =============================================================================

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return JSON result"""

    if tool_name == "search_contacts":
        result = search_contacts(tool_input.get("query", ""))
    elif tool_name == "search_web":
        result = search_web(tool_input.get("query", ""))
    elif tool_name == "get_movie_showtimes":
        result = get_movie_showtimes(
            tool_input.get("location", ""),
            tool_input.get("movie", "")
        )
    elif tool_name == "make_call":
        result = make_call(
            tool_input.get("phone_number", ""),
            tool_input.get("objective", ""),
            tool_input.get("agent_id", "sales_rep")
        )
    elif tool_name == "send_sms":
        result = send_sms(
            tool_input.get("phone_number", ""),
            tool_input.get("message", "")
        )
    elif tool_name == "check_calendar_availability":
        result = check_calendar_availability(
            tool_input.get("date", "")
        )
    elif tool_name == "book_calendar_appointment":
        result = book_calendar_appointment(
            tool_input.get("date", ""),
            tool_input.get("time", ""),
            tool_input.get("name", ""),
            tool_input.get("phone", ""),
            tool_input.get("email", "")
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)
