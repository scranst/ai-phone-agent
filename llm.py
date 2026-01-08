"""
LLM Engine - supports Claude and OpenAI models

Generates conversational responses for phone calls.
Supports calendar integration for appointment scheduling.
Uses multi-agent architecture for different call roles.
"""

import logging
from typing import Optional
from datetime import date, timedelta
import anthropic
import openai

import config
from agents import get_agent_manager, Agent, ModelTier, MODEL_IDS

logger = logging.getLogger(__name__)


class LLMEngine:
    """LLM client for generating phone conversation responses"""

    def __init__(
        self,
        provider: str = "claude",  # "claude" or "openai"
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM engine.

        Args:
            provider: "claude" for Claude API, "openai" for OpenAI API
            model: Model to use (defaults based on provider)
            api_key: API key (uses config if not provided)
        """
        self.provider = provider
        # Load settings to get agent configurations (tools, etc.)
        import json
        import os
        try:
            settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
            with open(settings_path) as f:
                settings = json.load(f)
        except:
            settings = None
        self.agent_manager = get_agent_manager(settings)
        self.current_agent: Optional[Agent] = None

        if provider == "openai":
            self.api_key = api_key or config.OPENAI_API_KEY
            self.model = model or "gpt-4o-mini"
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"LLM engine initialized (provider=openai, model={self.model})")
        else:
            # Default to Claude
            self.provider = "claude"
            self.api_key = api_key or config.ANTHROPIC_API_KEY
            self.model = model or "claude-3-5-haiku-latest"
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"LLM engine initialized (provider=claude, model={self.model})")

        self.system_prompt = ""
        self.conversation_history = []
        self.is_main_user_mode = False  # Whether tools are enabled
        self.has_calendar_tools = False  # Whether calendar tools are enabled

    def set_model(self, model_id: str):
        """Change the model being used"""
        self.model = model_id
        logger.info(f"LLM model changed to: {model_id}")

    def reload_api_key(self):
        """Hot-reload API key from settings without restarting service"""
        import json
        import os
        try:
            settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
            with open(settings_path) as f:
                settings = json.load(f)
            api_keys = settings.get("api_keys", {})

            if self.provider == "openai":
                new_key = api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
                if new_key and new_key != self.api_key:
                    self.api_key = new_key
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logger.info("OpenAI API key hot-reloaded")
            else:
                new_key = api_keys.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
                if new_key and new_key != self.api_key:
                    self.api_key = new_key
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logger.info("Anthropic API key hot-reloaded")
        except Exception as e:
            logger.error(f"Failed to reload API key: {e}")

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Simple one-shot text generation without conversation context.
        Useful for auto-replies, summaries, etc.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Generated text
        """
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"LLM generate error: {e}")
            return ""

    def _get_knowledge_content(self, objective: str, context: dict) -> str:
        """Get relevant knowledge base content for the call"""
        try:
            import knowledge_base

            # Search for relevant content based on objective
            knowledge = knowledge_base.get_knowledge_for_prompt(
                objective=objective,
                context=context,
                max_tokens=1500  # Limit to avoid prompt bloat
            )

            return knowledge

        except ImportError:
            logger.debug("Knowledge base module not available")
            return ""
        except Exception as e:
            logger.debug(f"Could not get knowledge content: {e}")
            return ""

    def _get_agent_knowledge(self, knowledge_path: str) -> str:
        """Load agent-specific knowledge base from file"""
        try:
            from pathlib import Path

            kb_file = Path(__file__).parent / knowledge_path
            if kb_file.exists():
                content = kb_file.read_text()
                # Limit to avoid prompt bloat
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                return content
            else:
                logger.debug(f"Agent knowledge base not found: {knowledge_path}")
                return ""
        except Exception as e:
            logger.debug(f"Could not load agent knowledge: {e}")
            return ""

    def _get_calendar_availability(self) -> str:
        """Get upcoming calendar availability for the next 7 days (parallel fetching)"""
        try:
            from calendar_integration import get_calendar_integration
            from concurrent.futures import ThreadPoolExecutor, as_completed

            calendar = get_calendar_integration()
            if not calendar:
                return ""

            self._calendar = calendar  # Store for booking later
            today = date.today()

            # Fetch all 7 days in parallel for speed
            def fetch_day(day_offset):
                check_date = today + timedelta(days=day_offset)
                slots = calendar.check_availability(check_date)
                return (day_offset, check_date, slots)

            results = {}
            with ThreadPoolExecutor(max_workers=7) as executor:
                futures = {executor.submit(fetch_day, i): i for i in range(7)}
                for future in as_completed(futures):
                    try:
                        day_offset, check_date, slots = future.result()
                        results[day_offset] = (check_date, slots)
                    except:
                        pass

            # Build output in order - show ALL available times clearly
            lines = ["CALENDAR AVAILABILITY (ONLY offer times from this list):"]
            for i in range(7):
                if i not in results:
                    continue
                check_date, slots = results[i]
                if slots:
                    if i == 0:
                        day_name = f"Today"
                    elif i == 1:
                        day_name = f"Tomorrow"
                    else:
                        day_name = check_date.strftime("%A")
                    date_str = check_date.strftime("%Y-%m-%d")
                    # Sort slots by time and show all so AI doesn't make up times
                    sorted_slots = sorted(slots, key=lambda s: s.start)
                    slot_times = [slot.start.strftime("%I:%M %p").lstrip("0") for slot in sorted_slots]
                    lines.append(f"  {day_name} ({date_str}): {', '.join(slot_times)}")

            if len(lines) == 1:
                lines.append("  No availability in the next 7 days")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Could not get calendar availability: {e}")
            return ""

    def _is_main_user(self, phone: str, settings: dict) -> bool:
        """Check if the phone number belongs to the main user"""
        if not phone:
            return False
        # Normalize phone numbers for comparison
        clean_phone = ''.join(c for c in phone if c.isdigit())
        primary_phone = settings.get("sms", {}).get("PRIMARY_PHONE", "")
        clean_primary = ''.join(c for c in primary_phone if c.isdigit())
        callback = settings.get("CALLBACK_NUMBER", "")
        clean_callback = ''.join(c for c in callback if c.isdigit())

        # Check last 10 digits (handles country codes)
        return (clean_phone and clean_primary and clean_phone[-10:] == clean_primary[-10:]) or \
               (clean_phone and clean_callback and clean_phone[-10:] == clean_callback[-10:])

    def set_objective(self, objective: str, context: dict, caller_phone: str = None, direction: str = "outgoing", enable_tools: bool = False, agent: Agent = None):
        """
        Set the call objective and context.

        Args:
            objective: What the AI should accomplish
            context: Additional context (name, business info, etc.)
            caller_phone: Phone number of the other party (for main user detection)
            direction: "outgoing" or "incoming"
            enable_tools: Force enable tools (for main-user-initiated calls via SMS)
            agent: Optional agent to use for this call (overrides auto-detection)
        """
        context = context or {}

        # Load settings to check if this is the main user
        try:
            import json
            with open("settings.json") as f:
                settings = json.load(f)
        except:
            settings = {}

        is_main_user = self._is_main_user(caller_phone, settings)

        # Get the appropriate agent for this call if not specified
        if agent is None:
            # Check if agent_id is specified in context
            agent_id = context.get("AGENT_ID")
            if agent_id:
                agent = self.agent_manager.get_agent(agent_id)

            # Fall back to auto-detection if agent not found
            if agent is None:
                is_campaign = context.get("IS_CAMPAIGN", False)
                agent = self.agent_manager.get_agent_for_call(
                    phone=caller_phone or "",
                    direction=direction,
                    is_main_user=is_main_user,
                    is_campaign=is_campaign
                )

        self.current_agent = agent
        logger.info(f"Using agent: {agent.name} (model: {agent.model_id})")

        # Switch to agent's model and provider
        if agent.model_id.startswith("claude"):
            if self.provider != "claude":
                logger.info(f"Switching provider from {self.provider} to claude for agent {agent.name}")
                self.provider = "claude"
                self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.model = agent.model_id
        elif agent.model_id.startswith("gpt"):
            if self.provider != "openai":
                logger.info(f"Switching provider from {self.provider} to openai for agent {agent.name}")
                self.provider = "openai"
                self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            self.model = agent.model_id
        else:
            # Unknown model format, keep as-is
            self.model = agent.model_id

        # Enable tools based on agent config and call context
        self.is_main_user_mode = is_main_user and "search_web" in agent.tools
        # Enable calendar tools for any agent that has them (sales_rep, receptionist)
        # Check for both short names (from settings) and full names (from ai_tools.py)
        self.has_calendar_tools = (
            "book_calendar" in agent.tools or
            "check_calendar" in agent.tools or
            "book_calendar_appointment" in agent.tools or
            "check_calendar_availability" in agent.tools
        )
        self.agent_tools = agent.tools  # Store agent's tools for filtering
        logger.info(f"Agent tools: {agent.tools}, has_calendar_tools: {self.has_calendar_tools}")
        main_objective = settings.get("MAIN_OBJECTIVE", "")

        # Determine the actual objective based on caller and agent
        if is_main_user:
            logger.info("Caller is MAIN USER - acting as personal assistant")
            actual_objective = objective if objective else agent.objective
        else:
            if objective:
                actual_objective = objective
            elif agent.objective:
                actual_objective = agent.objective
            elif main_objective:
                actual_objective = main_objective
                logger.info(f"Using MAIN OBJECTIVE: {main_objective[:80]}...")
            else:
                actual_objective = "Answer questions helpfully and professionally."

        # Check for special context keys (use get to not modify original dict)
        self.transfer_to = context.get("TRANSFER_TO")
        self.transfer_when = context.get("TRANSFER_IF") or context.get("TRANSFER_WHEN")
        self.strict_budget = context.get("STRICT_BUDGET", "")

        # Log transfer config if present
        if self.transfer_to:
            logger.info(f"Transfer configured: TO={self.transfer_to}, IF={self.transfer_when}")
        if self.strict_budget:
            logger.info(f"Strict budget: {self.strict_budget}")

        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        # Get current date/time for context awareness
        from datetime import datetime
        now = datetime.now()
        current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")

        # Build context for persona substitution
        persona_context = {
            "my_name": settings.get("MY_NAME", context.get("MY_NAME", "your boss")),
            "company_name": settings.get("COMPANY", context.get("COMPANY", "the company")),
            "location": settings.get("CITY", context.get("CITY", "")),
        }

        # Build base prompt from agent's persona
        base_persona = self.agent_manager.get_system_prompt(agent, persona_context)

        # Add tool instructions if agent has tools
        tool_instructions = ""
        if is_main_user and agent.tools:
            tool_list = "\n".join([f"  * {tool}" for tool in agent.tools])
            tool_instructions = f"""
You have TOOLS available to help accomplish tasks:
{tool_list}
- If they ask you to call a place, FIRST use search_web to find the phone number, THEN use make_call
- If they mention a person by name, use search_contacts first to find their info
- Keep responses conversational - confirm what you're doing
- After using tools, summarize what you found or did"""
        elif enable_tools:
            # Main user initiated this call via SMS - we're calling a business
            tool_instructions = """
CRITICAL RULES:
- You are ON THE PHONE with a business/person - ASK THEM DIRECTLY for what you need
- DO NOT use search_web while on the call - you're talking to a human who can answer you!
- State your request clearly and simply in your FIRST response
- LISTEN to their answer - they will tell you what you need to know
- Keep responses SHORT (1-2 sentences max)
- When you get the information you need, thank them and end the call"""

        # Store direction for greeting generation
        self.call_direction = direction

        # Build direction-specific instructions
        if direction == "outgoing":
            direction_instructions = """
CALL DIRECTION: OUTBOUND (You are MAKING this call)
- YOU initiated this call - introduce yourself first
- State who you are and why you're calling
- Be polite but direct about your purpose
- Do NOT answer like you're receiving a call"""
        else:
            direction_instructions = """
CALL DIRECTION: INBOUND (You are RECEIVING this call)
- Someone called YOU - greet them professionally
- Ask how you can help them"""

        self.system_prompt = f"""{base_persona}

CURRENT DATE/TIME: {current_datetime}
{direction_instructions}

YOUR GOAL:
{actual_objective}

YOUR INFORMATION:
{context_str}
{tool_instructions}"""

        # Add agent-specific knowledge base content
        if agent.knowledge_base:
            agent_knowledge = self._get_agent_knowledge(agent.knowledge_base)
            if agent_knowledge:
                logger.info(f"Loading agent knowledge base: {agent.knowledge_base} ({len(agent_knowledge)} chars)")
                self.system_prompt += f"""

KNOWLEDGE BASE:
{agent_knowledge}

Use this knowledge to answer questions accurately. If asked about something covered in the knowledge above, reference it directly."""
            else:
                logger.warning(f"Agent knowledge base empty or not found: {agent.knowledge_base}")

        # Add calendar tool instructions if agent has calendar tools
        # NOTE: We do NOT pre-fetch availability - the AI will use tools on-demand for speed
        if self.has_calendar_tools:
            from datetime import date, timedelta
            today = date.today()
            # Just give date references, AI will fetch availability via tools
            date_hints = f"Today is {today.strftime('%A, %B %d, %Y')}. Tomorrow is {(today + timedelta(days=1)).strftime('%A %m/%d')}."

            self.system_prompt += f"""

*** SCHEDULING - YOU HAVE CALENDAR TOOLS ***

{date_hints}

You have two tools for scheduling:
1. check_calendar_availability(date) - Call this to get available times (use YYYY-MM-DD format)
2. book_calendar_appointment(date, time, name, phone, email) - Call this to book after they confirm

SCHEDULING FLOW:
1. When they want to schedule, call check_calendar_availability to get REAL available times
2. Offer them times from the tool result (NEVER make up times!)
3. When they confirm, call book_calendar_appointment to actually book it
4. The tool will tell you if it succeeded or failed - respond based on the ACTUAL result

CRITICAL RULES:
- ALWAYS call check_calendar_availability BEFORE suggesting ANY times
- NEVER invent times - only offer what the tool returns
- Use info from YOUR INFORMATION section (LEAD_NAME, LEAD_PHONE, LEAD_EMAIL) when booking
- Only confirm booking AFTER the tool returns success"""

        # Add callback instructions if configured
        if self.transfer_when:
            self.system_prompt += f"""

CALLBACK INSTRUCTIONS:
If {self.transfer_when}, offer to have someone call them back.
Say something like: "I'll have them call you right back at this number. Thank you for your time!" then add [CALLBACK] at the end.
Do NOT say you'll transfer them - just offer a callback."""

        # Add strict budget instructions if configured
        if self.strict_budget:
            self.system_prompt += f"""

STRICT BUDGET: {self.strict_budget}
- You MUST keep the total under {self.strict_budget}
- Decline any upsells, add-ons, or extras that would push the total over budget
- If they offer something, ask the price first, then decline if it would exceed the budget
- Say something like "No thank you, I'm trying to keep it under {self.strict_budget}" """

        # Add relevant knowledge base content
        knowledge_content = self._get_knowledge_content(actual_objective, context)
        if knowledge_content:
            self.system_prompt += f"""

{knowledge_content}

Use this knowledge to answer questions accurately. If asked about something covered in the knowledge above, reference it directly."""

        self.conversation_history = []
        logger.info(f"Objective set: {actual_objective[:100]}...")

    def generate_response(self, user_text: str) -> str:
        """
        Generate AI response to user's speech.

        Args:
            user_text: What the user said

        Returns:
            AI response text
        """
        if not user_text.strip():
            return ""

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })

        try:
            if self.provider == "openai":
                # Use OpenAI API
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.extend(self.conversation_history)

                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=250,  # Increased to allow room for [BOOK:] tags
                    messages=messages
                )
                assistant_text = response.choices[0].message.content.strip()
            else:
                # Use Claude API with tools if main user mode OR agent has calendar tools
                logger.debug(f"generate_response: is_main_user_mode={self.is_main_user_mode}, has_calendar_tools={self.has_calendar_tools}")
                if self.is_main_user_mode:
                    assistant_text = self._generate_with_tools()
                elif self.has_calendar_tools:
                    logger.info("Using calendar tools for response generation")
                    assistant_text = self._generate_with_calendar_tools()
                else:
                    logger.debug("No tools enabled, using basic Claude API")
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=250,
                        system=self.system_prompt,
                        messages=self.conversation_history
                    )
                    assistant_text = response.content[0].text.strip()

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_text
            })

            logger.info(f"LLM response: {assistant_text}")

            # Check for booking request and process it
            import re
            booking_result = self._process_booking(assistant_text)

            # Strip the [BOOK:...] tag from spoken response
            clean_response = re.sub(r'\s*\[BOOK:[^\]]+\]', '', assistant_text).strip()

            # If booking failed, generate a follow-up response with FRESH availability
            if booking_result is False:
                # Get fresh calendar availability
                fresh_availability = ""
                try:
                    fresh_availability = self._get_calendar_availability()
                    if fresh_availability:
                        fresh_availability = f"\n\nUPDATED AVAILABILITY (use these times ONLY):\n{fresh_availability}"
                except:
                    pass

                # Add context about the failure and get a new response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": clean_response
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"[SYSTEM: The booking just failed - that time slot is not actually available. Apologize and offer ONLY times from this fresh availability:{fresh_availability}]"
                })

                # Generate recovery response
                try:
                    if self.provider == "openai":
                        recovery = self.client.chat.completions.create(
                            model=self.model,
                            max_tokens=150,
                            messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history
                        )
                        clean_response = recovery.choices[0].message.content.strip()
                    else:
                        recovery = self.client.messages.create(
                            model=self.model,
                            max_tokens=150,
                            system=self.system_prompt,
                            messages=self.conversation_history
                        )
                        clean_response = recovery.content[0].text.strip()

                    # Strip any new booking tags from recovery
                    clean_response = re.sub(r'\s*\[BOOK:[^\]]+\]', '', clean_response).strip()
                    logger.info(f"Recovery response after booking failure: {clean_response}")
                except Exception as e:
                    logger.error(f"Failed to generate recovery response: {e}")
                    clean_response = "I apologize, but that time slot isn't available after all. Could you suggest another time?"

            return clean_response

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm sorry, I'm having trouble responding. Could you repeat that?"

    def _generate_with_tools(self) -> str:
        """Generate response with tool use for main user mode (Claude only)"""
        from ai_tools import ASSISTANT_TOOLS, execute_tool

        # Initial API call with tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self.system_prompt,
            messages=self.conversation_history,
            tools=ASSISTANT_TOOLS
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            if not tool_uses:
                break

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            # Execute each tool and collect results
            tool_results = []
            for tool_use in tool_uses:
                logger.info(f"Executing tool: {tool_use.name} with input: {tool_use.input}")

                result = execute_tool(tool_use.name, tool_use.input)
                logger.info(f"Tool result: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

            # Add tool results to history
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Continue conversation with tool results
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=ASSISTANT_TOOLS
            )

        # Extract final text response
        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        return " ".join(text_blocks).strip() if text_blocks else "I've completed the task."

    def _generate_with_calendar_tools(self) -> str:
        """Generate response with calendar tools for agents that need scheduling (Claude only)"""
        from ai_tools import ASSISTANT_TOOLS, execute_tool

        # Filter to only calendar-related tools
        CALENDAR_TOOLS = [
            tool for tool in ASSISTANT_TOOLS
            if tool["name"] in ["check_calendar_availability", "book_calendar_appointment"]
        ]

        logger.info(f"Using calendar tools: {[t['name'] for t in CALENDAR_TOOLS]}")

        # Initial API call with calendar tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self.system_prompt,
            messages=self.conversation_history,
            tools=CALENDAR_TOOLS
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            if not tool_uses:
                break

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            # Execute each tool and collect results
            tool_results = []
            for tool_use in tool_uses:
                logger.info(f"Executing calendar tool: {tool_use.name} with input: {tool_use.input}")

                result = execute_tool(tool_use.name, tool_use.input)
                logger.info(f"Calendar tool result: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

            # Add tool results to history
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Continue conversation with tool results
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=CALENDAR_TOOLS
            )

        # Extract final text response
        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        return " ".join(text_blocks).strip() if text_blocks else "I've completed the task."

    def _process_booking(self, response: str) -> Optional[bool]:
        """
        Parse and process any booking request in the response.
        If a previous booking exists for this call, cancel it first.

        Returns:
            None if no booking was requested
            True if booking succeeded
            False if booking failed
        """
        import re

        # Look for [BOOK: ...] pattern
        match = re.search(r'\[BOOK:\s*date="([^"]+)"\s*time="([^"]+)"\s*name="([^"]+)"\s*phone="([^"]+)"(?:\s*email="([^"]*)")?\]', response)

        if not match:
            return None  # No booking requested

        booking_date = match.group(1)
        booking_time = match.group(2)
        name = match.group(3)
        phone = match.group(4)
        email = match.group(5) or ""

        logger.info(f"Booking request detected: {name} on {booking_date} at {booking_time}")

        # Try to book via calendar API
        if hasattr(self, '_calendar') and self._calendar:
            try:
                from calendar_integration import TimeSlot
                from datetime import datetime
                from datetime import date as date_type

                # Cancel previous booking if exists (prevents double-booking when changing times)
                if hasattr(self, '_last_booking_id') and self._last_booking_id:
                    logger.info(f"Canceling previous booking {self._last_booking_id} before rebooking")
                    if hasattr(self._calendar, 'cancel_booking'):
                        self._calendar.cancel_booking(self._last_booking_id)
                    self._last_booking_id = None

                # Parse date and time
                dt = datetime.strptime(f"{booking_date} {booking_time}", "%Y-%m-%d %H:%M")
                target_date = dt.date()

                # VALIDATE: Check if the requested time is actually available
                available_slots = self._calendar.check_availability(target_date)
                matching_slot = None
                for slot in available_slots:
                    # Check if requested time matches a slot start (5 min tolerance)
                    slot_time = slot.start.replace(tzinfo=None) if slot.start.tzinfo else slot.start
                    if abs((slot_time - dt).total_seconds()) < 300:
                        matching_slot = slot
                        break

                if not matching_slot:
                    # Time is not actually available - log and return failure
                    available_times = [s.start.strftime('%I:%M %p') for s in available_slots[:5]]
                    logger.warning(f"Booking rejected: {booking_time} on {booking_date} is not available. Available: {available_times}")
                    return False

                # Use the validated slot for booking
                slot = matching_slot

                # Attempt booking
                result = self._calendar.book_appointment(
                    slot=slot,
                    name=name,
                    email=email or f"{phone}@phone.booking",
                    phone=phone,
                    notes=f"Booked via AI phone agent"
                )

                if result.success:
                    logger.info(f"Booking successful: {result.confirmation_id}")
                    # Track this booking ID in case they want to change it
                    self._last_booking_id = result.confirmation_id
                    return True
                else:
                    logger.warning(f"Booking failed: {result.message}")
                    return False

            except Exception as e:
                logger.error(f"Booking error: {e}")
                return False
        else:
            logger.warning("No calendar integration available for booking")
            return False

    def get_initial_greeting(self) -> str:
        """
        Generate initial greeting when call is answered.
        AI always speaks first - content differs based on direction.

        Returns:
            Initial greeting text
        """
        try:
            direction = getattr(self, 'call_direction', 'incoming')

            if direction == "outgoing":
                # Outbound: SHORT intro - let them respond before pitching
                prompt = "The call just connected. Give a VERY SHORT intro (max 10-15 words) - just your name and a quick question like 'do you have a second?' STOP there and wait for their response. Do NOT pitch yet."
            else:
                # Inbound: Greet the caller and offer help
                prompt = "Someone just called. Greet them warmly and ask how you can help them. Be brief (1 sentence)."

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                greeting = response.choices[0].message.content.strip()
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                greeting = response.content[0].text.strip()

            # Start history with just the greeting (AI speaks first)
            self.conversation_history = [
                {"role": "assistant", "content": greeting}
            ]

            return greeting

        except Exception as e:
            logger.error(f"LLM greeting error: {e}")
            return "Hello, how can I help you today?"

    def should_end_call(self, last_response: str) -> bool:
        """
        Check if the conversation should end.

        Args:
            last_response: The last AI response

        Returns:
            True if call should end
        """
        end_phrases = [
            "goodbye", "bye", "have a great day", "have a good day",
            "take care", "thank you for your time", "thanks for your time"
        ]

        last_lower = last_response.lower()
        return any(phrase in last_lower for phrase in end_phrases)

    def should_transfer(self, last_response: str) -> bool:
        """Check if response contains callback trigger"""
        return "[CALLBACK]" in last_response or "[TRANSFER]" in last_response

    def get_transfer_number(self) -> Optional[str]:
        """Get the number to transfer to"""
        return getattr(self, 'transfer_to', None)

    def summarize_call(self, transcript: list, objective: str) -> str:
        """
        Generate a summary of the call.

        Args:
            transcript: List of conversation turns
            objective: The original call objective

        Returns:
            Summary of what happened in the call
        """
        if not transcript:
            return "No conversation recorded."

        # Format transcript for summary
        conversation = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}"
            for turn in transcript
        )

        try:
            summary_system = "You summarize phone calls concisely. Focus on: what was discussed, any commitments made, key information obtained, and the outcome. Keep it to 2-3 sentences."
            summary_prompt = f"CALL OBJECTIVE: {objective}\n\nTRANSCRIPT:\n{conversation}\n\nSummarize this call:"

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": summary_system},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    system=summary_system,
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Summary error: {e}")
            # Fallback to last assistant message
            for turn in reversed(transcript):
                if turn["role"] == "assistant":
                    return turn["content"]
            return "Call completed."


# Test function
def test_llm():
    print("Testing LLM...")

    llm = LLMEngine()
    llm.set_objective(
        objective="You are calling from City Library to remind someone their book is ready for pickup.",
        context={"library": "City Library", "book": "The Great Gatsby"}
    )

    # Simulate conversation
    exchanges = [
        "Hello?",
        "Oh great, what book is it?",
        "When do I need to pick it up by?",
        "Okay thanks!"
    ]

    for user_text in exchanges:
        print(f"\nUser: {user_text}")
        response = llm.generate_response(user_text)
        print(f"AI: {response}")

        if llm.should_end_call(response):
            print("\n[Call should end]")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_llm()
