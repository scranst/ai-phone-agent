"""
AI-Powered SMS Handler

Multi-agent SMS processing - routes to different AI agents based on context.
Main user gets Personal Assistant (smart model), others get Receptionist (fast model).
"""

import logging
import json
import re
from typing import Optional, Callable
from datetime import datetime

import database
import api_keys
from agents import get_agent_manager, Agent, MODEL_IDS
from ai_tools import ASSISTANT_TOOLS, search_contacts, search_web, get_movie_showtimes

logger = logging.getLogger(__name__)


class SMSAIHandler:
    """
    Multi-agent SMS handler.

    Routes messages to appropriate agent based on sender:
    - Main user → Personal Assistant (Opus - reasoning model)
    - Others → Receptionist (Haiku - fast model)
    """

    def __init__(self, primary_number: str, settings: dict = None):
        self.primary_number = self._normalize_phone(primary_number)
        self.pending_calls = []  # Queue of calls to make

        # Callbacks for actions that need external resources
        self._send_sms_callback: Optional[Callable] = None
        self._queue_call_callback: Optional[Callable] = None

    @property
    def agent_manager(self):
        """Get agent manager with fresh settings"""
        return get_agent_manager()

    @property
    def settings(self):
        """Get fresh settings"""
        return api_keys.get_settings()

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to digits only"""
        if not phone:
            return ""
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            digits = "1" + digits
        return digits

    def is_main_user(self, sender: str) -> bool:
        """Check if the sender is the main user"""
        normalized = self._normalize_phone(sender)
        return normalized == self.primary_number

    def on_send_sms(self, callback: Callable[[str, str], bool]):
        """Register callback for sending SMS: callback(phone, message) -> success"""
        self._send_sms_callback = callback

    def on_queue_call(self, callback: Callable[[dict], None]):
        """Register callback for queuing calls: callback(call_info)"""
        self._queue_call_callback = callback

    def process_message(self, sender: str, message: str) -> Optional[str]:
        """
        Process an incoming SMS message using AI.

        Args:
            sender: Phone number that sent the message
            message: The SMS content

        Returns:
            Response to send back (or None if no response needed)
        """
        is_main = self.is_main_user(sender)

        logger.info(f"Processing SMS from {'main user' if is_main else sender}: {message[:50]}...")

        if is_main:
            return self._process_main_user_message(sender, message)
        else:
            return self._process_other_user_message(sender, message)

    def _process_main_user_message(self, sender: str, message: str) -> Optional[str]:
        """Process message from main user using Personal Assistant agent"""

        # Get the Personal Assistant agent
        agent = self.agent_manager.get_agent_for_sms(sender, is_main_user=True)
        logger.info(f"Using agent: {agent.name} (model: {agent.model_id})")

        # Build context for persona substitution
        context = {
            "my_name": self.settings.get("MY_NAME", "your boss"),
            "company_name": self.settings.get("COMPANY", "the company"),
            "location": self.settings.get("CITY", "Las Vegas"),
        }

        # Get conversation history
        history = self._get_conversation_history(sender)

        # Build system prompt from agent persona
        base_prompt = self.agent_manager.get_system_prompt(agent, context)

        system_prompt = f"""{base_prompt}

CURRENT CONTEXT:
- Location: {context['location']}
- Your boss: {context['my_name']}

{history}

Keep responses SHORT - this is SMS."""

        messages = [{"role": "user", "content": message}]

        # Call LLM with the agent's model
        try:
            client = api_keys.get_anthropic_client()
            response = client.messages.create(
                model=agent.model_id,  # Use agent's model (Opus for personal assistant)
                max_tokens=1000,
                system=system_prompt,
                tools=ASSISTANT_TOOLS,
                messages=messages
            )

            # Process response and any tool calls
            return self._handle_llm_response(response, messages, system_prompt, agent.model_id)

        except Exception as e:
            logger.error(f"Error processing main user message: {e}")
            return f"Error: {str(e)[:100]}"

    def _handle_llm_response(self, response, messages: list, system_prompt: str, model_id: str = None) -> str:
        """Handle LLM response, executing any tool calls"""

        # Use provided model or default to opus for tool handling
        if model_id is None:
            model_id = "claude-opus-4-20250514"

        final_text = ""

        # Process all content blocks
        for block in response.content:
            if block.type == "text":
                final_text += block.text
            elif block.type == "tool_use":
                # Execute the tool
                tool_result = self._execute_tool(block.name, block.input)

                logger.info(f"Tool {block.name}: {json.dumps(tool_result)[:200]}")

                # Continue conversation with tool result
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result)
                    }]
                })

                # Get next response using the same model
                try:
                    client = api_keys.get_anthropic_client()
                    next_response = client.messages.create(
                        model=model_id,
                        max_tokens=1000,
                        system=system_prompt,
                        tools=ASSISTANT_TOOLS,
                        messages=messages
                    )

                    # Recursively handle (in case of more tool calls)
                    return self._handle_llm_response(next_response, messages, system_prompt, model_id)

                except Exception as e:
                    logger.error(f"Error in tool follow-up: {e}")
                    return f"Done, but had an error: {str(e)[:50]}"

        return final_text.strip() if final_text else "Done."

    def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool and return the result"""

        logger.info(f"Executing tool: {tool_name} with {tool_input}")

        if tool_name == "search_contacts":
            return search_contacts(tool_input.get("query", ""))

        elif tool_name == "search_web":
            return search_web(tool_input.get("query", ""))

        elif tool_name == "get_movie_showtimes":
            return get_movie_showtimes(
                tool_input.get("location", "Las Vegas"),
                tool_input.get("movie", "")
            )

        elif tool_name == "make_call":
            phone = tool_input.get("phone_number", "")
            objective = tool_input.get("objective", "")

            # Clean phone number
            clean_phone = re.sub(r'\D', '', phone)
            if len(clean_phone) == 10:
                clean_phone = "1" + clean_phone

            # Look up lead if possible
            lead = None
            try:
                lead = database.get_lead_by_phone(clean_phone)
            except:
                pass

            call_info = {
                'phone': clean_phone,
                'objective': objective,
                'lead_id': lead['id'] if lead else None,
                'contact_name': f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip() if lead else phone,
                'agent_id': tool_input.get('agent_id', 'personal_assistant')  # Default to personal_assistant for outbound
            }

            # Queue the call
            self.pending_calls.append(call_info)

            if self._queue_call_callback:
                self._queue_call_callback(call_info)

            return {
                "success": True,
                "message": f"Call queued to {call_info['contact_name']} at {clean_phone}",
                "objective": objective
            }

        elif tool_name == "send_sms":
            phone = tool_input.get("phone_number", "")
            sms_message = tool_input.get("message", "")

            # Clean phone number
            clean_phone = re.sub(r'\D', '', phone)
            if len(clean_phone) == 10:
                clean_phone = "1" + clean_phone

            # Send via callback if available
            if self._send_sms_callback:
                success = self._send_sms_callback(clean_phone, sms_message)
                if success:
                    # Log to database
                    try:
                        my_phone = self.settings.get("CALLBACK_NUMBER", "")
                        database.save_message({
                            "channel": "sms",
                            "direction": "outbound",
                            "from_address": my_phone,
                            "to_address": clean_phone,
                            "body": sms_message,
                            "status": "sent",
                            "provider": "modem",
                            "sent_at": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Failed to log outbound SMS: {e}")

                    return {"success": True, "message": f"SMS sent to {phone}"}
                else:
                    return {"success": False, "error": "Failed to send SMS"}
            else:
                return {"success": False, "error": "SMS sending not configured"}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _process_other_user_message(self, sender: str, message: str) -> Optional[str]:
        """Process message from non-main user using Receptionist agent"""

        # Check if autopilot is disabled for this specific thread
        if database.is_thread_autopilot_disabled(sender):
            logger.info(f"Autopilot disabled for thread {sender}")
            return None

        # Get the Receptionist agent
        agent = self.agent_manager.get_agent_for_sms(sender, is_main_user=False)
        logger.info(f"Using agent: {agent.name} (model: {agent.model_id})")

        # Build context
        context = {
            "my_name": self.settings.get("MY_NAME", "Assistant"),
            "company_name": self.settings.get("COMPANY", "the company"),
            "location": self.settings.get("CITY", ""),
        }

        # Get lead context
        lead = None
        lead_context = ""
        try:
            lead = database.get_lead_by_phone(sender)
            if lead:
                lead_context = f"\nCaller info: {lead.get('first_name', '')} {lead.get('last_name', '')} at {lead.get('company', 'unknown company')}"
        except:
            pass

        # Get conversation history
        history = self._get_conversation_history(sender)

        # Build system prompt from agent persona
        base_prompt = self.agent_manager.get_system_prompt(agent, context)

        prompt = f"""{base_prompt}
{lead_context}
{history}

They just sent: "{message}"

Write a brief, natural SMS reply (under 160 chars if possible). Be conversational and human-like."""

        try:
            client = api_keys.get_anthropic_client()
            response = client.messages.create(
                model=agent.model_id,  # Use agent's model (Haiku for receptionist)
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()[:300]
        except Exception as e:
            logger.error(f"Error generating autopilot reply: {e}")
            return None

    def _get_conversation_history(self, phone: str, limit: int = 5) -> str:
        """Get recent conversation history with a phone number"""
        try:
            messages = database.get_conversation_messages(phone, limit=limit)
            if not messages:
                return ""

            history = "\nRecent conversation:\n"
            for m in messages[-limit:]:
                direction = "Them" if m.get('direction') == 'inbound' else "Me"
                body = m.get('body', '')[:100]
                history += f"{direction}: {body}\n"

            return history
        except:
            return ""

    def get_pending_call(self) -> Optional[dict]:
        """Get and remove the next pending call from the queue"""
        if self.pending_calls:
            return self.pending_calls.pop(0)
        return None

    def has_pending_calls(self) -> bool:
        """Check if there are pending calls"""
        return len(self.pending_calls) > 0
