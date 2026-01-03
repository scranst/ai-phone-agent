"""
LLM Engine - supports Claude Haiku and local Ollama models

Generates conversational responses for phone calls.
"""

import logging
from typing import Optional
import anthropic

import config

logger = logging.getLogger(__name__)

# Try to import ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMEngine:
    """LLM client for generating phone conversation responses"""

    def __init__(
        self,
        provider: str = "claude",  # "claude" or "ollama"
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM engine.

        Args:
            provider: "claude" for Claude API, "ollama" for local Ollama
            model: Model to use (defaults based on provider)
            api_key: Anthropic API key (only for Claude)
        """
        self.provider = provider

        if provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self.model = model or "llama3:8b"
            self.client = None
            logger.info(f"LLM engine initialized (provider=ollama, model={self.model})")
        else:
            self.api_key = api_key or config.ANTHROPIC_API_KEY
            self.model = model or "claude-3-5-haiku-latest"
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"LLM engine initialized (provider=claude, model={self.model})")

        self.system_prompt = ""
        self.conversation_history = []

    def set_objective(self, objective: str, context: dict):
        """
        Set the call objective and context.

        Args:
            objective: What the AI should accomplish
            context: Additional context (name, business info, etc.)
        """
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

        # Build base prompt
        self.system_prompt = f"""You are an AI assistant making a phone call on behalf of a client.

CURRENT DATE/TIME: {current_datetime}

YOUR GOAL:
{objective}

YOUR INFORMATION:
{context_str}

CRITICAL RULES:
- In your FIRST response, briefly mention you're an AI assistant calling on behalf of someone
- Keep responses SHORT (1-2 sentences max)
- LISTEN CAREFULLY - never ask for information they already provided
- When they give you an answer, acknowledge it and move on - don't ask again
- Be conversational and natural, not robotic or overly formal
- No asterisks or action words like *pauses*
- If they ask a question, answer it directly
- When the goal is achieved, wrap up the call naturally"""

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

        self.conversation_history = []
        logger.info(f"Objective set: {objective[:100]}...")

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
            if self.provider == "ollama":
                # Use Ollama for local inference
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.extend(self.conversation_history)

                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"num_predict": 150}  # Keep responses short
                )
                assistant_text = response["message"]["content"].strip()
            else:
                # Use Claude API
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=150,  # Keep responses short
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

            return assistant_text

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm sorry, I'm having trouble responding. Could you repeat that?"

    def get_initial_greeting(self) -> str:
        """
        Generate initial greeting when call is answered.

        Returns:
            Initial greeting text
        """
        try:
            if self.provider == "ollama":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Hello?"}
                ]
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"num_predict": 100}
                )
                greeting = response["message"]["content"].strip()
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": "Hello?"}]
                )
                greeting = response.content[0].text.strip()

            # Add the exchange to history
            self.conversation_history = [
                {"role": "user", "content": "Hello?"},
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

            if self.provider == "ollama":
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": summary_system},
                        {"role": "user", "content": summary_prompt}
                    ],
                    options={"num_predict": 200}
                )
                return response["message"]["content"].strip()
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
