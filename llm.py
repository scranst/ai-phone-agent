"""
LLM Engine using Claude Haiku

Generates conversational responses for phone calls.
"""

import logging
from typing import Optional
import anthropic

import config

logger = logging.getLogger(__name__)


class LLMEngine:
    """Claude Haiku client for generating phone conversation responses"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307"
    ):
        """
        Initialize LLM engine.

        Args:
            api_key: Anthropic API key (uses config if not provided)
            model: Model to use (haiku is cheapest/fastest)
        """
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

        self.system_prompt = ""
        self.conversation_history = []

        logger.info(f"LLM engine initialized (model={model})")

    def set_objective(self, objective: str, context: dict):
        """
        Set the call objective and context.

        Args:
            objective: What the AI should accomplish
            context: Additional context (name, business info, etc.)
        """
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        self.system_prompt = f"""You are a voice chatbot having a conversation. The other person has just sent you a message.

YOUR GOAL:
{objective}

ABOUT YOU:
{context_str}

RULES:
- Reply with SHORT responses (1-2 sentences)
- Just say words - no asterisks, no actions like *dials* or *waits*
- You are trying to accomplish YOUR goal - you need something from them
- Do not make up information you don't have"""

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
        # Create a prompt for the initial greeting
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": "Hello?"
                }]
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
