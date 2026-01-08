"""
Multi-Agent Architecture

Different AI agents for different roles, each with their own model, persona, and capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(Enum):
    PERSONAL_ASSISTANT = "personal_assistant"
    RECEPTIONIST = "receptionist"
    SALES_REP = "sales_rep"
    RESEARCHER = "researcher"


class ModelTier(Enum):
    FAST = "haiku"      # claude-3-5-haiku-latest - Fast, cheap
    SMART = "sonnet"    # claude-sonnet-4-20250514 - Balanced
    REASONING = "opus"  # claude-opus-4-20250514 - Best reasoning


# Model ID mapping
MODEL_IDS = {
    ModelTier.FAST: "claude-3-5-haiku-latest",
    ModelTier.SMART: "claude-sonnet-4-20250514",
    ModelTier.REASONING: "claude-opus-4-20250514",
}


@dataclass
class Agent:
    """Base agent configuration"""
    id: str
    name: str
    type: AgentType
    model_tier: ModelTier
    objective: str  # What this agent is trying to accomplish
    persona: str    # How the agent should behave
    enabled: bool = True
    tools: List[str] = field(default_factory=list)
    triggers: Dict[str, Any] = field(default_factory=dict)
    icon: str = ""  # Emoji icon for UI
    knowledge_base: str = ""  # Path to agent-specific knowledge base file

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.model_tier]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "model_tier": self.model_tier.value,
            "model_id": self.model_id,
            "objective": self.objective,
            "persona": self.persona,
            "enabled": self.enabled,
            "tools": self.tools,
            "triggers": self.triggers,
            "icon": self.icon,
            "knowledge_base": self.knowledge_base,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        return cls(
            id=data["id"],
            name=data["name"],
            type=AgentType(data["type"]),
            model_tier=ModelTier(data["model_tier"]),
            objective=data.get("objective", ""),
            persona=data["persona"],
            enabled=data.get("enabled", True),
            tools=data.get("tools", []),
            triggers=data.get("triggers", {}),
            icon=data.get("icon", ""),
            knowledge_base=data.get("knowledge_base", ""),
        )


# =============================================================================
# Default Agent Definitions
# =============================================================================

DEFAULT_AGENTS = {
    "personal_assistant": Agent(
        id="personal_assistant",
        name="Personal Assistant",
        type=AgentType.PERSONAL_ASSISTANT,
        model_tier=ModelTier.REASONING,  # Opus for complex tasks
        objective="Help the main user accomplish any task by taking action - searching, calling, texting on their behalf.",
        persona="""You are a highly capable personal AI assistant. Your boss is texting/calling you.

Your capabilities:
- Search contacts/leads database for people and companies
- Web search to find any information (movie times, restaurants, businesses, facts)
- Make phone calls on their behalf
- Send text messages
- Complex research and reasoning

CRITICAL RULES:
- When asked to call/text ANYONE by name, ALWAYS use search_contacts FIRST
- Only use web search if the person is NOT in contacts
- TAKE ACTION - don't ask for info you can find yourself
- Never say "I recommend checking..." - actually DO the lookup
- Be concise but thorough
- You have their location, calendar, and context - use it""",
        tools=["search_web", "get_movie_showtimes", "make_call", "send_sms", "search_contacts"],
        triggers={"sms_from_main_user": True, "call_from_main_user": True},
        icon="🤖",
        knowledge_base="knowledge/personal_assistant.md",
    ),

    "receptionist": Agent(
        id="receptionist",
        name="Receptionist",
        type=AgentType.RECEPTIONIST,
        model_tier=ModelTier.FAST,  # Haiku for speed/cost
        objective="Answer incoming calls professionally, take messages, and schedule appointments.",
        persona="""You are a professional receptionist answering calls for {company_name}.

Your role:
- Answer incoming calls professionally
- Take messages
- Schedule appointments if calendar is available
- Transfer urgent matters
- Be warm, friendly, and efficient

CRITICAL RULES:
- Keep responses SHORT (1-2 sentences)
- Get caller's name and reason for calling
- If they want to schedule, check availability and book
- If urgent, offer to have someone call back
- Be natural, not robotic""",
        tools=["search_contacts"],
        triggers={"incoming_call": True, "sms_from_unknown": True},
        icon="📞",
        knowledge_base="knowledge/receptionist.md",
    ),

    "sales_rep": Agent(
        id="sales_rep",
        name="Sales Rep",
        type=AgentType.SALES_REP,
        model_tier=ModelTier.FAST,  # Haiku for speed/cost
        objective="Make outbound sales calls, qualify leads, handle objections, and book meetings.",
        persona="""You are a sales representative making outbound calls.

IMPORTANT: Your identity, product, company, and script are defined in your KNOWLEDGE BASE below.
Follow the knowledge base EXACTLY for who you are, what you're selling, and how to handle the call.

CRITICAL RULES:
- Keep responses SHORT (1-2 sentences)
- Be conversational, not pushy
- Listen more than you talk
- Focus on their needs, not features
- Ask for the meeting/demo
- Handle "not interested" gracefully""",
        tools=["search_contacts", "check_calendar", "book_calendar"],
        triggers={"outbound_campaign": True},
        icon="💼",
        knowledge_base="knowledge/sales_rep.md",
    ),

    "researcher": Agent(
        id="researcher",
        name="Researcher",
        type=AgentType.RESEARCHER,
        model_tier=ModelTier.SMART,  # Sonnet for web search
        objective="Perform deep research and lookups to support other agents with accurate information.",
        persona="""You are a research specialist. Other agents dispatch tasks to you when they need information.

Your role:
- Deep web searches
- Find specific information (prices, hours, availability)
- Extract data from websites
- Compile research results

CRITICAL RULES:
- Be thorough - try multiple search approaches
- Return structured, actionable information
- If you can't find something, explain what you tried
- Include sources when possible""",
        tools=["search_web", "get_movie_showtimes"],
        triggers={"dispatched_by_agent": True},
        icon="🔍",
        knowledge_base="knowledge/researcher.md",
    ),
}


class AgentManager:
    """Manages all agents and routing logic"""

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        self.agents: Dict[str, Agent] = {}
        self._load_agents()

    def _load_agents(self):
        """Load agents from settings or use defaults"""
        agent_configs = self.settings.get("agents", {})

        for agent_id, default_agent in DEFAULT_AGENTS.items():
            if agent_id in agent_configs:
                # Merge settings with defaults
                config = agent_configs[agent_id]
                self.agents[agent_id] = Agent(
                    id=agent_id,
                    name=config.get("name", default_agent.name),
                    type=default_agent.type,
                    model_tier=ModelTier(config.get("model_tier", default_agent.model_tier.value)),
                    objective=config.get("objective", default_agent.objective),
                    persona=config.get("persona", default_agent.persona),
                    enabled=config.get("enabled", default_agent.enabled),
                    tools=config.get("tools", default_agent.tools),
                    triggers=config.get("triggers", default_agent.triggers),
                    icon=config.get("icon", default_agent.icon),
                    knowledge_base=config.get("knowledge_base", default_agent.knowledge_base),
                )
            else:
                self.agents[agent_id] = default_agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agent_for_sms(self, sender: str, is_main_user: bool) -> Agent:
        """Route SMS to appropriate agent"""
        if is_main_user:
            agent = self.agents.get("personal_assistant")
            if agent and agent.enabled:
                return agent

        # Default to receptionist for others
        return self.agents.get("receptionist", DEFAULT_AGENTS["receptionist"])

    def get_agent_for_call(self, phone: str, direction: str, is_main_user: bool = False,
                           is_campaign: bool = False) -> Agent:
        """Route call to appropriate agent"""
        if is_main_user:
            agent = self.agents.get("personal_assistant")
            if agent and agent.enabled:
                return agent

        if direction in ("outbound", "outgoing") and is_campaign:
            agent = self.agents.get("sales_rep")
            if agent and agent.enabled:
                return agent

        if direction == "incoming":
            agent = self.agents.get("receptionist")
            if agent and agent.enabled:
                return agent

        # Fallback
        return self.agents.get("receptionist", DEFAULT_AGENTS["receptionist"])

    def list_agents(self) -> List[dict]:
        """List all agents as dicts"""
        return [agent.to_dict() for agent in self.agents.values()]

    def update_agent(self, agent_id: str, updates: dict) -> Optional[Agent]:
        """Update agent configuration"""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]

        if "name" in updates:
            agent.name = updates["name"]
        if "model_tier" in updates:
            agent.model_tier = ModelTier(updates["model_tier"])
        if "objective" in updates:
            agent.objective = updates["objective"]
        if "persona" in updates:
            agent.persona = updates["persona"]
        if "enabled" in updates:
            agent.enabled = updates["enabled"]
        if "tools" in updates:
            agent.tools = updates["tools"]
        if "icon" in updates:
            agent.icon = updates["icon"]
        if "knowledge_base" in updates:
            agent.knowledge_base = updates["knowledge_base"]

        return agent

    def get_system_prompt(self, agent: Agent, context: dict) -> str:
        """Generate system prompt for agent with context substitution"""
        prompt = agent.persona

        # Substitute context variables
        prompt = prompt.replace("{company_name}", context.get("company_name", "the company"))
        prompt = prompt.replace("{my_name}", context.get("my_name", "your boss"))
        prompt = prompt.replace("{location}", context.get("location", ""))

        return prompt


def get_agent_manager(settings: dict = None) -> AgentManager:
    """
    Get agent manager with fresh settings.

    Always creates a new instance to ensure settings are read fresh.
    If settings not provided, reads from settings.json.
    """
    if settings is None:
        import api_keys
        settings = api_keys.get_settings()
    return AgentManager(settings)
