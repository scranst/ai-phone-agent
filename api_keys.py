"""
Centralized Settings Management

Single source of truth for ALL settings. Reads from SQLite database.
No caching, no hot-reload needed - just call the function when you need it.

Includes:
- API keys
- User info (name, address, phone, etc.)
- Agent configurations (personas, objectives, tools)
- Knowledge bases
- Integrations (calendar, etc.)
"""

import json
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Core Settings Access
# =============================================================================

def get_settings() -> dict:
    """Get all settings from database"""
    try:
        import database
        return database.get_all_settings()
    except Exception as e:
        logger.warning(f"Failed to read settings from database: {e}")
        return {'api_keys': {}, 'agents': {}, 'integrations': {}}


def save_settings(settings: dict):
    """Save settings to database"""
    try:
        import database
        database.set_settings_bulk(settings)
    except Exception as e:
        logger.error(f"Failed to save settings to database: {e}")
        raise


def get_setting(key: str, default: Any = None) -> Any:
    """Get a top-level setting by key"""
    return get_settings().get(key, default)


# =============================================================================
# API Keys
# =============================================================================

def get_key(service: str) -> str:
    """
    Get API key for a service. Always reads fresh from settings.

    Args:
        service: One of 'anthropic', 'openai', 'google', 'google_cse', etc.

    Returns:
        API key string (empty string if not set)
    """
    key_mapping = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'google_cse': 'GOOGLE_CSE_ID',
        'apify': 'APIFY_API_KEY',
        'amadeus': 'AMADEUS_API_KEY',
        'amadeus_secret': 'AMADEUS_API_SECRET',
        'neverbounce': 'NEVERBOUNCE_API_KEY',
        'phonevalidator': 'PHONEVALIDATOR_API_KEY',
        'calcom': 'CAL_COM_API_KEY',
    }

    settings_key = key_mapping.get(service, service.upper() + '_API_KEY')
    settings = get_settings()

    # Check api_keys section first
    api_keys = settings.get('api_keys', {})
    if settings_key in api_keys and api_keys[settings_key]:
        return api_keys[settings_key]

    # Check integrations section (for calcom, etc.)
    integrations = settings.get('integrations', {})
    if settings_key in integrations and integrations[settings_key]:
        return integrations[settings_key]

    # Fall back to environment variable
    return os.getenv(settings_key, '')


def get_anthropic_client():
    """Get a fresh Anthropic client with current API key"""
    import anthropic
    key = get_key('anthropic')
    if not key:
        raise ValueError("Anthropic API key not configured. Add it in Settings > API Keys.")
    return anthropic.Anthropic(api_key=key)


def get_openai_client():
    """Get a fresh OpenAI client with current API key"""
    import openai
    key = get_key('openai')
    if not key:
        raise ValueError("OpenAI API key not configured. Add it in Settings > API Keys.")
    return openai.OpenAI(api_key=key)


# =============================================================================
# User Info
# =============================================================================

def get_user_info() -> dict:
    """Get user's personal info (name, address, phone, etc.)"""
    settings = get_settings()
    return {
        'name': settings.get('MY_NAME', ''),
        'phone': settings.get('CALLBACK_NUMBER', ''),
        'email': settings.get('EMAIL', ''),
        'company': settings.get('COMPANY', ''),
        'address': settings.get('ADDRESS', ''),
        'city': settings.get('CITY', ''),
        'state': settings.get('STATE', ''),
        'zip': settings.get('ZIP', ''),
        'card_number': settings.get('CARD_NUMBER', ''),
        'card_exp': settings.get('CARD_EXP', ''),
        'card_cvv': settings.get('CARD_CVV', ''),
        'billing_zip': settings.get('BILLING_ZIP', ''),
        'vehicle_year': settings.get('VEHICLE_YEAR', ''),
        'vehicle_make': settings.get('VEHICLE_MAKE', ''),
        'vehicle_model': settings.get('VEHICLE_MODEL', ''),
        'vehicle_color': settings.get('VEHICLE_COLOR', ''),
    }


def get_user_context() -> dict:
    """Get user info formatted for LLM context"""
    info = get_user_info()
    # Only include non-empty values
    return {k: v for k, v in info.items() if v}


# =============================================================================
# Agent Configuration
# =============================================================================

def get_agent_config(agent_id: str) -> Optional[dict]:
    """Get configuration for a specific agent"""
    settings = get_settings()
    agents = settings.get('agents', {})
    return agents.get(agent_id)


def get_all_agents() -> dict:
    """Get all agent configurations"""
    return get_settings().get('agents', {})


def get_agent_persona(agent_id: str) -> str:
    """Get an agent's persona/system prompt"""
    config = get_agent_config(agent_id)
    if config:
        return config.get('persona', '')
    return ''


def get_agent_objective(agent_id: str) -> str:
    """Get an agent's objective"""
    config = get_agent_config(agent_id)
    if config:
        return config.get('objective', '')
    return ''


def get_agent_tools(agent_id: str) -> list:
    """Get an agent's enabled tools"""
    config = get_agent_config(agent_id)
    if config:
        return config.get('tools', [])
    return []


def get_agent_model_tier(agent_id: str) -> str:
    """Get an agent's model tier (haiku, sonnet, opus)"""
    config = get_agent_config(agent_id)
    if config:
        return config.get('model_tier', 'haiku')
    return 'haiku'


# =============================================================================
# Knowledge Bases
# =============================================================================

def get_knowledge_base(agent_id: str) -> str:
    """Load knowledge base content for an agent"""
    config = get_agent_config(agent_id)
    if not config:
        return ''

    kb_path = config.get('knowledge_base', '')
    if not kb_path:
        return ''

    # Resolve path relative to project root
    full_path = Path(__file__).parent / kb_path
    try:
        if full_path.exists():
            return full_path.read_text()
    except Exception as e:
        logger.warning(f"Failed to load knowledge base {kb_path}: {e}")

    return ''


# =============================================================================
# Integrations
# =============================================================================

def get_integration(name: str) -> dict:
    """Get integration settings (calendar, email, etc.)"""
    settings = get_settings()
    integrations = settings.get('integrations', {})

    if name == 'calendar':
        return {
            'provider': integrations.get('CALENDAR_PROVIDER', 'cal.com'),
            'calcom_api_key': integrations.get('CAL_COM_API_KEY', ''),
            'calcom_event_type_id': integrations.get('CAL_COM_EVENT_TYPE_ID', ''),
            'calendly_api_key': integrations.get('CALENDLY_API_KEY', ''),
            'calendly_user_uri': integrations.get('CALENDLY_USER_URI', ''),
        }

    return integrations


def get_sms_settings() -> dict:
    """Get SMS/autopilot settings"""
    settings = get_settings()
    return {
        'primary_phone': settings.get('sms', {}).get('PRIMARY_PHONE', ''),
        'autopilot_enabled': settings.get('autopilot', {}).get('ENABLED', False),
        'reply_delay_min': settings.get('autopilot', {}).get('REPLY_DELAY_MIN', 30),
        'reply_delay_max': settings.get('autopilot', {}).get('REPLY_DELAY_MAX', 120),
    }


def get_incoming_settings() -> dict:
    """Get incoming call settings"""
    settings = get_settings()
    incoming = settings.get('incoming', {})
    return {
        'enabled': incoming.get('ENABLED', False),
        'persona': incoming.get('PERSONA', ''),
        'greeting': incoming.get('GREETING', ''),
    }
