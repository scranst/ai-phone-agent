"""
Centralized API Key Management

Single source of truth for all API keys. Always reads fresh from settings.
No caching, no hot-reload needed - just call get_key() when you need it.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


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

    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)

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

    except Exception as e:
        logger.warning(f"Failed to read API key for {service}: {e}")
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
