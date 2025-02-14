import os
import yaml
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Loads configuration from config.yaml and environment variables."""
    config = {}
    
    # Load from config.yaml
    try:
        with open("config.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    except FileNotFoundError:
        logger.warning("config.yaml not found, using environment variables only.")

    # Load environment variables
    load_dotenv()

    # Override with environment variables if set
    config['bot_token'] = os.environ.get('BOT_TOKEN', config.get('bot_token'))
    config['gemini_api_key'] = os.environ.get('GEMINI_API_KEY', config.get('gemini_api_key'))
    config['serpapi_api_key'] = os.environ.get('SERPAPI_API_KEY', config.get('serpapi', {}).get('api_key'))
    config['assemblyai_api_key'] = os.environ.get('ASSEMBLYAI_API_KEY')
    config['elevenlabs_api_key'] = os.environ.get('ELEVENLABS_API_KEY')

    # Validate required configuration
    required_keys = ['bot_token', 'gemini_api_key', 'elevenlabs_api_key']
    for key in required_keys:
        if not config.get(key):
            logger.critical(f"{key} is missing in config.yaml and environment variables.")
            raise ValueError(f"Missing required configuration: {key}")

    return config