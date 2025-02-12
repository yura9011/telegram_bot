import asyncio
import sys
from config.config_loader import load_config
from config.logging_config import setup_logging
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
from telegram.ext import Application
from telegram import Update
from handlers.command_handlers import register_command_handlers
from handlers.message_handlers import register_message_handlers
from handlers.voice_handlers import register_voice_handlers

def create_app():
    """
    Initializes and returns the Telegram application, GeminiModel, configuration, and persistent memory.
    """
    config = load_config()
    setup_logging(config)

    TOKEN = config["bot_token"]
    application = Application.builder().token(TOKEN).build()

    # Initialize models
    gemini_instance = GeminiModel(config)
    gemini_instance.application = application
    persistent_memory_instance = PersistentHybridMemory(config)

    # Register all handlers
    register_command_handlers(application, config, gemini_instance, persistent_memory_instance)
    register_message_handlers(application, config, gemini_instance, persistent_memory_instance)
    register_voice_handlers(application, config, gemini_instance, persistent_memory_instance)

    return application, gemini_instance, config, persistent_memory_instance

def main():
    """Runs the bot application."""
    application, _, _, _ = create_app()
    print("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()