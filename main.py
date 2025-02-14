import asyncio
import sys
from config.config_loader import load_config
from config.logging_config import setup_logging
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
from telegram.ext import Application
from telegram import Update
from telegram.ext import CallbackContext, CallbackQueryHandler, Application, ConversationHandler
from handlers.command_handlers import register_command_handlers
from handlers.message_handlers import register_message_handlers
from handlers.callback_handlers import button_callback
from handlers.voice_handlers import register_voice_handlers
from handlers.calendar_handlers import register_calendar_handlers

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
    persistent_memory_instance = None  # Initialize persistent_memory_instance here
    gemini_instance.application = application
    persistent_memory_instance = PersistentHybridMemory(config)
    gemini_instance.application = application
    persistent_memory_instance = PersistentHybridMemory(config)

    # Register all handlers
    register_command_handlers(application, config, gemini_instance, persistent_memory_instance)
    register_message_handlers(application, config, gemini_instance, persistent_memory_instance)
    register_voice_handlers(application, config, gemini_instance, persistent_memory_instance)
    register_calendar_handlers(application, config, gemini_instance, persistent_memory_instance) 
    application.add_handler(CallbackQueryHandler(button_callback))
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