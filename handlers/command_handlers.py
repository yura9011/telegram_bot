import logging
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
from handlers.calendar_handlers import calendar_create

logger = logging.getLogger(__name__)

@error_handler
async def calendar_create_handler(update: Update, context: CallbackContext) -> None:
    """
    Handles the /calendar_create command.
    """
    print("calendar_create_handler: Handler is running!")  # Add print statement
    await calendar_create(update, context)

@error_handler
async def start(update: Update, context: CallbackContext):
    """Handles the /start command."""
    logger.info("Command /start received")
    message = context.bot_data["config"].get('bot_messages', {}).get('welcome_message')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

@error_handler
async def help_command(update: Update, context: CallbackContext):
    """Handles the /help command."""
    logger.info("Command /help received")
    message = context.bot_data["config"].get('bot_messages', {}).get('help_message')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

@error_handler
async def search_command(update: Update, context: CallbackContext):
    """Handles the /search command."""
    query = ' '.join(context.args)
    if not query:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="Please provide a search query after the /search command."
        )
        return

    from utils.search_utils import search_serpapi  # Import here to avoid circular imports
    
    serpapi_conf = context.bot_data["config"].get('serpapi', {})
    search_results = search_serpapi(query, serpapi_conf)

    if search_results:
        message_parts = []
        for result in search_results:
            title = result["title"]
            snippet = result["snippet"]
            
            from utils.markdown_utils import escape_markdown_v2
            escaped_title = escape_markdown_v2(title)
            escaped_snippet = escape_markdown_v2(snippet)
            
            message_parts.append(f"*{escaped_title}*\n{escaped_snippet}")
            
        message_text = "\n\n".join(message_parts) + "\n\n"
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=message_text, 
            parse_mode="MarkdownV2"
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="No relevant search results found or search failed."
        )

def register_command_handlers(
    application: Application,
    config: Dict[str, Any],
    gemini: GeminiModel,
    memory: PersistentHybridMemory
) -> None:
    """Registers all command handlers with the application."""
    # Store config in bot_data for access in handlers
    application.bot_data["config"] = config
    application.bot_data["gemini"] = gemini
    application.bot_data["memory"] = memory

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("calendar_create", calendar_create_handler)) 