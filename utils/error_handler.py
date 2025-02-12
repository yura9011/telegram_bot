import logging
import uuid
import traceback
from functools import wraps
from telegram import Update
from telegram.ext import CallbackContext

logger = logging.getLogger(__name__)

def error_handler(func):
    """Decorator for handling errors in bot handlers."""
    @wraps(func)
    async def wrapper(update: Update, context: CallbackContext):
        try:
            return await func(update, context)
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_message = f"Error in handler {func.__name__}: {e}"
            logger.exception(error_message)
            
            config = context.bot_data.get("config", {})
            user_message_template = config.get('bot_messages', {}).get(
                'generic_error_message', 
                "An error occurred. Please try again later. (Error ID: {error_id})"
            )
            user_message = user_message_template.format(error_id=error_id)

            if update and update.effective_chat:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=user_message
                )

            # Send error to developer if configured
            gemini = context.bot_data.get("gemini")
            if gemini:
                await gemini._send_developer_error(
                    error_message,
                    traceback.format_exc(),
                    update.effective_chat.id if update and update.effective_chat else None
                )

    return wrapper