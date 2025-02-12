import logging
from telegram import Update
from telegram.ext import MessageHandler, filters, CallbackContext, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from PIL import Image
import io
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory

logger = logging.getLogger(__name__)

@error_handler
async def text_message_handler(update: Update, context: CallbackContext):
    """Handles text messages."""
    chat_id = update.effective_chat.id
    user_message = update.message.text
    
    gemini = context.bot_data["gemini"]
    memory = context.bot_data["memory"]
    
    # Get context from memory
    context_memory = memory.get_context(chat_id, user_message)
    prompt = f"{gemini.system_instruction}\n{context_memory}\nuser: {user_message}"

    # Add user message to memory
    memory.add_message(chat_id, "user", user_message)

    response_text = ""
    async for chunk in gemini.generate_response_non_streaming(prompt, chat_id=chat_id):
        response_text += chunk

    # Add bot response to memory
    memory.add_message(chat_id, "assistant", response_text)
    await context.bot.send_message(chat_id=chat_id, text=response_text)

@error_handler
async def image_message_handler(update: Update, context: CallbackContext):
    """Handles image messages."""
    chat_id = update.effective_chat.id
    gemini = context.bot_data["gemini"]

    try:
        # Get the image file
        if update.message.photo:
            image_file = await context.bot.get_file(update.message.photo[-1].file_id)
        else:  # document
            image_file = await context.bot.get_file(update.message.document.file_id)

        # Download and process image
        image_bytes = await image_file.download_as_bytearray()
        image = Image.open(io.BytesIO(image_bytes))

        # Get caption or default prompt
        prompt = update.message.caption if update.message.caption else "Analyze this image"

        # Generate response
        response_text = ""
        async for chunk in gemini.generate_response_non_streaming(image, chat_id=chat_id):
            response_text += chunk

        await context.bot.send_message(chat_id=chat_id, text=response_text)

    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Sorry, there was an error processing the image."
        )

def register_message_handlers(
    application: Application,
    config: Dict[str, Any],
    gemini: GeminiModel,
    memory: PersistentHybridMemory
) -> None:
    """Registers all message handlers with the application."""
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        text_message_handler
    ))
    application.add_handler(MessageHandler(
        filters.PHOTO | filters.Document.IMAGE,
        image_message_handler
    ))