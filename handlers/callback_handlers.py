import logging
import io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import CallbackContext, CallbackQueryHandler, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory

logger = logging.getLogger(__name__)

@error_handler
async def button_callback(update: Update, context: CallbackContext):
    """Processes the callback query when a button is pressed (Text or Voice)."""
    query: CallbackQuery = update.callback_query
    await query.answer()  # Acknowledge the callback

    choice = query.data
    chat_id = update.effective_chat.id
    response_text = context.user_data.get("gemini_response")

    if not response_text:
        logger.warning("Gemini response not found in user_data for callback.")
        await query.edit_message_text(text="Error: No response available.")
        return

    if choice == 'text':
        response_message = "Response sent via text."  # Message to show after sending text
        await context.bot.send_message(chat_id=chat_id, text=response_text)
    elif choice == 'voice':
        response_message = "Response sent via voice."
        try:
            import elevenlabs
            from elevenlabs.client import ElevenLabs  # Import ElevenLabs Client

            client = ElevenLabs()  # Initialize ElevenLabs client

            voice_id = context.bot_data["config"]["elevenlabs"]["voice_id"]
            model_id = context.bot_data["config"]["elevenlabs"]["model_id"]
            output_format = "mp3_44100_128"

            audio_generator = client.text_to_speech.convert(  # Get the generator
                text=response_text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format
            )

            audio_bytes = b""  # Initialize empty bytes object
            for chunk in audio_generator:  # Iterate through the generator
                audio_bytes += chunk  # Append each chunk to audio_bytes

            audio_io = io.BytesIO(audio_bytes)  # Create BytesIO from accumulated bytes
            audio_io.name = "response.mp3"
            await context.bot.send_voice(chat_id=chat_id, voice=audio_io)


        except Exception as e:
            logger.exception(f"ElevenLabs error: {e}")
            await context.bot.send_message(chat_id=chat_id, text="Sorry, there was an error generating audio response.")
            response_message = "Error: Audio generation failed."
    else:
        response_message = f"Error: Unknown choice: {choice}"

    # Edit the button message to show "Response sent..." and remove buttons in ONE edit_message_text call
    try:
        await query.edit_message_text(
            text=response_message,
            reply_markup=None  # REMOVE reply_markup=None from here
        )
    except Exception as e:
        logger.warning(f"Error editing message after button press (likely 'message not modified' error): {e}")


def register_callback_handlers(
    application: Application,
    config: Dict[str, Any],
    gemini: GeminiModel,
    memory: PersistentHybridMemory
) -> None:
    """Registers callback query handlers with the application."""
    # Store config and models in bot_data for access in handlers if not already done
    if "config" not in application.bot_data:
        application.bot_data["config"] = config
    if "gemini" not in application.bot_data:
        application.bot_data["gemini"] = gemini
    if "memory" not in application.bot_data:
        application.bot_data["memory"] = memory

    application.add_handler(CallbackQueryHandler(button_callback))