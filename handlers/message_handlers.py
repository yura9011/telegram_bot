import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import MessageHandler, filters, CallbackContext, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from PIL import Image, ImageFile
import io
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
from handlers.command_handlers import calendar_create

logger = logging.getLogger(__name__)


@error_handler
async def text_message_handler(update: Update, context: CallbackContext):
    """Handles text messages, including calendar event creation via natural language."""
    chat_id = update.effective_chat.id
    user_message = update.message.text

    gemini: GeminiModel = context.bot_data["gemini"]
    memory: PersistentHybridMemory = context.bot_data["memory"]
    config: Dict[str, Any] = context.bot_data["config"]

    # --- Intent Detection (Calendar Creation) ---
    logger.debug("text_message_handler: Starting intent detection.")
    calendar_creation_keywords = [
        "crear evento",
        "agendar",
        "programar cita",
        "añadir al calendario",
        "crear en calendario",
    ]  # Keywords to detect calendar event creation intent
    is_calendar_intent = any(
        user_message.lower().startswith(keyword) for keyword in calendar_creation_keywords
    )
    logger.debug(f"text_message_handler: is_calendar_intent = {is_calendar_intent}")

    if is_calendar_intent:
        logger.info("Calendar event creation intent detected.")
        # --- Calendar Event Creation Flow ---
        calendar_prompt_config = config.get("bot_messages", {}).get("calendar_creation_prompt")
        if not calendar_prompt_config:
            calendar_creation_prompt = """Extract ONLY the event information from the user's text and respond EXCLUSIVELY in valid JSON format. Do not include explanatory text, introductions, goodbyes, or Markdown code blocks. The JSON must have the following keys: 'title', 'date' (format YYYY-MM-DD, if not specified, use 'not specified'), 'time' (format HH:MM, if not specified, use 'not specified'), 'description' (if not specified, use 'No description'). If you cannot extract all the necessary information, or if the text is not about creating an event, respond with JSON: {'title': 'not specified', 'date': 'not specified', 'time': 'not specified', 'description': 'not specified'}. Example of a valid JSON response: {'title': 'Meeting with John', 'date': '2025-03-10', 'time': '15:30', 'description': 'Review proposal'}"""

        prompt = f"{gemini.system_instruction}\n{calendar_prompt_config}\nUser message: {user_message}"  # Build prompt for Gemini

        response_text = ""
        async for chunk in gemini.generate_response_non_streaming(prompt, chat_id=chat_id):
            response_text += chunk

        try:
            import json

            logger.debug(f"Gemini JSON response (raw): {response_text}")

            # Remove Markdown code block if present to ensure valid JSON parsing
            json_string = response_text.replace("```json", "").replace("```", "").strip()
            logger.debug(f"Gemini JSON response (after markdown removal): {json_string}")

            event_details = json.loads(json_string)  # Parse Gemini's JSON response
            logger.debug(f"Gemini extracted event details: {event_details}")

            # Call calendar_create handler programmatically, passing extracted event details as arguments
            context.args = [
                event_details.get("title", "Evento sin título"),
                event_details.get("date", None),
                event_details.get("time", None),
                event_details.get("description", "No description extracted from user message."),
            ]
            logger.debug(f"text_message_handler: Calling calendar_create with args: {context.args}")
            await calendar_create(update, context)
            return  # Exit handler after calendar creation flow

        except json.JSONDecodeError as e:
            logger.error(
                f"Could not parse Gemini's JSON response for calendar creation: {response_text}"
            )
            logger.error(f"JSONDecodeError details: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="Lo siento, no pude entender los detalles del evento. ¿Podrías reformular tu solicitud?", # Sorry, I couldn't understand the event details. Could you rephrase your request?
            )
            return

        except Exception as e:
            logger.exception(f"Error processing calendar creation intent: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="Lo siento, hubo un error al intentar crear el evento. Por favor, inténtalo de nuevo más tarde.", # Sorry, there was an error trying to create the event. Please try again later.
            )
            return

    # --- Default Assistant Response (if not calendar intent) ---
    logger.info("Default assistant handler for general query.")
    context_memory = memory.get_context(chat_id, user_message)
    prompt = f"{gemini.system_instruction}\n{context_memory}\nuser: {user_message}" # Build prompt with memory context

    # Add user message to memory
    memory.add_message(chat_id, "user", user_message)

    response_text = ""
    async for chunk in gemini.generate_response_non_streaming(prompt, chat_id=chat_id):
        response_text += chunk
    if not response_text:
        response_text = "The model provided an empty response."
    # Store bot response to memory
    memory.add_message(chat_id, "assistant", response_text)

    # Keyboard for Text or Voice choice
    context.user_data["gemini_response"] = response_text
    keyboard = [[InlineKeyboardButton("Text", callback_data="text"),
                 InlineKeyboardButton("Voice", callback_data="voice")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(
        chat_id=chat_id, text="Response options:", reply_markup=reply_markup # Response options:
    )


@error_handler
async def image_message_handler(update: Update, context: CallbackContext):
    """Handles image messages."""
    chat_id = update.effective_chat.id
    gemini = context.bot_data["gemini"]

    try:
        # Get the image file from Telegram
        if update.message.photo:
            image_file = await context.bot.get_file(update.message.photo[-1].file_id)
        else:  # document
            image_file = await context.bot.get_file(update.message.document.file_id)

        # Download image as byte array and open with PIL
        image_bytes = await image_file.download_as_bytearray()
        image = Image.open(io.BytesIO(image_bytes))

        # Use message caption as prompt, or default if no caption
        prompt = update.message.caption if update.message.caption else "Analyze this image"

        # Generate response from Gemini model
        response_text = ""
        async for chunk in gemini.generate_response_non_streaming(image, chat_id=chat_id):
            response_text += chunk

        await context.bot.send_message(chat_id=chat_id, text=response_text)

    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Sorry, there was an error processing the image." # Sorry, there was an error processing the image.
        )


def register_message_handlers(
    application: Application, config: Dict[str, Any], gemini: GeminiModel, memory: PersistentHybridMemory
) -> None:
    """Registers message handlers with the application."""
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            lambda update, context: text_message_handler(update, context),
        )
    )
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_message_handler)
    )