import logging
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from utils.date_utils import parse_datetime
from utils.google_calendar_utils import get_calendar_service, list_next_events
from utils.markdown_utils import escape_markdown_v2
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
import datetime
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


@error_handler
async def calendar_auth(update: Update, context: CallbackContext):
    """Handles the /calendar_auth command to authenticate with Google Calendar."""
    config = context.bot_data["config"]
    service = get_calendar_service(config)
    if service:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Calendar authentication successful. You can now use /calendar_list to see your upcoming events.",
        )
        # Store service in bot_data for later use
        context.bot_data["calendar_service"] = service
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Calendar authentication failed. Check the logs for more details.",
        )


@error_handler
async def calendar_create(update: Update, context: CallbackContext):
    """Handles the /calendar_create command to create a new calendar event."""
    print("calendar_create: Handler is running!") # Keep print for immediate confirmation
    service = context.bot_data.get("calendar_service")
    if not service:
        config = context.bot_data["config"]
        service = get_calendar_service(config)
        if not service:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Not authenticated with Google Calendar. Use /calendar_auth first.",
            )
            logger.debug("calendar_create: Calendar service not found in bot_data or config, authentication required.")
            return

    # Get arguments from command
    args = context.args
    if not args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Usage: /calendar_create <title> [date] [time] [description].\nDate and time are optional, ISO format (YYYY-MM-DD HH:MM).",
        )
        return
    logger.debug(f"calendar_create: Args received from message handler: {context.args}")
    print(f"calendar_create: Args recibidos: {context.args}")  # Keep print for debugging

    title = args[0]
    date_str = args[1] if len(args) > 1 else None
    time_str = args[2] if len(args) > 2 else None
    description = " ".join(args[3:]) if len(args) > 3 else "No description provided."

    # Parse date and time using the utility function
    start_datetime_str = await parse_datetime(date_str, time_str)
    logger.debug(f"calendar_create: start_datetime_str after parse_datetime: {start_datetime_str}")
    print(f"calendar_create: start_datetime_str después de parse_datetime: {start_datetime_str}")  # Keep print for debugging

    if not start_datetime_str:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Invalid date/time format. Use ISO format YYYY-MM-DD [HH:MM].",
        )
        return

    print(f"calendar_create: start_datetime_str final: {start_datetime_str}")  # Keep print for debugging

    event = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_datetime_str,
            "timeZone": "America/Argentina/Buenos_Aires",  # User's timezone
        },
        "end": {  # Set end time 1 hour after start
            "dateTime": datetime.datetime.fromisoformat(start_datetime_str).replace(
                hour=datetime.datetime.fromisoformat(start_datetime_str).hour + 1
            ).isoformat(),
            "timeZone": "America/Argentina/Buenos_Aires",
        },
        "reminders": {  # Optional reminders
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 10}],
        },
    }
    logger.debug(f"calendar_create: Event object for Calendar API: {event}")
    print(f"calendar_create: Evento a insertar en Calendar API: {event}") # Keep print for debugging

    try:
        if not service: # Additional check
            logger.error("calendar_create: Calendar API service object is None. Cannot call insert().")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Internal error: Calendar service not available.")
            return
        logger.debug("calendar_create: Calling Calendar API events().insert()")
        created_event = service.events().insert(calendarId='primary', body=event).execute()

        print("calendar_create: Calendar API insert() call executed. Response:") # Keep print for debugging
        print(created_event)  # Print full API response for debugging

        logger.info(f"Event created: {created_event['htmlLink']}")
        logger.debug("calendar_create: Event creation successful, sending success message to user.")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Event created successfully: {created_event['summary']}\nLink: {created_event['htmlLink']}"
        )
        print(f"calendar_create: Evento creado exitosamente. Link: {created_event['htmlLink']}")  # Keep print for debugging

    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        logger.debug(f"calendar_create: HttpError during event creation: {error}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error creating event. Details: {error}",
        )
        print(f"calendar_create: Error al crear evento: {error}")  # Keep print for debugging


@error_handler
async def calendar_list(update: Update, context: CallbackContext):
    """Handles the /calendar_list command to list upcoming calendar events."""
    service = context.bot_data.get("calendar_service")  # Try to get service from bot_data
    if not service:
        config = context.bot_data["config"]  # Re-authenticate if not in bot_data
        service = get_calendar_service(config)
        if not service:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Not authenticated with Google Calendar. Use /calendar_auth first.",
            )
            return

    events = await list_next_events(service)
    if events is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Error fetching calendar events. Check the logs.",
        )
        return

    if not events:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="No upcoming events found in your calendar.",
        )
        return

    message_text = "*Upcoming Calendar Events:*\n\n"
    for event in events:
        escaped_summary = escape_markdown_v2(event["summary"])
        escaped_start = escape_markdown_v2(event["start"])
        escaped_description = escape_markdown_v2(event["description"]) if event["description"] else "_(No description)_"
        message_text += f"*{escaped_summary}*\n"
        message_text += f"  *Start:* {escaped_start}\n"
        message_text += f"  *Description:* {escaped_description}\n\n"

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=message_text, parse_mode="MarkdownV2"
    )


def register_calendar_handlers(
    application: Application, config: Dict[str, Any], gemini: GeminiModel, memory: PersistentHybridMemory
) -> None:
    """Registers calendar command handlers with the application."""
    # Store config in bot_data for access in handlers
    application.bot_data["config"] = config
    application.bot_data["gemini"] = gemini
    application.bot_data["memory"] = memory

    # Register command handlers
    application.add_handler(CommandHandler("calendar_auth", calendar_auth))
    application.add_handler(CommandHandler("crear_calendario", calendar_create)) # Consider renaming command to "calendar_create" for consistency
    application.add_handler(CommandHandler("calendar_list", calendar_list))