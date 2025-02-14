import os.path
import logging, datetime
from typing import List, Dict, Any, Optional
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory
from telegram import Update
from telegram.ext import CallbackContext, CommandHandler, Application
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils.date_utils import parse_datetime
from utils.error_handler import error_handler

logger = logging.getLogger(__name__)

def get_calendar_service(config: Dict[str, Any]) -> Optional[Any]:
    """
    Autentica y autoriza al usuario para acceder a la API de Google Calendar.
    Devuelve un objeto de servicio de Google Calendar API.
    """
    creds = None
    credentials_file = config["google_calendar"]["credentials_file"]
    token_file = config["google_calendar"]["token_file"]
    scopes = config["google_calendar"]["scopes"]

    # El archivo token.json guarda los tokens de acceso y actualización del usuario,
    # y se crea automáticamente cuando el flujo de autorización se completa
    # por primera vez.
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    # Si no hay credenciales válidas disponibles, el usuario debe loguearse.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                return None
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, scopes)
            creds = flow.run_local_server(port=0) # Abre el navegador para la autorización
        # Guardar las credenciales para la próxima ejecución
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('calendar', 'v3', credentials=creds) # Construye el servicio de Calendar API
        return service
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return None

async def list_next_events(service: Any, num_events: int = 5) -> Optional[List[Dict[str, str]]]:
    """
    Lista los próximos eventos del calendario principal.
    Args:
        service: Objeto de servicio de Google Calendar API.
        num_events: Número máximo de eventos a listar.
    Returns:
        Lista de diccionarios con información de los eventos, o None si hay error.
    """
    try:
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indica UTC
        logger.info(f"Getting the upcoming {num_events} events")
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                            maxResults=num_events, singleEvents=True,
                                            orderBy='startTime').execute()
        events = events_result.get('items', [])

        if not events:
            logger.info('No upcoming events found.')
            return []

        event_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_info = {
                "summary": event.get('summary', 'No title'),
                "start": start,
                "description": event.get('description', 'No description')
            }
            event_list.append(event_info)
        return event_list

    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return None

async def calendar_auth(update: Update, context: CallbackContext) -> None:
    """Handle the /calendar_auth command."""
    service = get_calendar_service(context.bot_data["config"])
    if service:
        await update.message.reply_text("Calendar authentication successful!")
    else:
        await update.message.reply_text("Calendar authentication failed. Please try again.")

async def calendar_list(update: Update, context: CallbackContext) -> None:
    """Handle the /calendar_list command."""
    service = get_calendar_service(context.bot_data["config"])
    if not service:
        await update.message.reply_text("Calendar service not available. Please authenticate first with /calendar_auth")
        return
    
    events = await list_next_events(service)
    if events is None:
        await update.message.reply_text("Error fetching calendar events.")
        return
    
    if not events:
        await update.message.reply_text("No upcoming events found.")
        return
    
    response = "Upcoming events:\n"
    for event in events:
        response += f"\n- {event['summary']} ({event['start']})"
    await update.message.reply_text(response)

@error_handler
async def calendar_create(update: Update, context: CallbackContext):
    """Handles the /calendar_create command to create a new calendar event."""
    print("calendar_create: Handler is running!") # <-- ¡AÑADE ESTE PRINT COMO PRIMERA LÍNEA!
    service = context.bot_data.get("calendar_service")
    if not service:
        config = context.bot_data["config"]
        service = get_calendar_service(config)
        if not service:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="No autenticado con Google Calendar. Usa /calendar_auth primero."
            )
            return

    # Get arguments from command
    args = context.args
    if not args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Uso: /calendar_create <titulo> [fecha] [hora] [descripcion].\nFecha y hora opcionales, formato ISO (YYYY-MM-DD HH:MM)."
        )
        return

    print(f"calendar_create: Args recibidos: {context.args}") # DEBUG PRINT

    title = args[0]
    date_str = args[1] if len(args) > 1 else None
    time_str = args[2] if len(args) > 2 else None
    description = ' '.join(args[3:]) if len(args) > 3 else "No description provided."

    # Parse date and time using the utility function
    start_datetime_str = await parse_datetime(date_str, time_str) # Assuming parse_datetime is async

    print(f"calendar_create: start_datetime_str después de parse_datetime: {start_datetime_str}") # DEBUG PRINT

    if not start_datetime_str:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Formato de fecha/hora inválido. Usa formato ISO YYYY-MM-DD [HH:MM]."
        )
        return

    print(f"calendar_create: start_datetime_str final: {start_datetime_str}") # DEBUG PRINT

    event = {
        'summary': title,
        'description': description,
        'start': {
            'dateTime': start_datetime_str,
            'timeZone': 'America/Argentina/Buenos_Aires', # Or user's timezone if you can get it
        },
        'end': { # For simplicity, setting end time 1 hour after start
            'dateTime': datetime.datetime.fromisoformat(start_datetime_str).replace(hour=datetime.datetime.fromisoformat(start_datetime_str).hour + 1).isoformat(),
            'timeZone': 'America/Argentina/Buenos_Aires',
        },
        'reminders': { # Optional reminders
            'useDefault': False,
            'overrides': [
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }

    print(f"calendar_create: Evento a insertar en Calendar API: {event}") # DEBUG PRINT

    try:
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        logger.info(f"Event created: {created_event['htmlLink']}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Evento creado exitosamente: {created_event['summary']}\nLink: {created_event['htmlLink']}"
        )
        print(f"calendar_create: Evento creado exitosamente. Link: {created_event['htmlLink']}") # DEBUG PRINT

    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error al crear el evento. Detalles: {error}"
        )
        print(f"calendar_create: Error al crear evento: {error}") # DEBUG PRINT

def register_calendar_handlers(
    application: Application,
    config: Dict[str, Any],
    gemini: GeminiModel,
    memory: PersistentHybridMemory
) -> None:
    """Registers calendar command handlers with the application."""
    # Store config in bot_data for access in handlers
    application.bot_data["config"] = config
    application.bot_data["gemini"] = gemini
    application.bot_data["memory"] = memory

    # Register command handlers
    application.add_handler(CommandHandler("calendar_auth", calendar_auth))
    application.add_handler(CommandHandler("calendar_list", calendar_list))