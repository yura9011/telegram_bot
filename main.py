import logging
import asyncio
import sys
import traceback
import uuid
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import yaml
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    CommandHandler,
    CallbackContext
)
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union
import base64 #Import base 64
from serpapi import GoogleSearch

# --- Configuration ---
import os
def load_config() -> Dict[str, Any]:
    config = {}
    # Load from config.yaml
    try:
        with open("config.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    except FileNotFoundError as e:
        logger.warning(f"config.yaml not found, using environment variables only.")

    # Override with environment variables if set
    config['bot_token'] = os.environ.get('BOT_TOKEN', config.get('bot_token')) # Fallback to config.yaml if env var not set
    config['gemini_api_key'] = os.environ.get('GEMINI_API_KEY', config.get('gemini_api_key'))
    config['serpapi_api_key'] = os.environ.get('SERPAPI_API_KEY', config.get('serpapi', {}).get('api_key')) # Nested access with get

    return config

# Disable sentence_transformers logger
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Create logger for this module
logger = logging.getLogger(__name__)

# --- Logging ---
def setup_logging(config: Dict[str, Any]):
    """Sets up logging according to the configuration."""
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "DEBUG").upper()  # Default to INFO
    format_str = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        level = getattr(logging, level_str)
    except AttributeError:
        level = logging.INFO  # Default to INFO if invalid level
        logger.warning(f"Invalid logging level in config.yaml: {level_str}. Using INFO instead.")

    logging.basicConfig(level=level, format=format_str)

# --- Gemini Model Class ---
class GeminiModel:
    """
    Clase principal para interactuar con el modelo Gemini de Google.
    Maneja la inicialización y generación de respuestas.
    """
    def __init__(self, config: Dict[str, Any]):
        # Configuración del logger
        self.logger = logging.getLogger(__name__)
        # Obtener configuración del modelo de la configuración general
        model_config = config["gemini_model"]
        # Configuración básica del modelo
        self.model_name = model_config["model_name"]
        self.temperature = model_config["temperature"]
        self.system_instruction = config.get("bot_messages", {}).get("system_prompt")
        self.image_analysis_prompt_config = config.get("bot_messages", {}).get("image_analysis_prompt")
        self.stop_sequences = model_config["stop_sequences"]
        self.max_output_tokens = model_config["max_output_tokens"]
        self.top_p = model_config["top_p"]
        self.top_k = model_config["top_k"]
        self.safety_settings = model_config["safety_settings"]

        # Configuración de herramientas (tools)
        raw_tools_config = model_config.get("tools", {})
        #self.tools_config = self._prepare_tools_config(raw_tools_config)
        self.tools_config = None
        # ID del desarrollador para mensajes de error
        self.developer_chat_id = config.get("developer_chat_id")

        # Inicializar el modelo Gemini
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            self.logger.critical(f"Error inicializando modelo Gemini: {e}")
            sys.exit(1)

        # La aplicación de Telegram se asignará más tarde
        self.application = None

    def _prepare_tools_config(self, raw_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepara la configuración de herramientas para la API de Gemini.
        Args:
            raw_config: Configuración en bruto de las herramientas
        Returns:
            Lista de configuraciones de herramientas estructuradas
        """
        if not raw_config:
            return None

        tools = []
        if "function_declarations" in raw_config:
            for func in raw_config["function_declarations"]:
                tool = {
                    "function_declarations": [{
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {
                                param_name: {
                                    "type": param_info["type"].lower(),
                                    "description": param_info["description"]
                                }
                                for param_name, param_info in func["parameters"]["properties"].items()
                            },
                            "required": [
                                param_name
                                for param_name, param_info in func["parameters"]["properties"].items()
                                if param_info.get("required", False)
                            ]
                        }
                    }]
                }
                tools.append(tool)
        return tools if tools else None

    def _build_prompt(self, content: str) -> str:
        """
        Construye el prompt apropiado basado en el contenido.
        Args:
            content: Contenido del mensaje del usuario
        Returns:
            Prompt construido
        """
        # Para consultas de tiempo, construye un prompt que activará la búsqueda web
        if content.lower().startswith(("what is the current time", "what's the current time", "what time is it")):
            return f"{self.system_instruction}\nPara proporcionar información precisa del tiempo, debo buscar datos actuales.\nuser: {content}"
        return f"{self.system_instruction}\nuser: {content}"

    async def generate_response_non_streaming(self, content: Union[str, Image.Image],
                                            chat_id: Optional[int] = None):
        """
        Genera una respuesta usando la API de Gemini (sin streaming).
        Args:
            content: Contenido del mensaje (texto o imagen)
            chat_id: ID del chat de Telegram
        Yields:
            Fragmentos de la respuesta generada
        """
        try:
            # Configuración de generación (se define aquí para que esté disponible en ambos bloques if/else)
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }

            # Manejo de contenido de texto
            if isinstance(content, str):
                prompt = self._build_prompt(content)
                self.logger.debug(f"Enviando prompt a Gemini: {prompt[:200]}...")

                # Generar respuesta inicial
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )

                if response.text:
                    yield response.text
                else:
                    yield "Lo siento, no pude procesar tu solicitud."

            # Manejo de contenido de imagen
            else:
                self.logger.debug("Procesando imagen...")
                # Convertir imagen a bytes y luego a base64
                buffered = io.BytesIO()
                content.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Construir contenido para análisis de imagen
                contents = [
                    {"mime_type": "image/jpeg", "data": img_str},
                    {"text": self.image_analysis_prompt_config}
                ]

                # Generar respuesta para la imagen
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=False
                )

                if response.text:
                    yield response.text
                else:
                    yield "Lo siento, no pude analizar la imagen."

        except Exception as e:
            error_message = f"Error generando respuesta: {e}"
            self.logger.exception(error_message)
            await self._send_developer_error(error_message, traceback.format_exc(), chat_id)
            yield "Lo siento, ocurrió un error al generar la respuesta."

    async def _send_developer_error(self, error_message: str, traceback_message: str, chat_id: int):
        """
        Envía un mensaje de error al desarrollador vía Telegram.
        Args:
            error_message: Mensaje de error
            traceback_message: Traceback completo del error
            chat_id: ID del chat donde ocurrió el error
        """
        if self.developer_chat_id and self.application and chat_id:
            if self.developer_chat_id == chat_id:
                self.logger.warning("ID del desarrollador es el mismo que el ID del bot. No se envía mensaje de error.")
                return

            try:
                await self.application.bot.send_message(
                    chat_id=self.developer_chat_id,
                    text=f"Error en chat {chat_id}:\n{error_message}\n\nTraceback:\n{traceback_message}"
                )
            except Exception as telegram_error:
                self.logger.exception(f"Error enviando mensaje al desarrollador vía Telegram: {telegram_error}")

async def process_image(update: Update, context: CallbackContext):
    """Processes an incoming image message with error handling."""
    bot = context.bot
    try:
        if update.message.photo:
            image_file = await bot.get_file(update.message.photo[-1].file_id)
            image_bytes = await image_file.download_as_bytearray()
            image = Image.open(io.BytesIO(image_bytes))
            return image
        elif update.message.document:
            if update.message.document.mime_type.startswith('image/'):
                image_file = await bot.get_file(update.message.document.file_id)
                image_bytes = await image_file.download_as_bytearray()
                image = Image.open(io.BytesIO(image_bytes))
                return image
    except Exception as e:
        logger.error(f"Error processing image: {e}") # Log error here
        return None # Still return None to indicate failure

    return None

# --- Persistent Hybrid Memory Class ---
class PersistentHybridMemory:
    """
    Manages persistent hybrid memory using SQLite.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        database_config = config.get("database", {})
        self.db_path = database_config.get("db_path", "chat_memory.db")
        self.short_term_limit = database_config.get("short_term_limit", 10)
        self.top_k = database_config.get("top_k", 3)
        self.tools_config = config.get("gemini_model", {}).get("tools") # Load tools configuration from config
        embedding_model_config = config.get("embedding_model", {})
        model_name = embedding_model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._create_table() # Call _create_table here in __init__

    def _create_table(self): # --- RESTORED _create_table METHOD HERE ---
        """Creates the messages table if it doesn't exist and adds indexes."""
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT
            )
        ''')
        # --- ADD INDEXES HERE, INSIDE _create_table ---
        self._cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON messages (chat_id, timestamp DESC)
        ''')
        self._conn.commit()
        
    def add_message(self, chat_id: int, role: str, content: str):
        """Adds a message to the database, calculating and storing its embedding."""
        embedding = self.embedding_model.encode(content).tolist()  # Convert to list of floats
        # Convert the embedding to JSON string for storage
        embedding_str = json.dumps(embedding)
        self._cursor.execute('''
            INSERT INTO messages (chat_id, role, content, embedding)
            VALUES (?, ?, ?, ?)
        ''', (chat_id, role, content, embedding_str))
        self._conn.commit()

    def get_recent_messages(self, chat_id: int) -> List[Dict[str, str]]:
        """Retrieves the last messages (short-term memory) for a given chat."""
        self._cursor.execute('''
            SELECT role, content FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp ASC  -- Changed to ASC for chronological order
            LIMIT ?
        ''', (chat_id, self.short_term_limit))
        rows = self._cursor.fetchall()
        messages = [{"role": row["role"], "content": row["content"]} for row in rows]
        return messages # No reversal needed now

    def get_similar_messages(self, chat_id: int, user_message: str) -> List[Dict[str, str]]:
        """
        Retrieves relevant messages (long-term memory) based on embedding similarity.
        Calculates cosine similarity between the current message's embedding and each stored message.
        """
        query_embedding = self.embedding_model.encode(user_message)
        self._cursor.execute('''
            SELECT role, content, embedding FROM messages
            WHERE chat_id = ?
        ''', (chat_id,))
        rows = self._cursor.fetchall()
        similarities = []
        for row in rows:
            emb_str = row["embedding"]
            # Convert the JSON string back to list of floats
            try:
                emb = np.array(json.loads(emb_str))
            except Exception as ex:
                self.logger.error(f"Error loading embedding from database: {ex}")
                continue
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((similarity, {"role": row["role"], "content": row["content"]}))
        # Select the top_k messages with highest similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_messages = [item[1] for item in similarities[:self.top_k]]
        return top_messages

    def get_context(self, chat_id: int, user_message: str) -> str:
        """
        Combines short-term memory (last messages) and relevant messages (embedding search)
        into a string to be incorporated into the prompt.
        """
        messages = []
        # Short-term memory:
        recent = self.get_recent_messages(chat_id)
        for msg in recent:
            messages.append(f"{msg['role']}: {msg['content']}")
        # Relevant long-term messages:
        similar = self.get_similar_messages(chat_id, user_message)
        for msg in similar:
            messages.append(f"{msg['role']} (remembered): {msg['content']}")
        return "\n".join(messages)

# --- Error Handler Decorator ---
def error_handler(func):
    """Decorator for error handling in bot handlers."""
    async def wrapper(update: Update, context: CallbackContext):
        try:
            await func(update, context)
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_message = f"Error in handler {func.__name__}: {e}"
            logger = logging.getLogger(__name__)
            logger.exception(error_message)
            # Get user error message from config, with fallback
            user_message_template = config.get('bot_messages', {}).get('generic_error_message', "An error occurred. Please try again later. (Error ID: {error_id})")
            user_message = user_message_template.format(error_id=error_id) # Format with error_id

            if update and update.effective_chat:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=user_message)
            # Assumes the 'gemini' variable is global
            await gemini._send_developer_error(error_message, traceback.format_exc(),
                                               update.effective_chat.id if update and update.effective_chat else None)
    return wrapper

# --- Search Tool Functions ---# --- Search Tool Functions ---
def search_serpapi(query: str, serpapi_config: Dict[str, Any]) -> Optional[List[Dict[str, str]]]: # Changed return type to List[Dict[str, str]]
    """
    Realiza una búsqueda en Google usando SerpAPI y devuelve los snippets de los resultados orgánicos.
    Now returns a list of dictionaries, each with 'title' and 'snippet'.
    """
    api_key = serpapi_config.get("api_key")
    if not api_key:
        logger.error("SerpAPI API key is missing in configuration.")
        return None # Return None if API key is missing for error handling

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": serpapi_config.get("num_results", 5),
            "gl": serpapi_config.get("gl", "mx"),
            "hl": serpapi_config.get("hl", "es"),
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        if not organic_results:
            return [] # Return empty list if no organic results found

        formatted_results = [] # List to store dictionaries of results
        for result in organic_results:
            snippet = result.get("snippet")
            title = result.get("title")
            if snippet and title:
                formatted_results.append({"title": title, "snippet": snippet}) # Append dictionary

        return formatted_results # Return list of dictionaries

    except Exception as e:
        logger.error(f"Error al realizar la búsqueda en SerpAPI: {e}")
        return None # Return None if there's an error

# --- Global variables ---
application: Optional[Application] = None
gemini: Optional[GeminiModel] = None
config: Dict[str, Any] = {}
persistent_memory: Optional[PersistentHybridMemory] = None

# --- Bot Handlers ---
@error_handler
async def start(update: Update, context: CallbackContext):
    """Sends the welcome message."""
    logger.info("Comando /start recibido")
    message = config.get('bot_messages', {}).get('welcome_message')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

@error_handler
async def help_command(update: Update, context: CallbackContext):
    """Sends the help message."""
    logger.info("Comando /help recibido")
    message = config.get('bot_messages', {}).get('help_message')
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

@error_handler
async def assistant_handler(update: Update, context: CallbackContext):
    """Processes text messages, using persistent memory to enrich the context."""
    chat_id = update.effective_chat.id
    user_message = update.message.text
    logger.info("assistant_handler called for message")
    logger.debug(f"Message received: {user_message}")

    # Get accumulated context from the database
    context_memory = persistent_memory.get_context(chat_id, user_message)
    # Build the prompt combining SYSTEM_PROMPT, context, and current message
    prompt = gemini.system_instruction + "\n" + context_memory + "\nuser: " + user_message

    # Add the user message to persistent memory
    persistent_memory.add_message(chat_id, "user", user_message)

    response_text = ""
    try:
        async for chunk in gemini.generate_response_non_streaming(prompt, chat_id=chat_id):
            response_text += chunk
        # Add the bot's response to persistent memory
        persistent_memory.add_message(chat_id, "assistant", response_text)
        await context.bot.send_message(chat_id=chat_id, text=response_text)
    except Exception as e:
        error_message = f"Error in assistant_handler: {e}"
        logger.exception(error_message)
        await context.bot.send_message(chat_id=chat_id,
                                       text="An error occurred while processing your request.")

@error_handler
async def image_handler(update: Update, context: CallbackContext):
    """Handles incoming image messages and processes them."""
    bot = context.bot
    chat_id = update.effective_chat.id

    # Detect if the message is an image
    if update.message and (
        update.message.photo
        or (update.message.document and update.message.document.mime_type.startswith('image/'))
    ):
        try:
            logger.info("Image received. Processing...")
            # Get the image
            if update.message.photo:
                image_file = await bot.get_file(update.message.photo[-1].file_id)
            elif update.message.document:
                image_file = await bot.get_file(update.message.document.file_id)

            image_bytes = await image_file.download_as_bytearray()
            image = Image.open(io.BytesIO(image_bytes))
            if image:
                logger.info("Image obtained successfully.")

                prompt = update.message.caption if update.message.caption else "Analyze this image and generate a response"
                #Construir el prompt combinando system_prompt y lo que ingrese el usuario
                image_prompt = prompt#gemini.system_instruction + "\nuser: " + prompt

                response_text = ""
                async for chunk in gemini.generate_response_non_streaming(image, chat_id=chat_id): #Pasar imagen aca
                    response_text += chunk

                # Send the response
                await bot.send_message(chat_id=chat_id, text=response_text)
            else:
                logger.warning("Could not get the image from the message.")
                await bot.send_message(chat_id=chat_id, text="Please send a valid image.")

        except Exception as e:
            error_message = f"Error processing image: {e}"
            logger.exception(error_message)
            await bot.send_message(
                chat_id=chat_id,
                text="An error occurred while processing the image. Please try again."
            )
    else:
        await bot.send_message(chat_id=chat_id, text="Please send a valid image.")

# Example of calling search_serpapi from a handler async def handle_search_command(update: Update, context: CallbackContext):
    user_query = update.message.text # Or get query from command arguments, etc.
    serpapi_conf = config.get('serpapi', {}) # Get the 'serpapi' config section
    search_results = search_serpapi(user_query, serpapi_conf) # Pass query and config

    if search_results:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=search_results)
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Search failed or no results found.")

# --- SEARCH HANDLER ---
@error_handler
async def search_handler(update: Update, context: CallbackContext):
    """Handles the /search command and performs a web search, formatting output in MarkdownV2 with escaping."""
    query = ' '.join(context.args)
    if not query:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a search query after the /search command.")
        return

    serpapi_conf = config.get('serpapi', {})
    search_results = search_serpapi(query, serpapi_conf)

    if search_results:
        message_text = ""
        for result in search_results:
            title = result["title"]
            snippet = result["snippet"]

            # Escape special MarkdownV2 characters in title and snippet
            escaped_title = escape_markdown_v2(title)
            escaped_snippet = escape_markdown_v2(snippet)

            message_text += f"*{escaped_title}*\n{escaped_snippet}\n\n"

        await context.bot.send_message(chat_id=update.effective_chat.id, text=message_text, parse_mode="MarkdownV2")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No relevant search results found or search failed.")

def escape_markdown_v2(text: str) -> str:
    
    """Escapes special characters for Telegram MarkdownV2. (Version 2 - added '|')"""
    escape_chars = r'\.()\-+={}!#*|' # Added '|' to the list of characters to escape
    return "".join([f"\{char}" if char in escape_chars else char for char in text])

# --- Main function ---
def main() -> None:
    """Sets up and runs the bot application."""
    global application, gemini, config, persistent_memory

    config = load_config()
    setup_logging(config)

    TOKEN = config["bot_token"]
    GOOGLE_API_KEY = config["gemini_api_key"]

    # Initialize the Telegram application
    application = Application.builder().token(TOKEN).build()

    # Configure the genai API
    genai.configure(api_key=GOOGLE_API_KEY)

    # Initialize GeminiModel and assign the application
    gemini = GeminiModel(config)
    gemini.application = application

    # Initialize persistent memory
    persistent_memory = PersistentHybridMemory(config)

    # Add the handlers
    application.add_handler(CommandHandler(config.get('commands',{}).get('start',"start"), start))
    application.add_handler(CommandHandler(config.get('commands',{}).get('help',"help"), help_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), assistant_handler))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler))
    #application.add_handler(CommandHandler("search", search_handler))

    logger.info("Bot is about to start polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_dotenv()
    main()