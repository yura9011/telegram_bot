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

# Disable sentence_transformers logger
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Create logger for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
def load_config() -> Dict[str, Any]:
    """Loads configuration from config.yaml."""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError as e:
        logger.critical(f"Error reading config.yaml: {e}. Exiting.")
        sys.exit(1)

# --- Logging ---
def setup_logging(config: Dict[str, Any]):
    """Sets up logging according to the configuration."""
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO").upper()  # Default to INFO
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
    Class to interact with Google's Gemini model.
    Handles initialization and response generation (without streaming).
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        model_config = config["gemini_model"]
        self.model_name = model_config["model_name"]
        self.temperature = model_config["temperature"]
        self.system_instruction = config.get("bot_messages", {}).get("system_prompt")
        self.stop_sequences = model_config["stop_sequences"]
        self.max_output_tokens = model_config["max_output_tokens"]
        self.top_p = model_config["top_p"]
        self.top_k = model_config["top_k"]
        self.safety_settings = model_config["safety_settings"]
        self.tools = model_config["tools"]
        self.developer_chat_id = config.get("developer_chat_id")
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            self.logger.critical(f"Error initializing Gemini model: {e}")
            sys.exit(1)
        # Telegram application will be assigned later
        self.application = None

    async def _send_developer_error(self, error_message: str, traceback_message: str, chat_id: int):
        """Sends an error message to the developer via Telegram."""
        if self.developer_chat_id and self.application and chat_id:
            if self.developer_chat_id == chat_id:
                self.logger.warning("Developer chat ID is the same as the bot ID. Not sending error message.")
                return

            try:
                #await self.application.bot.send_message(
                #    chat_id=self.developer_chat_id,
                #    text=f"{error_message}\n{traceback_message}"
                #)
                #Do not sent for being a bot
                self.logger.warning("Could not send the message to the developer for being a bot")
            except Exception as telegram_error:
                self.logger.exception(f"Failed to send message to developer: {telegram_error}")

    def _build_prompt(self, prompt: str) -> str:
        """Builds the prompt by combining system instructions and user message."""
        return self.system_instruction + prompt

    async def generate_response_non_streaming(self, content: Union[str, Image.Image], #Union para que reciba tanto imagen como str
                                              tools: Optional[List[Dict[str, Any]]] = None,
                                              safety_settings: Optional[List[Dict[str, Any]]] = None,
                                              system_instruction: Optional[str] = None,
                                              chat_id: Optional[int] = None):
        """Generates a response using the Gemini API (non-streaming)."""
        try:
            #log_content = content[:200] + "..." if len(content) > 200 else content
            #self.logger.debug(f"Sending prompt to Gemini: {log_content}")

            #Modified part
            if isinstance(content, str):
                prompt = self._build_prompt(content)
                self.logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
                response = self.model.generate_content(prompt, stream=False)
            else:
                self.logger.debug(f"Sending image to Gemini")
                # Convert the image to bytes and then to base64
                buffered = io.BytesIO()
                content.save(buffered, format="JPEG")  # Save the image in a buffer
                img_str = base64.b64encode(buffered.getvalue()).decode()  # Encode to base64

                response = self.model.generate_content(
                    [{"mime_type": "image/jpeg", "data": img_str}],
                    stream=False,
                )
            #End modified part

            if response and response.candidates:
                self.logger.debug("Gemini response has candidates.")
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        self.logger.debug("Candidate has content and parts.")
                        for part in candidate.content.parts:
                            self.logger.debug("Sending part of the response.")
                            yield part.text
                    else:
                        self.logger.debug("Candidate has no content.")
                        yield "Sorry, this candidate had no content."
            else:
                self.logger.debug("Gemini returned an empty response.")
                yield "Sorry, the model returned an empty response."
        except Exception as e:
            error_message = f"Error generating response: {e}"
            self.logger.exception(error_message)
            await self._send_developer_error(error_message, traceback.format_exc(), chat_id)
            yield "Sorry, an error occurred while generating the response."

async def process_image(update: Update, context: CallbackContext):
    """Processes an incoming image message."""
    bot = context.bot
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
    return None

# --- Persistent Hybrid Memory Class ---
class PersistentHybridMemory:
    """
    Manages persistent hybrid memory using SQLite.
    Messages and their embeddings (calculated with SentenceTransformer) are stored in a table
    to retrieve both short-term memory (last N messages) and relevant messages
    (using embedding similarity).
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        database_config = config.get("database", {})
        self.db_path = database_config.get("db_path", "chat_memory.db")
        self.short_term_limit = database_config.get("short_term_limit", 10)
        self.top_k = database_config.get("top_k", 3)

        embedding_model_config = config.get("embedding_model", {})
        model_name = embedding_model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._create_table()

    def _create_table(self):
        """Creates the messages table if it doesn't exist."""
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
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (chat_id, self.short_term_limit))
        rows = self._cursor.fetchall()
        # Invert the order to be chronological
        messages = [{"role": row["role"], "content": row["content"]} for row in rows]
        return messages[::-1]

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
            user_message = f"An error occurred. Please try again later. (Error ID: {error_id})"
            if update and update.effective_chat:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=user_message)
            # Assumes the 'gemini' variable is global
            await gemini._send_developer_error(error_message, traceback.format_exc(),
                                               update.effective_chat.id if update and update.effective_chat else None)
    return wrapper

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

    logger.info("Bot is about to start polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_dotenv()
    main()