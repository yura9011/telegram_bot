import logging
import os
import asyncio
import assemblyai as aai
from telegram import Update
from telegram.ext import MessageHandler, filters, CallbackContext, Application
from typing import Dict, Any
from utils.error_handler import error_handler
from models.gemini_model import GeminiModel
from models.persistent_memory import PersistentHybridMemory

logger = logging.getLogger(__name__)

@error_handler
async def voice_message_handler(update: Update, context: CallbackContext):
    """Handles voice messages."""
    chat_id = update.effective_chat.id
    voice = update.message.voice
    config = context.bot_data["config"]
    gemini = context.bot_data["gemini"]
    memory = context.bot_data["memory"]

    try:
        # Configure AssemblyAI
        assemblyai_api_key = config.get('assemblyai_api_key')
        aai.settings.api_key = assemblyai_api_key

        # Get transcription config
        assemblyai_config = config.get('assemblyai', {})
        transcription_config = aai.TranscriptionConfig(
            speech_model=assemblyai_config.get('speech_model', 'best'),
            punctuate=assemblyai_config.get('punctuate', True),
            speaker_labels=True
        )

        # Create transcriber
        transcriber = aai.Transcriber(config=transcription_config)

        # Get audio URL from Telegram
        file = await context.bot.get_file(voice.file_id)
        audio_url = file.file_path

        # Transcribe audio
        transcript = await asyncio.to_thread(transcriber.transcribe, audio_url)

        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"Transcription failed: {transcript.error}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Sorry, transcription failed: {transcript.error}"
            )
            return

        transcribed_text = transcript.text
        logger.info(f"Transcribed text: {transcribed_text}")

        # Process with Gemini
        response_text = ""
        async for chunk in gemini.generate_response_non_streaming(transcribed_text, chat_id=chat_id):
            response_text += chunk

        # Store in memory
        memory.add_message(chat_id, "user", f"Voice message: {transcribed_text}")
        memory.add_message(chat_id, "assistant", response_text)

        await context.bot.send_message(chat_id=chat_id, text=response_text)

    except Exception as e:
        logger.exception(f"Error processing voice message: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Sorry, there was an error processing your voice message."
        )

def register_voice_handlers(
    application: Application,
    config: Dict[str, Any],
    gemini: GeminiModel,
    memory: PersistentHybridMemory
) -> None:
    """Registers voice message handlers with the application."""
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler))