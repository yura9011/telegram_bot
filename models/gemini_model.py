import logging
import traceback
from typing import Dict, Any, Union, AsyncGenerator
import google.generativeai as genai
from PIL import Image
import base64
import io

class GeminiModel:
    """Main class for interacting with Google's Gemini model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        model_config = config["gemini_model"]
        
        # Basic model configuration
        self.model_name = model_config["model_name"]
        self.temperature = model_config["temperature"]
        self.system_instruction = config.get("bot_messages", {}).get("system_prompt")
        self.image_analysis_prompt_config = config.get("bot_messages", {}).get("image_analysis_prompt")
        self.stop_sequences = model_config["stop_sequences"]
        self.max_output_tokens = model_config["max_output_tokens"]
        self.top_p = model_config["top_p"]
        self.top_k = model_config["top_k"]
        self.safety_settings = model_config["safety_settings"]
        self.developer_chat_id = config.get("developer_chat_id")
        
        # Initialize Gemini model
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            self.logger.critical(f"Error initializing Gemini model: {e}")
            raise

        self.application = None

    def _build_prompt(self, content: str) -> str:
        """Builds appropriate prompt based on content."""
        if content.lower().startswith(("what is the current time", "what's the current time", "what time is it")):
            return f"{self.system_instruction}\nPara proporcionar información precisa del tiempo, debo buscar datos actuales.\nuser: {content}"
        return f"{self.system_instruction}\nuser: {content}"

    async def generate_response_non_streaming(
        self, 
        content: Union[str, Image.Image],
        chat_id: int = None
    ) -> AsyncGenerator[str, None]:
        """Generates a response using the Gemini API (non-streaming)."""
        try:
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }

            if isinstance(content, str):
                prompt = self._build_prompt(content)
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )
                yield response.text if response.text else "Lo siento, no pude procesar tu solicitud."
            
            else:  # Image content
                buffered = io.BytesIO()
                content.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                contents = [
                    {"mime_type": "image/jpeg", "data": img_str},
                    {"text": self.image_analysis_prompt_config}
                ]
                
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=False
                )
                yield response.text if response.text else "Lo siento, no pude analizar la imagen."

        except Exception as e:
            error_message = f"Error generating response: {e}"
            self.logger.exception(error_message)
            await self._send_developer_error(error_message, traceback.format_exc(), chat_id)
            yield "Lo siento, ocurrió un error al generar la respuesta."

    async def _send_developer_error(self, error_message: str, traceback_message: str, chat_id: int):
        """Sends error message to developer via Telegram."""
        if self.developer_chat_id and self.application and chat_id:
            if self.developer_chat_id == chat_id:
                self.logger.warning("Developer ID is the same as bot ID. Not sending error message.")
                return

            try:
                await self.application.bot.send_message(
                    chat_id=self.developer_chat_id,
                    text=f"Error in chat {chat_id}:\n{error_message}\n\nTraceback:\n{traceback_message}"
                )
            except Exception as telegram_error:
                self.logger.exception(f"Error sending message to developer via Telegram: {telegram_error}")