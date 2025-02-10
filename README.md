# Gemini Pro Personal Assistant Telegram Bot ( AI-GENERATE README)

This is a Telegram bot that uses the Gemini Pro and Gemini Pro Vision models to provide a personal assistant experience. The bot can respond to text messages and analyze images, providing helpful and informative responses.

## Features

*   Text Message Handling: Responds to text messages using the Gemini Pro model.
*   Image Analysis: Analyzes images using the Gemini Pro Vision model and provides a detailed description.
*   Persistent Memory: Remembers previous conversations using a SQLite database and sentence embeddings.
*   Configuration via YAML: All settings are configured via a `config.yaml` file.
*   Error Handling: Robust error handling with developer notifications.

## Requirements

*   Python 3.9 or higher
*   [Poetry](https://python-poetry.org/) for dependency management
*   A Telegram Bot token
*   A Google Gemini API key
*   Tesseract OCR (optional, for extracting text from images)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

3.  **Configure the bot:**

    *   Create a `config.yaml` file (see example below).
    *   Set the `bot_token` and `gemini_api_key` in the `config.yaml` file.
    *   Set the `developer_chat_id` to your Telegram user ID to receive error notifications.

4.  **Set up Tesseract OCR (Optional):**

    *   If you want the bot to be able to extract text from images, install Tesseract OCR: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   Configure the path to the Tesseract executable in your code (see comments in `main.py`).

5.  **Run the bot:**

    ```bash
    poetry run python main.py
    ```

## Configuration (config.yaml)

Here's an example `config.yaml` file:

```yaml
bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
gemini_api_key: "YOUR_GEMINI_API_KEY"
developer_chat_id: YOUR_TELEGRAM_USER_ID

gemini_model:
  model_name: "gemini-pro-vision"
  temperature: 0.7
  stop_sequences: null
  max_output_tokens: null
  top_p: null
  top_k: null
  safety_settings: null
  tools: null

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

bot_messages:
  system_prompt: |
    You are Aurora, an intelligent and empathetic personal assistant. 
    Analyze the image and provide a detailed description.
  welcome_message: "Hello! I'm your Personal Assistant Bot, now powered by the Gemini Model!"
  help_message: "I'm your Personal Assistant Bot, now powered by Gemini!\nType in any questions or requests and I'll use Gemini to help you.\nCommands:\n/start - Start the bot and display the welcome message.\n/help - Show this help message."

database:
  db_path: "chat_memory.db"
  short_term_limit: 10
  top_k: 3

embedding_model:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

commands:
  start: "start"
  help: "help"

Usage

Start a chat with the bot on Telegram.

Use the /start command to display the welcome message.

Use the /help command to display the help message.

Send text messages to the bot to generate a response.

Send images to the bot (with or without a caption) and the bot will analyze the image and provide a description.

Error Handling

The bot includes robust error handling with developer notifications. If an error occurs, a message with the error details will be sent to the Telegram user ID specified in the developer_chat_id setting.

Dependencies

python-telegram-bot

google-generativeai

python-dotenv

PyYAML

Pillow

pytesseract

sentence-transformers

numpy

aiosqlite (optional, for asynchronous database access)

Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.