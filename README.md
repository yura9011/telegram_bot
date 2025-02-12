# Gemini Pro Personal Assistant Telegram Bot (Version 2)

This is a Telegram bot that uses the Gemini Pro and Gemini Pro Vision models to provide a personal assistant experience. The bot can respond to text messages, analyze images, and perform web searches, providing helpful and informative responses.

## Features

*   **Text Message Handling:** Responds to text messages using the Gemini Pro model, leveraging persistent memory for context.
*   **Image Analysis:** Analyzes images using the Gemini Pro Vision model and provides a detailed description, with improved prompting.
*   **Persistent Memory:** Remembers previous conversations using a SQLite database and sentence embeddings, enhancing contextual understanding.
*   **Web Search (New in Version 2):** Integrates with SerpAPI to perform web searches and provide real-time information.  Can be enabled via configuration.
*   **Configuration via YAML:**  Bot settings are configured via a `config.yaml` file, allowing for easy customization.
*   **Error Handling:** Robust error handling with developer notifications for efficient debugging.
*   **MarkdownV2 Output:** Search results are formatted using MarkdownV2 for improved readability in Telegram.

## Requirements

*   Python 3.9 or higher
*   [Poetry](https://python-poetry.org/) for dependency management
*   A Telegram Bot token (obtained from BotFather on Telegram)
*   A Google Gemini API key (enable the Gemini API in the Google Cloud Console)
*   A SerpAPI API key (optional, for web search functionality)

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

    *   Create a `config.yaml` file.  See the project documentation or example files for configuration options.
    *   Set the `bot_token` and `gemini_api_key` in the `config.yaml` file.
    *   If you want to enable web search, set the `serpapi_api_key` in the `config.yaml` file.
    *   Set the `developer_chat_id` to your Telegram user ID to receive error notifications.



## Usage

Start a chat with the bot on Telegram.

Use the `/start` command to display the welcome message.

Use the `/help` command to display the help message.

Send text messages to the bot to generate a response. The bot will use its memory of past conversations to provide more relevant answers.

Send images to the bot (with or without a caption) and the bot will analyze the image and provide a description.

Use the `/search` command followed by a query to perform a web search (if enabled). For example: `/search current weather in Mexico City`. ** NOT WORKING NOW **

## What's New in Version 2

*   **Web Search Integration:** The bot can now perform web searches using the SerpAPI to answer questions that require real-time information.  Enable this feature by configuring your SerpAPI key in `config.yaml`.
*   **Improved Image Analysis:** Enhanced prompts and handling for image analysis, allowing the bot to provide more detailed and relevant descriptions.
*   **MarkdownV2 Output:** Search results are formatted using MarkdownV2 for cleaner and more readable output in Telegram.
*   **Enhanced Error Handling:** More detailed error messages and improved logging for easier debugging.
*   **Code Refactoring:**  Significant code refactoring for better maintainability and readability.

## Error Handling

The bot includes robust error handling with developer notifications. If an error occurs, a message with the error details will be sent to the Telegram user ID specified in the `developer_chat_id` setting. Check your Telegram for error notifications if the bot isn't behaving as expected.

## License

This project is licensed under the MIT License.