# Gemini Pro Personal Assistant Telegram Bot (Version 3)

A Telegram bot powered by Google Gemini Pro & Vision, and AssemblyAI.  Engages in text and voice conversations, analyzes images, and performs web searches, remembering context.

## Features

*   **Text & Voice Chat:**  Converses in text and transcribes voice messages using Gemini Pro.
*   **Image Analysis:** Describes images with Gemini Pro Vision.
*   **Web Search:**  Provides real-time info via SerpAPI.
*   **Persistent Memory:** Remembers conversations.
*   **YAML Config:** Easy setup with `config.yaml`.
*   **Error Notifications:**  Developer error alerts via Telegram.

## Requirements

*   Python 3.9+
*   Poetry
*   Telegram Bot Token
*   Google Gemini API Key
*   AssemblyAI API Key
*   SerpAPI API Key (optional)

## Installation

1.  **Clone:** `git clone [repository URL]`
2.  **Install:** `poetry install`
3.  **Config:** Create `config.yaml` (see example), set API keys.

## Usage

1.  **Run:** `python main.py`
2.  **Telegram:** Start chat with your bot.
3.  **Commands:**
    *   `/start`: Welcome message
    *   `/help`: Help info
    *   Text & Voice messages: Conversational AI
    *   Images: Image analysis
    *   `/search [query]`: Web search

## Version 3 Highlights

*   **Voice Message Support:**  Interact via voice.
*   **Web Search:** Real-time information access.
*   **Concise & Modular Code.**

## Error Handling

Developer receives Telegram error alerts. Check Telegram for bot notifications.

## License

MIT License