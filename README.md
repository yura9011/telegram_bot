# Gemini Pro Personal Assistant Telegram Bot

A versatile Telegram bot powered by Google Gemini Pro & Vision, AssemblyAI, and Google Calendar API. This bot engages in text and voice conversations, analyzes images, performs web searches, manages calendar events, and remembers conversation context for a more personalized experience.

## Features

*   **Intelligent Text & Voice Chat:**  Engages in natural text-based conversations and transcribes voice messages to text using Gemini Pro for seamless voice interaction.
*   **Advanced Image Analysis:**  Analyzes and describes images in detail using Google Gemini Pro Vision, providing insights into visual content.
*   **Real-time Web Search:**  Provides access to up-to-date information by performing web searches via SerpAPI, delivering relevant search results directly within Telegram.
*   **Google Calendar Integration:**
    *   **List Upcoming Events:**  Retrieve and display upcoming events from your Google Calendar using the `/calendar_list` command.
    *   **Create Events via Commands:** Create new calendar events directly from Telegram using the `/calendar_create` command, specifying event details.
    *   **Natural Language Event Creation:**  Intelligently understands natural language requests to create calendar events from text messages.
*   **Persistent Hybrid Memory:**  Remembers conversation history using a persistent hybrid memory system, allowing for contextual and more coherent interactions over time.
*   **Flexible YAML Configuration:**  Easy and customizable setup through a `config.yaml` file, allowing you to adjust bot behavior and API keys without modifying code.
*   **Proactive Error Notifications:**  Sends detailed error alerts to the developer via Telegram, facilitating quick debugging and maintenance.

## Requirements

*   Python 3.9+
*   pip package installer (or your preferred Python package manager)
*   Telegram Bot Token (Get from BotFather on Telegram)
*   Google Gemini API Key (Get from Google AI Studio)
*   AssemblyAI API Key (Get from AssemblyAI)
*   SerpAPI API Key (Optional, for web search functionality - Get from SerpAPI)
*   Google Calendar API Credentials: `credentials.json` (Download from Google Cloud Console for Calendar API access)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    *   Create a `config.yaml` file in the root directory based on the `config.yaml.example` provided in the repository.
    *   **Set your API keys** in the `config.yaml` file for Telegram Bot, Google Gemini, AssemblyAI, and SerpAPI (if you intend to use web search).
    *   **Place your `credentials.json` file** (downloaded from Google Cloud Console for Google Calendar API) in the `config` directory as specified in `config.yaml`.

## Usage

1.  **Run the bot:**
    ```bash
    python main.py
    ```

2.  **Start a chat with your bot on Telegram.**

3.  **Available Commands:**
    *   `/start`: Displays a welcome message from the bot.
    *   `/help`: Shows helpful information and available commands.
    *   `/search [query]`: Performs a web search using SerpAPI and returns summarized results.
    *   `/calendar_auth`: Authenticates with Google Calendar to authorize access. **(Run this command first to enable calendar features)**
    *   `/calendar_list`: Lists upcoming events from your Google Calendar.
    *   `/calendar_create [title] [date (YYYY-MM-DD, optional)] [time (HH:MM, optional)] [description (optional)]`: Creates a new event in your Google Calendar using command arguments.
    *   **Text & Voice Messages:** Send text or voice messages to engage in conversational AI with Gemini Pro.
    *   **Images:** Send images to the bot for detailed image analysis using Gemini Pro Vision.
    *   **Natural Language Calendar Event Creation:**  Simply send a text message with your event details (e.g., "Create an event called 'Team Meeting' tomorrow at 2 PM") and the bot will attempt to understand and create the calendar event.

## Version 3 Highlights

*   **Voice Message Support:**  Enhanced user experience with voice message transcription and response capabilities.
*   **Web Search Integration:**  Access real-time information and summaries directly through the bot.
*   **Google Calendar Functionality:**
    *   List upcoming calendar events.
    *   Create new calendar events using commands and natural language.
*   **Improved Natural Language Understanding for Calendar Events:**  Create calendar events simply by describing them in a text message.
*   **Concise and Modular Codebase:**  Improved code structure for better maintainability and extensibility.

## Error Handling

In case of errors, detailed error messages are logged, and the developer will receive error notifications directly via Telegram, enabling prompt issue resolution.

## License

[MIT License](LICENSE) 