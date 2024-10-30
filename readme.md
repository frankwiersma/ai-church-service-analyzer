# AI Church Service Analyzer

A Dockerized tool for transcribing and analyzing church service recordings using AI. It provides a structured overview of each service, including sermon themes, liturgy, community announcements, and notable moments, delivered through a Telegram bot.

## Project Structure

```
ai-church-service-analyzer
├── Dockerfile              # Defines the Docker image setup
├── analyses_prompt.txt     # Format template for the analysis output
├── requirements.txt        # Python dependencies
├── readme.md               # Project documentation
├── transcribe.py           # Main application script
└── docker-compose.yml      # Docker Compose configuration
```

## Setup Instructions

### 1. Clone the Repository

Clone and enter the project directory:
```bash
git clone https://github.com/frankwiersma/ai-church-service-analyzer.git
cd ai-church-service-analyzer
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory with the following keys:

```plaintext
DEEPGRAM_API_KEY=your_deepgram_api_key
GEMINI_API_KEY=your_gemini_api_key
AUTHORIZATION_TOKEN=your_auth_token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
RECENT_AMOUNT=1
TRANSCRIPTION_SERVICE=deepgram  # or 'gemini'
```

### 3. Run with Docker Compose

Use Docker Compose to build and run the service:

```bash
docker-compose up --build
```

This setup binds the `recordings/` folder to the Docker container, which will store downloaded audio files and analyses.

## Usage

Interact with the bot on Telegram using `/start` to begin selecting and analyzing church services.

## Features

- **Automatic Downloads**: Retrieves recent recordings from specified churches.
- **Transcription and Analysis**: Transcribes audio files using either Deepgram or Gemini APIs and formats the analysis.
- **Telegram Bot Interface**: Users can interact and retrieve analyses directly from a Telegram bot.

## Notes

- Analysis prompts are customized in `analyses_prompt.txt`.
- The bot handles cancellation requests during download and transcription.
  
## Dependencies

This project requires Python libraries defined in `requirements.txt`:

- `requests`
- `google-generativeai`
- `moviepy`
- `python-dotenv`
- `python-telegram-bot`
- `deepgram-sdk`
