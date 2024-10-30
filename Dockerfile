# Use a Python 3.10 or later image to support match-case statements
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Ensure any required environment variables are accessible
ENV DEEPGRAM_API_KEY=""
ENV GEMINI_API_KEY=""
ENV AUTHORIZATION_TOKEN=""
ENV TELEGRAM_BOT_TOKEN=""
ENV RECENT_AMOUNT=1
ENV TRANSCRIPTION_SERVICE="deepgram"

# Expose port if necessary (e.g., for the bot server)
EXPOSE 5000

# Run the main script when the container starts
CMD ["python", "transcribe.py"]
