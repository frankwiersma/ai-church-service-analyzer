# Dockerfile
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files (excluding recordings folder)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (explicitly exclude recordings)
COPY analyses_prompt.txt .
COPY transcribe.py .
COPY readme.md .
COPY NieuweKerkToevoegen.png .

# Ensure any required environment variables are accessible
ENV DEEPGRAM_API_KEY=""
ENV GEMINI_API_KEY=""
ENV AUTHORIZATION_TOKEN=""
ENV TELEGRAM_BOT_TOKEN=""
ENV RECENT_AMOUNT=1
ENV TRANSCRIPTION_SERVICE="deepgram"

# Expose port if necessary (e.g., for the bot server)
EXPOSE 5000

# Create directory for mounting
RUN mkdir -p /app/recordings

# Run the main script when the container starts
CMD ["python", "transcribe.py"]