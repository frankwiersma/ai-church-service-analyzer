# Dockerfile
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install cron
RUN apt-get update && apt-get install -y \
    cron \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files (excluding recordings folder)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (explicitly exclude recordings)
COPY jobschedule.py .
COPY analysis_prompt.txt .
COPY readme.md .
COPY html_report_prompt.txt .
COPY .env .

# Ensure any required environment variables are accessible
ENV DEEPGRAM_API_KEY=""
ENV GEMINI_API_KEY=""
ENV AUTHORIZATION_TOKEN=""
ENV RECENT_AMOUNT=""
ENV TRANSCRIPTION_SERVICE="deepgram"
ENV SENDGRID_API_KEY=""
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY=""
ENV NEXT_PUBLIC_SUPABASE_URL=""
ENV EMAIL_DOMAIN=""
ENV EMAIL_RECIPIENT=""
ENV TELEGRAM_BOT_TOKEN=""

# Create directory for mounting
RUN mkdir -p /app/processed_services

# Add crontab file
RUN echo "0 * * * * cd /app && python jobschedule.py" > /etc/cron.d/jobschedule
RUN chmod 0644 /etc/cron.d/jobschedule
RUN crontab /etc/cron.d/jobschedule

# Run cron in the foreground
CMD ["cron", "-f"]