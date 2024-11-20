# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    requests \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python requirements file first for dependency caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir moviepy==2.0.0

# Copy application files to the container
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

# Create a directory for processed services
RUN mkdir -p /app/processed_services

# Add crontab file for the job schedule
RUN echo "0 * * * * cd /app && python jobschedule.py" > /etc/cron.d/jobschedule
RUN chmod 0644 /etc/cron.d/jobschedule
RUN crontab /etc/cron.d/jobschedule

# Set the default command to run cron in the foreground
CMD ["cron", "-f"]
