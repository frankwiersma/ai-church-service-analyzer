# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
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
    pip3 install --no-cache-dir requests && \
    pip3 install --no-cache-dir moviepy==1.0.3 numpy>=1.18.1 imageio>=2.5.0 decorator>=4.3.0 tqdm>=4.0.0 Pillow>=7.0.0 scipy>=1.3.0 pydub>=0.23.0 audiofile>=0.0.0 opencv-python>=4.5 && \
    pip3 install --no-cache-dir -r requirements.txt

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
ENV TELEGRAM_CHAT_ID=""

# Create a directory for processed services
RUN mkdir -p /app/processed_services

# **Add executable permissions to the jobschedule.py script**
RUN chmod +x /app/jobschedule.py

# **Add crontab file for the job schedule with 'root' user and redirect output to a log file**
RUN echo "0 * * * * root cd /app && python3 jobschedule.py >> /var/log/cron.log 2>&1" > /etc/cron.d/jobschedule
RUN chmod 0644 /etc/cron.d/jobschedule
RUN crontab /etc/cron.d/jobschedule

# **Create the log file to capture cron job output**
RUN touch /var/log/cron.log

# Set the default command to run cron in the foreground
CMD ["cron", "-f"]
