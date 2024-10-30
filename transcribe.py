import asyncio
import os
import requests
import json
from datetime import datetime
import time
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, ApplicationBuilder
from deepgram import DeepgramClient, PrerecordedOptions

# Load environment variables
load_dotenv()

# Retrieve variables from .env file
ROOT_FOLDER = os.path.join(os.getcwd(), 'recordings')
recent_amount = int(os.getenv('RECENT_AMOUNT', 1))
transcription_service = os.getenv('TRANSCRIPTION_SERVICE', 'deepgram')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
authorization_token = os.getenv('AUTHORIZATION_TOKEN')

# List of URLs for different churches
BASE_URL = "https://api.kerkdienstgemist.nl/api/v2/stations/{}/recordings"
API_URLS = {
    "Wijngaarden": BASE_URL.format("1306"),
    "Nieuwe Kerk Utrecht": BASE_URL.format("1341"),
    "NGK Doorn": BASE_URL.format("2281"),
    "Elimkerk Hendrik-Ido-Ambacht": BASE_URL.format("876")
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(church, callback_data=f"church_{church}") for church in API_URLS.keys()]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text(
            "🏛️ Welcome! Please select a church to analyze the latest sermon:",
            reply_markup=reply_markup
        )
    else:
        await update.callback_query.message.reply_text(
            "🏛️ Welcome! Please select a church to analyze the latest sermon:",
            reply_markup=reply_markup
        )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data.startswith("church_"):
        church = query.data.replace("church_", "")
        await query.edit_message_text(f"🔄 Fetching available sermons from {church}...")
        
        try:
            services_by_date = await fetch_church_services(API_URLS[church], query)
            if services_by_date:
                keyboard = [[InlineKeyboardButton(date, callback_data=f"date_{church}_{date}")]
                            for date in services_by_date.keys()]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"📅 Please select a date for the sermon from {church}:",
                    reply_markup=reply_markup
                )
            else:
                await query.edit_message_text("❌ No services found for this church.")
        except Exception as e:
            await query.edit_message_text(f"❌ An error occurred: {str(e)}")

    elif query.data.startswith("date_"):
        _, church, date = query.data.split("_", 2)
        services_by_date = await fetch_church_services(API_URLS[church], query)
        services = services_by_date.get(date, [])

        if services:
            keyboard = [[InlineKeyboardButton(service["title"], callback_data=f"service_{church}_{service['id']}")]
                        for service in services]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f"🕒 Please select the specific service for {date}:",
                reply_markup=reply_markup
            )
        else:
            await query.edit_message_text("❌ No services found for this date.")

    elif query.data.startswith("service_"):
        _, church, service_id = query.data.split("_", 2)
        await query.edit_message_text(f"🔄 Processing selected sermon from {church}...")

        try:
            analysis = await process_selected_service(API_URLS[church], service_id, query)
            await handle_analysis_response(query, analysis, church, context)
        except Exception as e:
            await query.edit_message_text(f"❌ An error occurred: {str(e)}")

    elif query.data == "start":
        await start(query, context)



async def fetch_church_services(api_url: str, query: Update.callback_query):
    session = requests.Session()
    headers = {
        'Accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {authorization_token}',
    }
    params = {
        'include': 'media',
        'page': 1,
        'size': 5  # Adjust this number to control how many services are displayed
    }

    # Run blocking operations in a thread pool
    api_response = await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: session.get(api_url, headers=headers, params=params)
    )
    api_response.raise_for_status()
    data = api_response.json()
    recordings = data.get('data', [])

    services_by_date = {}
    for recording in recordings:
        attributes = recording.get('attributes', {})
        service_id = recording.get('id')
        title = attributes.get('title', 'Untitled')
        start_time = attributes.get('start_at', '')

        date_str = parse_date(start_time)
        if date_str not in services_by_date:
            services_by_date[date_str] = []
        
        services_by_date[date_str].append({
            "id": service_id,
            "title": title,
            "date": date_str
        })

    return services_by_date

async def handle_analysis_response(query, analysis, church, context):
    if analysis:
        max_length = 4096
        messages = [analysis[i:i+max_length] for i in range(0, len(analysis), max_length)]
        for i, message in enumerate(messages):
            if i == 0:
                await query.edit_message_text(f"📊 Analysis for {church}:\n\n{message}")
            else:
                await context.bot.send_message(chat_id=query.message.chat_id, text=message)
        keyboard = [[InlineKeyboardButton("Analyze another sermon", callback_data="start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Would you like to analyze another sermon?",
            reply_markup=reply_markup
        )
    else:
        await query.edit_message_text("❌ Failed to generate analysis. Please try again.")

# Modified process_church to handle blocking operations properly
async def process_selected_service(api_url: str, service_id: str, query: Update.callback_query) -> str:
    session = requests.Session()
    headers = {
        'Accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {authorization_token}',
    }

    # Fetch specific service
    await query.edit_message_text("🔍 Fetching selected recording...")
    api_response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: session.get(f"{api_url}/{service_id}", headers=headers)
    )
    api_response.raise_for_status()
    data = api_response.json()
    recording = data.get('data')

    if not recording:
        return "No recording found for the selected service."

    included_media = data.get('included', [])
    attributes = recording.get('attributes', {})
    title = attributes.get('title', 'Untitled')
    start_time = attributes.get('start_at', '')

    date_str = parse_date(start_time)
    folder_name = sanitize_filename(f"{date_str}_{title}")
    recording_folder = os.path.join(ROOT_FOLDER, folder_name)

    if not os.path.exists(recording_folder):
        os.makedirs(recording_folder)

    download_url = get_download_url(recording, included_media)
    if not download_url:
        return "No download URL found for the recording."

    mp4_filename = os.path.join(recording_folder, f"{date_str}_{title.replace(' ', '_')}.mp4")
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')

    # Download with progress
    await query.edit_message_text(f"⏬ Downloading recording... 0% complete")
    await download_file_with_progress(session, download_url, mp4_filename, query)

    # Convert to MP3 with progress
    await query.edit_message_text("🎵 Converting video to audio... 0% complete")
    await convert_to_mp3_with_progress(mp4_filename, query)

    transcription_filename = mp3_filename.replace('.mp3', '_full.txt')
    extracted_filename = mp3_filename.replace('.mp3', '.txt')

    # Transcription
    if not os.path.exists(transcription_filename):
        await query.edit_message_text("🎙️ Transcribing audio...")
        await asyncio.get_event_loop().run_in_executor(
            None,
            transcribe_audio,
            mp3_filename,
            transcription_filename,
            extracted_filename
        )

    # Analysis generation
    await query.edit_message_text("🤖 Generating analysis...")
    analysis_filename = os.path.join(recording_folder, 'analysis.txt')
    await asyncio.get_event_loop().run_in_executor(
        None,
        generate_analysis,
        extracted_filename,
        analysis_filename
    )

    with open(analysis_filename, 'r') as f:
        analysis_text = f.read()

    return analysis_text

# Your existing utility functions remain the same
def parse_date(start_time):
    try:
        date_obj = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S%z')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return start_time[:10]

def get_download_url(recording, included_media):
    relationships = recording.get('relationships', {})
    media_data = relationships.get('media', {}).get('data', [])
    media_ids = {m['id'] for m in media_data}

    for media_item in included_media:
        if media_item['id'] in media_ids and media_item['type'] == 'video_files':
            return media_item.get('attributes', {}).get('download_url', '')
    return None

last_message_data = {}

async def safe_edit_message_text(query, new_text):
    """Edit message text only if the content has changed and enough time has passed."""
    message_id = query.message.message_id
    chat_id = query.message.chat_id
    current_time = time.time()

    # Retrieve the last content and timestamp
    last_text, last_timestamp = last_message_data.get((chat_id, message_id), ("", 0))
    
    # Only update if the content is new or enough time (e.g., 1 second) has passed
    if new_text != last_text and (current_time - last_timestamp > 1):
        # Update the message
        await query.edit_message_text(new_text)
        
        # Store the new content and timestamp
        last_message_data[(chat_id, message_id)] = (new_text, current_time)

async def download_file_with_progress(session, url, filename, query):
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = int((downloaded / total_length) * 100)
                    # Alternate text with an ellipsis to force uniqueness
                    update_text = f"⏬ Downloading recording... {progress}% complete" + ("." if progress % 2 == 0 else "")
                    await safe_edit_message_text(query, update_text)

async def convert_to_mp3_with_progress(mp4_filename, query):
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')
    video_clip = VideoFileClip(mp4_filename)
    audio_clip = video_clip.audio

    # Intermediate status update
    await safe_edit_message_text(query, "🎵 Converting video to audio... 50% complete.")
    audio_clip.write_audiofile(mp3_filename, verbose=False, logger=None)

    # Final status update
    await safe_edit_message_text(query, "🎵 Conversion complete! 100%")

    video_clip.close()
    audio_clip.close()
    return mp3_filename

def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename.replace(' ', '_')

def transcribe_audio(mp3_filename, transcription_filename, extracted_filename):
    if transcription_service == 'gemini':
        transcribe_with_gemini(mp3_filename, transcription_filename)
    elif transcription_service == 'deepgram':
        transcribe_with_deepgram(mp3_filename, transcription_filename, extracted_filename)

def transcribe_with_deepgram(mp3_filename, transcription_filename, extracted_filename):
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        with open(mp3_filename, 'rb') as buffer_data:
            options = PrerecordedOptions(
                punctuate=True, model="nova-2", language="nl"
            )
            response = deepgram.listen.prerecorded.v('1').transcribe_file({'buffer': buffer_data}, options)
            response_data = response.to_dict()
            with open(transcription_filename, 'w') as f:
                json.dump(response_data, f, indent=4)
            print(f"✅ Full Deepgram response saved to {transcription_filename}")

            transcript = ''
            channels = response_data.get('results', {}).get('channels', [])
            if channels:
                alternatives = channels[0].get('alternatives', [])
                if alternatives:
                    transcript = alternatives[0].get('transcript', '')

            if transcript:
                with open(extracted_filename, 'w') as f:
                    f.write(transcript)
                print(f"✅ Extracted transcript saved to {extracted_filename}")
            else:
                print("❌ Transcript not found in the provided JSON data.")
    except Exception as e:
        print(f"❌ Operation failed: {e}")

async def main():
    # Create the application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    
    # Start the bot
    print("Starting bot...")
    await application.initialize()
    await application.start()
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

def load_analysis_prompt():
    prompt_file = os.path.join(os.path.dirname(__file__), 'analyses_prompt.txt')
    try:
        with open(prompt_file, 'r') as file:
            prompt = file.read()
            print(f"📄 Loaded analysis prompt from {prompt_file}")
            return prompt
    except Exception as e:
        print(f"❌ Failed to load analysis prompt: {e}")
        return None

def generate_analysis(extracted_filename, analysis_filename):
    try:
        with open(extracted_filename, 'r') as f:
            transcript = f.read()
        
        prompt_template = load_analysis_prompt()
        if not prompt_template:
            print("❌ Analysis prompt not loaded. Skipping analysis.")
            return
        prompt = f"{prompt_template}\n\nTranscript:\n{transcript}"
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        result = model.generate_content(prompt)
        if result and result.text:
            with open(analysis_filename, 'w') as f:
                f.write(result.text)
            print(f"✅ Analysis saved to {analysis_filename}")
        else:
            print("❌ Failed to generate analysis.")
    except Exception as e:
        print(f"❌ Analysis generation failed: {e}")

def run_application():
    """Run the application in a synchronous context"""
    # Create the application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    
    # Start the bot (this handles its own event loop)
    print("Starting bot...")
    application.run_polling(poll_interval=1.0, timeout=20)

if __name__ == '__main__':
    try:
        run_application()
    except KeyboardInterrupt:
        print("Bot stopped by user")