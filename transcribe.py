import asyncio
import os
import requests
import json
from datetime import datetime
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
API_URLS = {
    "Wijngaarden": "https://api.kerkdienstgemist.nl/api/v2/stations/1306/recordings",
    "Nieuwe Kerk Utrecht": "https://api.kerkdienstgemist.nl/api/v2/stations/1341/recordings"
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(church, callback_data=f"church_{church}") for church in API_URLS.keys()]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🏛️ Welcome! Please select a church to analyze the latest sermon:",
        reply_markup=reply_markup
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data.startswith("church_"):
        church = query.data.replace("church_", "")
        await query.edit_message_text(f"🔄 Processing latest sermon from {church}...")

        try:
            analysis = await process_church(API_URLS[church], church, query)
            await handle_analysis_response(query, analysis, church, context)
        except Exception as e:
            await query.edit_message_text(f"❌ An error occurred: {str(e)}")

    elif query.data == "start":
        await start(update, context)

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
async def process_church(api_url: str, church_name: str, query: Update.callback_query) -> str:
    session = requests.Session()
    headers = {
        'Accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {authorization_token}',
    }
    params = {
        'include': 'media',
        'page': 1,
        'size': 1
    }

    await query.edit_message_text(f"🔍 Fetching latest recording from {church_name}...")

    # Run blocking operations in a thread pool
    api_response = await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: session.get(api_url, headers=headers, params=params)
    )
    api_response.raise_for_status()
    data = api_response.json()
    recordings = data.get('data', [])
    
    if not recordings:
        return "No recordings found."

    recording = recordings[0]
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

    await query.edit_message_text(f"⬇️ Downloading recording from {church_name}...")
    mp4_filename = os.path.join(recording_folder, f"{date_str}_{title.replace(' ', '_')}.mp4")
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')

    if not os.path.exists(mp4_filename):
        await asyncio.get_event_loop().run_in_executor(
            None,
            download_file,
            session,
            download_url,
            mp4_filename
        )

    if not os.path.exists(mp3_filename):
        await query.edit_message_text("🎵 Converting video to audio...")
        await asyncio.get_event_loop().run_in_executor(
            None,
            convert_to_mp3,
            mp4_filename
        )

    transcription_filename = mp3_filename.replace('.mp3', '_full.txt')
    extracted_filename = mp3_filename.replace('.mp3', '.txt')

    if not os.path.exists(transcription_filename):
        await query.edit_message_text("🎙️ Transcribing audio...")
        await asyncio.get_event_loop().run_in_executor(
            None,
            transcribe_audio,
            mp3_filename,
            transcription_filename,
            extracted_filename
        )

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

def download_file(session, url, filename):
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def convert_to_mp3(mp4_filename):
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')
    video_clip = VideoFileClip(mp4_filename)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(mp3_filename)
    audio_clip.close()
    video_clip.close()
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