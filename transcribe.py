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
import sys
import locale

# Set up UTF-8 encoding for the entire script
sys.stdout.reconfigure(encoding='utf-8')
locale.getpreferredencoding = lambda: 'UTF-8'

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

# Define a global dictionary to store cancellation requests for each chat
cancellation_flags = {}
last_message_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Reset cancellation flag for the new session
    chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
    cancellation_flags[chat_id] = False
    keyboard = []
    for church in API_URLS.keys():
        keyboard.append([InlineKeyboardButton(church, callback_data=f"church_{church}")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.message:
        await update.message.reply_text(
            "⛪ Welkom! Selecteer een kerk:",
            reply_markup=reply_markup
        )
    else:
        await update.callback_query.message.reply_text(
            "⛪ Welkom! Selecteer een kerk:",
            reply_markup=reply_markup
        )

async def fetch_church_services(api_url: str, query: Update.callback_query):
    session = requests.Session()
    headers = {
        'Accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {authorization_token}',
    }
    params = {
        'include': 'media',
        'page': 1,
        'size': 5
    }

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

async def process_selected_service(api_url: str, service_id: str, query: Update.callback_query) -> str:
    session = requests.Session()
    headers = {
        'Accept': 'application/vnd.api+json',
        'Authorization': f'Bearer {authorization_token}',
    }

    await query.edit_message_text("🔍 Geselecteerde opname wordt opgehaald...")
    api_response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: session.get(f"{api_url}/{service_id}", headers=headers)
    )
    api_response.raise_for_status()
    data = api_response.json()
    recording = data.get('data')

    if not recording:
        return "Geen opname gevonden voor de geselecteerde dienst."

    included_media = data.get('included', [])
    attributes = recording.get('attributes', {})
    title = attributes.get('title', 'Untitled')
    start_time = attributes.get('start_at', '')

    # Extract church name from API_URLS by matching the api_url
    church_name = None
    for name, url in API_URLS.items():
        if url == api_url:
            church_name = name
            break
    
    if not church_name:
        church_name = "Unknown_Church"

    # Create church-specific folder
    church_folder = os.path.join(ROOT_FOLDER, sanitize_filename(church_name))
    if not os.path.exists(church_folder):
        os.makedirs(church_folder)

    date_str = parse_date(start_time)
    folder_name = sanitize_filename(f"{date_str}_{title}")
    recording_folder = os.path.join(church_folder, folder_name)

    # Check if analysis.txt already exists
    analysis_filename = os.path.join(recording_folder, 'analysis.txt')
    if os.path.exists(analysis_filename):
        with open(analysis_filename, 'r', encoding='utf-8') as f:
            analysis_text = f.read()
        return analysis_text

    if not os.path.exists(recording_folder):
        os.makedirs(recording_folder)

    download_url = get_download_url(recording, included_media)
    if not download_url:
        return "Geen download URL gevonden voor de opname."

    mp4_filename = os.path.join(recording_folder, f"{date_str}_{title.replace(' ', '_')}.mp4")
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')

    if not os.path.exists(mp3_filename):
        await download_file_with_progress(session, download_url, mp4_filename, query)
        if cancellation_flags[query.message.chat_id]:
            return "Bewerking geannuleerd."

        await convert_to_mp3_with_progress(mp4_filename, query)
        if cancellation_flags[query.message.chat_id]:
            return "Bewerking geannuleerd."
            
        if os.path.exists(mp4_filename):
            os.remove(mp4_filename)

    transcription_filename = mp3_filename.replace('.mp3', '_full.txt')
    extracted_filename = mp3_filename.replace('.mp3', '.txt')

    if not os.path.exists(transcription_filename):
        await query.edit_message_text("🎙️ Audio wordt getranscribeerd...")
        await asyncio.get_event_loop().run_in_executor(
            None,
            transcribe_audio,
            mp3_filename,
            transcription_filename,
            extracted_filename
        )
        if cancellation_flags[query.message.chat_id]:
            return "Bewerking geannuleerd."

    await query.edit_message_text("🤖 Analyse wordt gegenereerd...")
    await asyncio.get_event_loop().run_in_executor(
        None,
        generate_analysis,
        extracted_filename,
        analysis_filename
    )

    with open(analysis_filename, 'r', encoding='utf-8') as f:
        analysis_text = f.read()

    return analysis_text

async def handle_analysis_response(query, analysis, church, context):
    if analysis:
        max_length = 4096
        if isinstance(analysis, bytes):
            analysis = analysis.decode('utf-8')
        
        messages = [analysis[i:i+max_length] for i in range(0, len(analysis), max_length)]
        for i, message in enumerate(messages):
            if i == 0:
                await query.edit_message_text(f"📊 Analyse voor {church}:\n\n{message}")
            else:
                await context.bot.send_message(chat_id=query.message.chat_id, text=message)
        
        keyboard = [[InlineKeyboardButton("Analyseer een andere preek", callback_data="start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Wil je een andere preek analyseren?",
            reply_markup=reply_markup
        )
    else:
        await query.edit_message_text("❌ Genereren van analyse is mislukt. Probeer het opnieuw.")

async def safe_edit_message_text(query, new_text, reply_markup=None):
    message_id = query.message.message_id
    chat_id = query.message.chat_id
    current_time = time.time()
    
    last_text, last_timestamp = last_message_data.get((chat_id, message_id), ("", 0))
    
    if new_text != last_text and (current_time - last_timestamp > 1):
        await query.edit_message_text(new_text, reply_markup=reply_markup)
        last_message_data[(chat_id, message_id)] = (new_text, current_time)

async def download_file_with_progress(session, url, filename, query):
    await safe_edit_message_text(query, "⏬ Download wordt gestart met annuleeroptie...", 
                                 InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]]))
    
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
                    
                    if progress % 5 == 0:
                        update_text = f"⏬ Opname wordt gedownload... {progress}% voltooid"
                        await safe_edit_message_text(query, update_text, 
                                                     InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]]))
                
                if cancellation_flags.get(query.message.chat_id):
                    await query.edit_message_text("🚫 Download geannuleerd. Terug naar kerklijst.")
                    os.remove(filename)
                    await start(query, None)
                    return

    await safe_edit_message_text(query, "✅ Download voltooid!", InlineKeyboardMarkup([]))

async def convert_to_mp3_with_progress(mp4_filename, query):
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')
    video_clip = VideoFileClip(mp4_filename)
    audio_clip = video_clip.audio

    await safe_edit_message_text(query, "🎵 Video wordt omgezet naar audio... 50% voltooid")
    audio_clip.write_audiofile(mp3_filename, verbose=False, logger=None)

    if cancellation_flags[query.message.chat_id]:
        await query.edit_message_text("🚫 Conversie geannuleerd.")
        video_clip.close()
        audio_clip.close()
        return None

    await safe_edit_message_text(query, "🎵 Conversie voltooid! 100%")
    video_clip.close()
    audio_clip.close()
    return mp3_filename

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
            
            with open(transcription_filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Volledige Deepgram-reactie opgeslagen naar {transcription_filename}")

            transcript = ''
            channels = response_data.get('results', {}).get('channels', [])
            if channels:
                alternatives = channels[0].get('alternatives', [])
                if alternatives:
                    transcript = alternatives[0].get('transcript', '')

            if transcript:
                with open(extracted_filename, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                print(f"✅ Getranscribeerde tekst opgeslagen naar {extracted_filename}")
            else:
                print("❌ Transcript niet gevonden in de gegeven JSON-data.")
    except Exception as e:
        print(f"❌ Operatie mislukt: {e}")

def generate_analysis(extracted_filename, analysis_filename):
    try:
        with open(extracted_filename, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        prompt_template = load_analysis_prompt()
        if not prompt_template:
            print("❌ Analyseprompt niet geladen. Analyse wordt overgeslagen.")
            return
        
        prompt = f"{prompt_template}\n\nTranscript:\n{transcript}"
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        result = model.generate_content(prompt)
        
        if result and result.text:
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"✅ Analyse opgeslagen naar {analysis_filename}")
        else:
            print("❌ Genereren van analyse mislukt.")
    except Exception as e:
        print(f"❌ Genereren van analyse mislukt: {e}")

def load_analysis_prompt():
    prompt_file = os.path.join(os.path.dirname(__file__), 'analyses_prompt.txt')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()
            print(f"📄 Analyseprompt geladen uit {prompt_file}")
            return prompt
    except Exception as e:
        print(f"❌ Laden van analyseprompt mislukt: {e}")
        return None
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

def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename.replace(' ', '_')

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    
    # Answer the callback query immediately to prevent timeout issues
    await query.answer()

    # Check if "start" is triggered from the "Analyseer een andere preek" button
    if query.data == "start":
        await start(update, context)

    elif query.data.startswith("church_"):
        church = query.data.replace("church_", "")
        await query.edit_message_text(
            f"🔄 Beschikbare preken van {church} worden opgehaald...",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        
        try:
            services_by_date = await fetch_church_services(API_URLS[church], query)
            if services_by_date:
                keyboard = [[InlineKeyboardButton(date, callback_data=f"date_{church}_{date}")]
                            for date in services_by_date.keys()]
                keyboard.append([InlineKeyboardButton("Annuleren", callback_data="cancel")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"📅 Selecteer een datum voor de preek van {church}:",
                    reply_markup=reply_markup
                )
            else:
                await query.edit_message_text("❌ Geen preken gevonden voor deze kerk.")
        except Exception as e:
            await query.edit_message_text(f"❌ Er is een fout opgetreden: {str(e)}")

    elif query.data.startswith("date_"):
        _, church, date = query.data.split("_", 2)
        services_by_date = await fetch_church_services(API_URLS[church], query)
        services = services_by_date.get(date, [])

        if services:
            if len(services) == 1:
                service_id = services[0]["id"]
                cancellation_flags[query.message.chat_id] = False
                await query.edit_message_text(
                    f"🔄 Geselecteerde preek van {church} wordt verwerkt...",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
                )

                try:
                    analysis = await process_selected_service(API_URLS[church], service_id, query)
                    if cancellation_flags[query.message.chat_id]:
                        await query.edit_message_text("🚫 Verwerking geannuleerd.")
                        return
                    await handle_analysis_response(query, analysis, church, context)
                except Exception as e:
                    await query.edit_message_text(f"❌ Er is een fout opgetreden: {str(e)}")
            else:
                keyboard = [[InlineKeyboardButton(service["title"], callback_data=f"service_{church}_{service['id']}")]
                            for service in services]
                keyboard.append([InlineKeyboardButton("Annuleren", callback_data="cancel")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    f"🕒 Selecteer de specifieke dienst voor {date}:",
                    reply_markup=reply_markup
                )
        else:
            await query.edit_message_text("❌ Geen preken gevonden voor deze datum.")

    elif query.data.startswith("service_"):
        _, church, service_id = query.data.split("_", 2)
        cancellation_flags[query.message.chat_id] = False
        await query.edit_message_text(
            f"🔄 Geselecteerde preek van {church} wordt verwerkt...",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )

        try:
            analysis = await process_selected_service(API_URLS[church], service_id, query)
            if cancellation_flags[query.message.chat_id]:
                await query.edit_message_text("🚫 Verwerking geannuleerd.")
                return
            await handle_analysis_response(query, analysis, church, context)
        except Exception as e:
            await query.edit_message_text(f"❌ Er is een fout opgetreden: {str(e)}")

    elif query.data == "cancel":
        cancellation_flags[query.message.chat_id] = True
        await query.edit_message_text("🔙 Bewerking wordt geannuleerd... Terug naar start.")
        await start(update, context)

async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    
    print("Bot wordt gestart...")
    await application.initialize()
    await application.start()
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

def run_application():
    """Run the application in a synchronous context"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    
    print("Bot wordt gestart...")
    application.run_polling(poll_interval=1.0, timeout=20)

if __name__ == '__main__':
    try:
        run_application()
    except KeyboardInterrupt:
        print("Bot gestopt door gebruiker")