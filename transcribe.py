from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
import asyncio
from typing import Tuple
import logging

from sendgrid.helpers.mail import (
    Mail, 
    Email, 
    To, 
    Content, 
    HtmlContent
)
import os
import requests
import json
from datetime import datetime
import time
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, ApplicationBuilder, MessageHandler, filters
from deepgram import DeepgramClient, PrerecordedOptions
import sys
import locale
from telegram import InputFile

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

class ChurchManager:
    def __init__(self):
        self.churches_file = 'churches.json'
        self.base_url = "https://api.kerkdienstgemist.nl/api/v2/stations/{}/recordings"
        self.load_churches()

    def load_churches(self):
        if os.path.exists(self.churches_file):
            with open(self.churches_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.churches = data.get('churches', {})
        else:
            # Default churches if file doesn't exist
            self.churches = {
                "Wijngaarden": "1306",
                "Nieuwe Kerk Utrecht": "1341",
                "NGK Doorn": "2281",
                "Elimkerk Hendrik-Ido-Ambacht": "876"
            }
            self.save_churches()

    def save_churches(self):
        with open(self.churches_file, 'w', encoding='utf-8') as f:
            json.dump({'churches': self.churches}, f, indent=4, ensure_ascii=False)

    def add_church(self, name: str, church_id: str) -> bool:
        if name in self.churches:
            return False
        self.churches[name] = church_id
        self.save_churches()
        return True

    def get_api_urls(self):
        return {name: self.base_url.format(church_id) 
                for name, church_id in self.churches.items()}

# Dictionary to store cancellation flags for each chat
cancellation_flags = {}
last_message_data = {}

class EmailService:
    def __init__(self, api_key: str, default_from_domain: str):
        """Initialize EmailService with SendGrid API key and default from domain."""
        self.api_key = api_key
        self.default_from_domain = default_from_domain
        self.client = SendGridAPIClient(api_key)
        self.logger = logging.getLogger(__name__)

    async def send_report(
        self,
        html_content: str,
        recipient_email: str,
        church_name: str,
        date: str
    ) -> Tuple[bool, str]:
        """
        Asynchronously send an HTML report via email using SendGrid.
        
        Args:
            html_content: The HTML content of the report
            recipient_email: The recipient's email address
            church_name: Name of the church for the subject line
            date: Date of the service
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Create message object
            message = Mail(
                from_email=f'no-reply@{self.default_from_domain}',
                to_emails=recipient_email,
                subject=f'Preekanalyse - {church_name} - {date}',
                html_content=html_content
            )

            # Send email in the executor to prevent blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                message
            )

            return True, "Email succesvol verzonden"

        except Exception as e:
            error_msg = f"Email verzenden mislukt: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _send_email(self, message: Mail) -> None:
        """
        Execute the actual email sending operation.
        This runs in an executor to prevent blocking.
        """
        try:
            response = self.client.send(message)
            if response.status_code not in (200, 201, 202):
                raise Exception(f"SendGrid returned status code {response.status_code}")
        except Exception as e:
            self.logger.error(f"SendGrid API error: {str(e)}", exc_info=True)
            raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
    cancellation_flags[chat_id] = False
    
    # Initialize page number if not set
    if not context.user_data.get('page'):
        context.user_data['page'] = 1
    
    church_manager = ChurchManager()
    reply_markup = await get_paginated_keyboard(church_manager.churches, context.user_data['page'])
    
    if update.message:
        await update.message.reply_text(
            "⛪ Welkom! Selecteer een kerk of voeg een nieuwe toe:",
            reply_markup=reply_markup
        )
    else:
        await update.callback_query.message.reply_text(
            "⛪ Welkom! Selecteer een kerk of voeg een nieuwe toe:",
            reply_markup=reply_markup
        )

async def handle_add_church(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get('adding_church'):
        # Start the process of adding a new church
        context.user_data['adding_church'] = True
        
        # Post the image to show users where to get the ID
        with open('NieuweKerkToevoegen.png', 'rb') as image_file:
            await update.callback_query.message.reply_photo(
                photo=InputFile(image_file),
                caption="🔢 Voer het Kerkdienstgemist kerk ID in. Zie de afbeelding hierboven voor hulp bij het vinden van de ID.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
            )
    return

async def handle_church_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get('adding_church'):
        return
    
    message_text = update.message.text
    
    if not context.user_data.get('church_id'):
        # Process church ID
        if message_text.isdigit():
            context.user_data['church_id'] = message_text
            await update.message.reply_text(
                "✍️ Geef deze kerk een naam:",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
            )
        else:
            await update.message.reply_text(
                "❌ Voer een geldig nummer in als kerk ID.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
            )
    else:
        # Process church name
        church_manager = ChurchManager()
        success = church_manager.add_church(message_text, context.user_data['church_id'])
        
        if success:
            await update.message.reply_text(f"✅ Kerk '{message_text}' is succesvol toegevoegd!")
        else:
            await update.message.reply_text("❌ Deze kerknaam bestaat al. Kies een andere naam.")
        
        # Reset status and return to main menu
        context.user_data.clear()
        await start(update, context)

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

    await query.edit_message_text(
        "🔍 Geselecteerde opname wordt opgehaald...",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )
    
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
    church_manager = ChurchManager()
    api_urls = church_manager.get_api_urls()
    for name, url in api_urls.items():
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
    sanitized_title = sanitize_filename(title)
    folder_name = f"{date_str}_{sanitized_title}"
    recording_folder = os.path.join(church_folder, folder_name)

    # Check if analysis already exists
    analysis_filename = os.path.join(recording_folder, 'analysis.txt')
    html_report_filename = os.path.join(recording_folder, 'report.html')
    
    if os.path.exists(analysis_filename) and os.path.exists(html_report_filename):
        # Store the report path in context
        if hasattr(query.message, '_bot') and hasattr(query.message._bot, 'context'):
            query.message._bot.context.user_data['last_report_path'] = html_report_filename
            query.message._bot.context.user_data['last_report_date'] = date_str
            query.message._bot.context.user_data['last_church_name'] = church_name
        
        with open(analysis_filename, 'r', encoding='utf-8') as f:
            analysis_text = f.read()
        return analysis_text

    if not os.path.exists(recording_folder):
        os.makedirs(recording_folder)

    download_url, file_type = get_download_url(recording, included_media)
    if not download_url:
        return "Geen download URL gevonden voor de opname."

    # Define filenames based on the file type
    base_filename = f"{date_str}_{sanitized_title}"
    if file_type == 'mp4':
        downloaded_filename = os.path.join(recording_folder, f"{base_filename}.mp4")
        final_mp3_filename = os.path.join(recording_folder, f"{base_filename}.mp3")
    else:  # mp3
        downloaded_filename = os.path.join(recording_folder, f"{base_filename}.mp3")
        final_mp3_filename = downloaded_filename

    if not os.path.exists(final_mp3_filename):
        # Download phase
        await download_file_with_progress(session, download_url, downloaded_filename, query)
        if cancellation_flags[query.message.chat_id]:
            return "Bewerking geannuleerd."

        await asyncio.sleep(1)
        
        # Convert if it's an MP4 file
        if file_type == 'mp4':
            await safe_edit_message_text(
                query,
                "🎥 Videoverwerking wordt gestart...",
                InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
            )
            
            await asyncio.sleep(1)
            
            await convert_to_mp3_with_progress(downloaded_filename, query)
            if cancellation_flags[query.message.chat_id]:
                return "Bewerking geannuleerd."
                
            await asyncio.sleep(1)

            # Clean up MP4 file after conversion
            if os.path.exists(downloaded_filename):
                os.remove(downloaded_filename)

    transcription_filename = os.path.join(recording_folder, f"{base_filename}_full.txt")
    extracted_filename = os.path.join(recording_folder, f"{base_filename}.txt")

    await safe_edit_message_text(
        query,
        "🎙️ Voorbereiding transcriptie...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )
    
    await asyncio.sleep(1)

    if not os.path.exists(transcription_filename):
        await safe_edit_message_text(
            query,
            "🎙️ Audio wordt getranscribeerd...",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            transcribe_audio,
            final_mp3_filename,
            transcription_filename,
            extracted_filename
        )
        
        if cancellation_flags[query.message.chat_id]:
            return "Bewerking geannuleerd."

    # Generate Analysis
    await safe_edit_message_text(
        query,
        "🤖 Voorbereiding analyse...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )
    
    await asyncio.sleep(1)

    await safe_edit_message_text(
        query,
        "🤖 Analyse wordt gegenereerd...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )

    await asyncio.get_event_loop().run_in_executor(
        None,
        generate_analysis,
        extracted_filename,
        analysis_filename,
        church_name,
        date_str
    )

    if cancellation_flags[query.message.chat_id]:
        return "Bewerking geannuleerd."

    # Read the generated analysis
    try:
        with open(analysis_filename, 'r', encoding='utf-8') as f:
            analysis_content = f.read()
    except Exception as e:
        print(f"Error reading analysis file: {e}")
        return "Er is een fout opgetreden bij het lezen van de analyse."

    # Generate HTML Report with analysis content
    await safe_edit_message_text(
        query,
        "📊 HTML rapport wordt gegenereerd...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )

    await asyncio.get_event_loop().run_in_executor(
        None,
        generate_html_report,
        html_report_filename,
        church_name,
        date_str,
        analysis_content
    )

    if cancellation_flags[query.message.chat_id]:
        return "Bewerking geannuleerd."

    # Store the report path in context
    if hasattr(query.message, '_bot') and hasattr(query.message._bot, 'context'):
        query.message._bot.context.user_data['last_report_path'] = html_report_filename
        query.message._bot.context.user_data['last_report_date'] = date_str
        query.message._bot.context.user_data['last_church_name'] = church_name

    try:
        with open(analysis_filename, 'r', encoding='utf-8') as f:
            analysis_text = f.read()
        return analysis_text
    except Exception as e:
        print(f"Error reading analysis file: {e}")
        return "Er is een fout opgetreden bij het lezen van de analyse."

def clean_html_report(file_path):
    """
    Cleans the HTML report file by removing the ```html at the start and the closing ```
    at the end of the file, leaving valid HTML tags intact.
    Args:
        file_path (str): Path to the HTML file to clean.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Strip ```html from the first line and ``` from the last line
        if lines:
            if lines[0].strip() == "```html":
                lines = lines[1:]  # Remove the first line
            if lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove the last line
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            print(f"✅ HTML report cleaned and saved: {file_path}")
        else:
            print("❌ HTML report is empty.")
    except Exception as e:
        print(f"❌ Error cleaning HTML report: {e}")



def generate_html_report(html_report_filename, church_name, date, analysis_content):
    try:
        prompt_template = load_html_report_prompt()
        if not prompt_template:
            print("❌ HTML report prompt not loaded. Skipping HTML report generation.")
            return
        
        # Create the dynamic prompt with the church name, date, and analysis only
        prompt = f"Church: {church_name}\nDate: {date}\n\nAnalysis Content:\n{analysis_content}\n\n{prompt_template}"
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        result = model.generate_content(prompt)
        
        if result and result.text:
            with open(html_report_filename, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"✅ HTML report saved to {html_report_filename}")
            
            # Clean the generated HTML report
            clean_html_report(html_report_filename)
        else:
            print("❌ Failed to generate HTML report.")
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")


# Add function to load HTML report prompt
def load_html_report_prompt():
    prompt_file = os.path.join(os.path.dirname(__file__), 'html_report_prompt.txt')
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            prompt = file.read()
            print(f"📄 HTML rapport prompt geladen uit {prompt_file}")
            return prompt
    except Exception as e:
        print(f"❌ Laden van HTML rapport prompt mislukt: {e}")
        return None
    

class ReportManager:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder

    def get_report_path(self, church_name: str, target_date: str) -> tuple[str | None, str | None]:
        """
        Get the path to a report file for a specific church and date.
        
        Args:
            church_name (str): Name of the church
            target_date (str): Date in YYYY-MM-DD format
            
        Returns:
            tuple: (report_path, folder_name) or (None, None) if not found
        """
        try:
            # Sanitize church name for filesystem
            safe_church_name = self._sanitize_name(church_name)
            church_folder = os.path.join(self.root_folder, safe_church_name)
            
            if not os.path.exists(church_folder):
                print(f"Church folder not found: {church_folder}")
                return None, None

            # Look for exact date match in folder names
            for folder_name in os.listdir(church_folder):
                folder_path = os.path.join(church_folder, folder_name)
                
                # Skip if not a directory
                if not os.path.isdir(folder_path):
                    continue
                
                # Extract date from folder name and compare with target date
                folder_date = folder_name.split('_')[0]
                if folder_date == target_date:
                    # Check for report file
                    report_path = os.path.join(folder_path, 'report.html')
                    if os.path.exists(report_path):
                        print(f"Found report at: {report_path}")
                        return report_path, folder_name
            
            print(f"No report found for {church_name} on {target_date}")
            return None, None
            
        except Exception as e:
            print(f"Error finding report: {str(e)}")
            return None, None

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for filesystem use"""
        # Replace biblical reference characters and invalid filename chars
        name = name.replace(':', '_').replace('/', '_')
        invalid_chars = r'<>"\|?*'
        for char in invalid_chars:
            name = name.replace(char, '')
        
        # Replace spaces and clean up underscores
        name = name.replace(' ', '_')
        while '__' in name:
            name = name.replace('__', '_')
        return name.strip('_')

# Update the send_html_report function to use the new EmailService
async def request_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Request email address from user.
    
    Args:
        update (Update): The update object
        context (ContextTypes.DEFAULT_TYPE): The context object
    """
    # Store the report details in context for later use
    parts = update.callback_query.data.split('_')
    context.user_data['pending_report'] = {
        'church_name': '_'.join(parts[2:-1]),
        'date': parts[-1]
    }
    
    keyboard = [
        [InlineKeyboardButton("❌ Annuleren", callback_data="cancel_email")],
        [InlineKeyboardButton("📧 Gebruik standaard email", callback_data="use_default_email")]
    ]
    
    await update.callback_query.edit_message_text(
        "📧 Voer het email adres in waar het rapport naartoe gestuurd moet worden:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    
    # Set state to wait for email input
    context.user_data['awaiting_email'] = True

async def handle_email_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle email input from user.
    
    Args:
        update (Update): The update object
        context (ContextTypes.DEFAULT_TYPE): The context object
    """
    if not context.user_data.get('awaiting_email'):
        return
    
    email = update.message.text.strip()
    # Basic email validation
    if '@' not in email or '.' not in email:
        await update.message.reply_text(
            "❌ Ongeldig email adres. Probeer opnieuw of kies een andere optie:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Annuleren", callback_data="cancel_email")],
                [InlineKeyboardButton("📧 Gebruik standaard email", callback_data="use_default_email")]
            ])
        )
        return
    
    # Clear awaiting state
    context.user_data['awaiting_email'] = False
    
    # Get stored report details
    report_details = context.user_data.get('pending_report', {})
    if not report_details:
        await update.message.reply_text(
            "❌ Rapport details niet gevonden. Start opnieuw.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔄 Start opnieuw", callback_data="start")
            ]])
        )
        return
    
    # Send report
    await send_report_to_email(
        update,
        context,
        report_details['church_name'],
        report_details['date'],
        email
    )

async def send_report_to_email(update, context, church_name: str, date: str, email_recipient: str):
    """
    Send the report to the specified email address.
    
    Args:
        update: The update object
        context: The context object
        church_name (str): Name of the church
        date (str): Date of the service
        email_recipient (str): Email address to send the report to
    """
    email_domain = os.getenv('EMAIL_DOMAIN')
    if not email_domain:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⚠️ Email configuratie ontbreekt. Neem contact op met de beheerder.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔄 Start opnieuw", callback_data="start")
            ]])
        )
        return

    report_manager = ReportManager(ROOT_FOLDER)
    html_report_filename, _ = report_manager.get_report_path(church_name, date)

    if not html_report_filename:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❌ Rapport niet gevonden. Kies een andere optie:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Analyseer andere preek", callback_data="start")],
                [InlineKeyboardButton("❌ Terug naar kerkenlijst", callback_data="start")]
            ])
        )
        return

    # Send the email
    asyncio.create_task(send_html_report(
        html_report_filename,
        church_name,
        date,
        email_recipient,
        email_domain
    ))

    # Show options to continue
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"📧 Rapport wordt verzonden naar {email_recipient}. Wat wil je nu doen?",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Analyseer andere preek", callback_data="start")],
            [InlineKeyboardButton("❌ Terug naar kerkenlijst", callback_data="start")]
        ])
    )
    
# Add keyboard helper function
def get_report_options_keyboard(church_name: str, date: str = None) -> InlineKeyboardMarkup:
    """
    Creates keyboard with report options
    
    Args:
        church_name (str): Name of the church
        date (str): Date of the service in YYYY-MM-DD format
    """
    callback_data = f"email_report_{church_name}_{date}" if date else f"email_report_{church_name}"
    keyboard = [
        [InlineKeyboardButton("📧 Verstuur rapport per email", callback_data=callback_data)],
        [InlineKeyboardButton("📊 Analyseer een andere preek", callback_data="start")],
        [InlineKeyboardButton("❌ Afsluiten", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

def chunk_message(text: str, max_length: int = 4096) -> list:
    """
    Split message into chunks that respect Telegram's limits.
    Tries to split on paragraph boundaries when possible.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        # If this paragraph alone is longer than max length, split it
        if len(paragraph) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Split long paragraph into chunks
            while paragraph:
                if len(paragraph) > max_length:
                    # Try to split on a sentence boundary
                    split_point = paragraph[:max_length].rfind('. ') + 1
                    if split_point <= 0:  # No sentence boundary found
                        split_point = max_length
                    chunks.append(paragraph[:split_point])
                    paragraph = paragraph[split_point:].lstrip()
                else:
                    current_chunk = paragraph
                    paragraph = ""
        
        # If adding this paragraph would exceed max length
        elif len(current_chunk) + len(paragraph) + 2 > max_length:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += '\n\n'
            current_chunk += paragraph

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def escape_markdown_v2_text(text: str) -> str:
    """
    Properly escape MarkdownV2 characters while preserving intended formatting.
    First replace style markers with temporary placeholders, then escape special chars,
    then restore style markers with proper MarkdownV2 syntax.
    """
    # Step 1: Replace style markers with unique placeholders
    formatted_text = text
    formatted_text = formatted_text.replace('**', '§BOLD§')  # Replace double asterisks with placeholder
    formatted_text = formatted_text.replace('_', '§ITALIC§')  # Replace underscores with placeholder

    # Step 2: Escape special characters
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        formatted_text = formatted_text.replace(char, f'\\{char}')

    # Step 3: Restore style markers with proper MarkdownV2 syntax
    formatted_text = formatted_text.replace('§BOLD§', '*')      # Replace bold placeholder with single asterisk
    formatted_text = formatted_text.replace('§ITALIC§', '_')    # Restore italic placeholder

    return formatted_text



async def handle_analysis_response(query, analysis, church, context):
    """
    Handles the response after analysis is complete, including displaying the analysis
    and providing options for further actions.
    """
    if not analysis:
        await query.edit_message_text(
            "❌ Genereren van analyse is mislukt. Probeer opnieuw.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
        )
        return

    try:
        # Use the date stored in context instead of looking for the newest folder
        current_date = context.user_data.get('current_service_date')
        
        if not current_date:
            # Fallback to finding the date from folder name only if not in context
            church_folder = os.path.join(ROOT_FOLDER, sanitize_filename(church))
            if os.path.exists(church_folder):
                folders = [f for f in os.listdir(church_folder) if os.path.isdir(os.path.join(church_folder, f))]
                if folders:
                    newest_folder = max(folders, key=lambda x: os.path.getctime(os.path.join(church_folder, x)))
                    current_date = newest_folder.split('_')[0]

        # Convert to string if not already
        if isinstance(analysis, bytes):
            analysis = analysis.decode('utf-8')
        elif not isinstance(analysis, str):
            analysis = str(analysis)

        # Split into chunks
        chunks = chunk_message(analysis)
        
        try:
            # Try with markdown formatting
            escaped_church = escape_markdown_v2_text(church)
            escaped_chunks = [escape_markdown_v2_text(chunk) for chunk in chunks]
            
            # Send first chunk with header
            first_message = f"📊 Analyse voor {escaped_church}:\n\n{escaped_chunks[0]}"
            await query.edit_message_text(
                text=first_message,
                parse_mode='MarkdownV2'
            )
            
            # Send remaining chunks
            for chunk in escaped_chunks[1:]:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=chunk,
                    parse_mode='MarkdownV2'
                )

        except Exception as markdown_error:
            print(f"Markdown formatting failed: {markdown_error}. Falling back to plain text.")
            # Fallback to plain text if markdown fails
            first_message = f"📊 Analyse voor {church}:\n\n{chunks[0]}"
            await query.edit_message_text(text=first_message)
            
            for chunk in chunks[1:]:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=chunk
                )
        
        # Send options message with the correct date
        if current_date:
            print(f"Using date for report options: {current_date}")  # Debug log
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="Wat wil je nu doen?",
                reply_markup=get_report_options_keyboard(church, current_date)
            )
        else:
            print("No date found for report options")  # Debug log
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="❌ Kan datum niet bepalen. Probeer opnieuw.",
                reply_markup=get_report_options_keyboard(church)
            )
        
    except Exception as e:
        print(f"Error in handle_analysis_response: {str(e)}")
        await query.edit_message_text(
            "❌ Er is een fout opgetreden bij het tonen van de analyse.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
        )
async def safe_edit_message_text(query, new_text, reply_markup=None):
    """
    Safely edit message text, handling long messages by splitting them and preserving markdown.
    """
    message_id = query.message.message_id
    chat_id = query.message.chat_id
    current_time = time.time()
    
    last_text, last_timestamp = last_message_data.get((chat_id, message_id), ("", 0))
    
    if new_text != last_text and (current_time - last_timestamp > 1):
        try:
            # Split message if too long
            chunks = chunk_message(new_text)
            
            try:
                # Try with markdown formatting
                escaped_chunks = [escape_markdown_v2_text(chunk) for chunk in chunks]
                
                # Update first chunk
                await query.edit_message_text(
                    text=escaped_chunks[0],
                    reply_markup=reply_markup,
                    parse_mode='MarkdownV2'
                )
                
                # Send additional chunks if any
                for chunk in escaped_chunks[1:]:
                    await query.message.reply_text(
                        text=chunk,
                        parse_mode='MarkdownV2'
                    )
                
            except Exception as markdown_error:
                print(f"Markdown formatting failed: {markdown_error}. Falling back to plain text.")
                # Fallback to plain text if markdown fails
                await query.edit_message_text(
                    text=chunks[0],
                    reply_markup=reply_markup
                )
                
                for chunk in chunks[1:]:
                    await query.message.reply_text(text=chunk)
            
            last_message_data[(chat_id, message_id)] = (new_text, current_time)
            
        except Exception as e:
            print(f"Error in safe_edit_message_text: {str(e)}")
            # Try to send a simple error message
            await query.edit_message_text(
                text="Er is een fout opgetreden bij het bijwerken van het bericht.",
                reply_markup=reply_markup
            )
async def download_file_with_progress(session, url, filename, query):
    """
    Downloads a file with progress updates and cancellation option.
    Shows progress every 5% during download.
    """
    await safe_edit_message_text(
        query,
        "⏬ Download wordt gestart...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )
    
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0))
        downloaded = 0
        last_percentage = -1  # Track last shown percentage

        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = int((downloaded / total_length) * 100)
                    
                    # Update message every 5% progress
                    if progress % 5 == 0 and progress != last_percentage:
                        last_percentage = progress
                        update_text = f"⏬ Opname wordt gedownload... {progress}% voltooid"
                        await safe_edit_message_text(
                            query,
                            update_text,
                            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
                        )
                
                if cancellation_flags.get(query.message.chat_id):
                    await query.edit_message_text("🚫 Download geannuleerd. Terug naar kerklijst.")
                    os.remove(filename)
                    await start(query, None)
                    return

    # Show final completion message
    await safe_edit_message_text(
        query,
        "✅ Download voltooid! Conversie wordt gestart...",
        InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
    )

async def convert_to_mp3_with_progress(mp4_filename, query):
    """
    Converts MP4 to MP3 with progress updates and cancellation option.
    Shows multiple stages of conversion process.
    """
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')
    
    try:
        # Load video file
        await safe_edit_message_text(
            query,
            "🎥 Video wordt geladen... (25% voltooid)",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        video_clip = VideoFileClip(mp4_filename)
        
        if cancellation_flags.get(query.message.chat_id):
            video_clip.close()
            return None

        # Extract audio
        await safe_edit_message_text(
            query,
            "🎵 Audio wordt geëxtraheerd... (50% voltooid)",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        audio_clip = video_clip.audio
        
        if cancellation_flags.get(query.message.chat_id):
            video_clip.close()
            audio_clip.close()
            return None

        # Write audio file
        await safe_edit_message_text(
            query,
            "💾 Audio wordt opgeslagen... (75% voltooid)",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        audio_clip.write_audiofile(mp3_filename, verbose=False, logger=None)

        if cancellation_flags.get(query.message.chat_id):
            video_clip.close()
            audio_clip.close()
            if os.path.exists(mp3_filename):
                os.remove(mp3_filename)
            return None

        # Cleanup and completion
        await safe_edit_message_text(
            query,
            "🎵 Conversie voltooid! (100% voltooid)",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        
        video_clip.close()
        audio_clip.close()
        return mp3_filename

    except Exception as e:
        await safe_edit_message_text(
            query,
            f"❌ Fout tijdens conversie: {str(e)}",
            InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        if 'video_clip' in locals():
            video_clip.close()
        if 'audio_clip' in locals():
            audio_clip.close()
        return None

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

def generate_analysis(extracted_filename, analysis_filename, church_name, date):
    try:
        with open(extracted_filename, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        prompt_template = load_analysis_prompt()
        if not prompt_template:
            print("❌ Analyseprompt niet geladen. Analyse wordt overgeslagen.")
            return
        
        # Create the dynamic prompt with the church name and date
        prompt = f"Church: {church_name}\nDate: {date}\n\n{prompt_template}\n\nTranscript:\n{transcript}"
        
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
    """
    Get download URL and file type from the recording data.
    Returns tuple of (url, file_type) where file_type is either 'mp3' or 'mp4'
    """
    relationships = recording.get('relationships', {})
    media_data = relationships.get('media', {}).get('data', [])
    media_ids = {m['id'] for m in media_data}

    for media_item in included_media:
        if media_item['id'] in media_ids:
            attributes = media_item.get('attributes', {})
            if media_item['type'] == 'audio_files':
                return attributes.get('download_url', ''), 'mp3'
            elif media_item['type'] == 'video_files':
                return attributes.get('download_url', ''), 'mp4'
    return None, None

def sanitize_filename(filename):
    """
    Sanitize filename by:
    1. Replacing biblical reference characters (: and /) with proper alternatives
    2. Removing other invalid filename characters
    3. Replacing spaces with underscores
    """
    # First handle biblical references specifically
    filename = filename.replace(':', '_').replace('/', '_')
    
    # Then handle all other invalid characters
    invalid_chars = r'<>"\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
        
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove multiple consecutive underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
        
    # Remove any leading/trailing underscores
    filename = filename.strip('_')
    
    return filename

async def get_paginated_keyboard(churches: dict, page: int = 1, churches_per_page: int = 5) -> InlineKeyboardMarkup:
    """
    Create a paginated keyboard for church selection.
    
    Args:
        churches (dict): Dictionary of churches
        page (int): Current page number (1-based)
        churches_per_page (int): Number of churches to show per page
    
    Returns:
        InlineKeyboardMarkup: Keyboard markup with pagination
    """
    church_items = list(churches.items())
    total_pages = (len(church_items) + churches_per_page - 1) // churches_per_page
    start_idx = (page - 1) * churches_per_page
    end_idx = start_idx + churches_per_page
    
    # Create church buttons for current page
    keyboard = []
    for name, church_id in church_items[start_idx:end_idx]:
        keyboard.append([InlineKeyboardButton(name, callback_data=f"church_{name}")])
    
    # Add navigation buttons if needed
    nav_buttons = []
    if total_pages > 1:
        if page > 1:
            nav_buttons.append(InlineKeyboardButton("⬅️ Vorige", callback_data=f"page_{page-1}"))
        if page < total_pages:
            nav_buttons.append(InlineKeyboardButton("Volgende ➡️", callback_data=f"page_{page+1}"))
    
    if nav_buttons:
        keyboard.append(nav_buttons)
    
    # Add the "Add new church" button at the bottom
    keyboard.append([InlineKeyboardButton("➕ Nieuwe kerk toevoegen", callback_data="add_church")])
    
    return InlineKeyboardMarkup(keyboard)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button interactions in the Telegram bot."""
    query = update.callback_query
    await query.answer()

    # Handle pagination
    if query.data.startswith("page_"):
        page = int(query.data.split("_")[1])
        context.user_data['page'] = page
        church_manager = ChurchManager()
        reply_markup = await get_paginated_keyboard(church_manager.churches, page)
        await query.edit_message_text(
            "⛪ Welkom! Selecteer een kerk of voeg een nieuwe toe:",
            reply_markup=reply_markup
        )
        return

    # Handle add church request
    if query.data == "add_church":
        await handle_add_church(update, context)
        
    # Handle start request
    elif query.data == "start":
        context.user_data['page'] = 1  # Reset page when starting over
        await start(update, context)
        
    # Handle cancel request
    elif query.data == "cancel":
        context.user_data.clear()
        cancellation_flags[query.message.chat_id] = True
        await query.edit_message_text("🔙 Terug naar start...")
        await start(update, context)

    # Handle email-related actions
    elif query.data.startswith("email_report_"):
        await request_email(update, context)
    
    elif query.data == "cancel_email":
        context.user_data.clear()
        keyboard = [
            [InlineKeyboardButton("📊 Analyseer andere preek", callback_data="start")],
            [InlineKeyboardButton("❌ Terug naar kerkenlijst", callback_data="start")]
        ]
        await query.edit_message_text(
            "❌ Email verzending geannuleerd. Wat wil je nu doen?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif query.data == "use_default_email":
        email_recipient = os.getenv('EMAIL_RECIPIENT')
        if not email_recipient:
            keyboard = [
                [InlineKeyboardButton("📊 Analyseer andere preek", callback_data="start")],
                [InlineKeyboardButton("❌ Terug naar kerkenlijst", callback_data="start")]
            ]
            await query.edit_message_text(
                "⚠️ Standaard email niet geconfigureerd. Kies een andere optie:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
            
        report_details = context.user_data.get('pending_report', {})
        if report_details:
            await send_report_to_email(
                update,
                context,
                report_details['church_name'],
                report_details['date'],
                email_recipient
            )
        else:
            await query.edit_message_text(
                "❌ Rapport details niet gevonden. Start opnieuw.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔄 Start opnieuw", callback_data="start")
                ]])
            )

    # Handle church selection
    elif query.data.startswith("church_"):
        church = query.data.replace("church_", "")
        await query.edit_message_text(
            f"🔄 Beschikbare preken van {church} worden opgehaald...",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )
        
        try:
            church_manager = ChurchManager()
            api_urls = church_manager.get_api_urls()
            services_by_date = await fetch_church_services(api_urls[church], query)
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
                await query.edit_message_text(
                    "❌ Geen preken gevonden voor deze kerk.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
                )
        except Exception as e:
            await query.edit_message_text(
                "❌ Er is een fout opgetreden bij het ophalen van de preken.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
            )

    # Handle date selection
    elif query.data.startswith("date_"):
        _, church, date = query.data.split("_", 2)
        church_manager = ChurchManager()
        api_urls = church_manager.get_api_urls()
        services_by_date = await fetch_church_services(api_urls[church], query)
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
                    analysis = await process_selected_service(api_urls[church], service_id, query)
                    if cancellation_flags[query.message.chat_id]:
                        await query.edit_message_text(
                            "🚫 Verwerking geannuleerd.",
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
                        )
                        return

                    context.user_data['current_service_date'] = date
                    await handle_analysis_response(query, analysis, church, context)
                except Exception as e:
                    await query.edit_message_text(
                        "❌ Er is een fout opgetreden bij het verwerken van de preek.",
                        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
                    )
            else:
                keyboard = [[InlineKeyboardButton(service["title"], 
                            callback_data=f"service_{church}_{service['id']}_{date}")]
                            for service in services]
                keyboard.append([InlineKeyboardButton("Annuleren", callback_data="cancel")])
                await query.edit_message_text(
                    f"🕒 Selecteer de specifieke dienst voor {date}:",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
        else:
            await query.edit_message_text(
                "❌ Geen preken gevonden voor deze datum.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
            )

    # Handle service selection
    elif query.data.startswith("service_"):
        parts = query.data.split("_")
        if len(parts) >= 4:
            church = parts[1]
            service_id = parts[2]
            date = parts[3]
        else:
            church = parts[1]
            service_id = parts[2]
            date = None
            
        cancellation_flags[query.message.chat_id] = False
        await query.edit_message_text(
            f"🔄 Geselecteerde preek van {church} wordt verwerkt...",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Annuleren", callback_data="cancel")]])
        )

        try:
            church_manager = ChurchManager()
            api_urls = church_manager.get_api_urls()
            analysis = await process_selected_service(api_urls[church], service_id, query)
            if cancellation_flags[query.message.chat_id]:
                await query.edit_message_text(
                    "🚫 Verwerking geannuleerd.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
                )
                return
                
            if date:
                context.user_data['current_service_date'] = date
                
            await handle_analysis_response(query, analysis, church, context)
        except Exception as e:
            await query.edit_message_text(
                "❌ Er is een fout opgetreden bij het verwerken van de preek.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Terug naar kerkenlijst", callback_data="start")]])
            )



async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: handle_email_input(u, c) if c.user_data.get('awaiting_email')
        else handle_church_input(u, c)
    ))
    
    print("Bot wordt gestart...")
    await application.initialize()
    await application.start()
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

def run_application():
    """Run the application in a synchronous context"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        lambda u, c: handle_email_input(u, c) if c.user_data.get('awaiting_email')
        else handle_church_input(u, c)
    ))
    
    print("Bot wordt gestart...")
    application.run_polling(poll_interval=1.0, timeout=20)

if __name__ == '__main__':
    try:
        # Ensure the recordings directory exists
        if not os.path.exists(ROOT_FOLDER):
            os.makedirs(ROOT_FOLDER)
            
        run_application()
    except KeyboardInterrupt:
        print("Bot gestopt door gebruiker")