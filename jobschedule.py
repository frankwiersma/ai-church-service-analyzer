import os
import json
import requests
import asyncio
from datetime import datetime
from moviepy.editor import VideoFileClip
from deepgram import DeepgramClient, PrerecordedOptions
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple, List, NamedTuple
from supabase import create_client, Client
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
import logging
import telegram

# Load environment variables
load_dotenv()

# Initialize Telegram bot
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
bot = telegram.Bot(os.getenv('TELEGRAM_BOT_TOKEN'))

async def send_telegram_message(message: str):
    """Send message to Telegram."""
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")


async def deactivate_unlinked_churches(supabase: Client):
    """
    Deactivate rows in the `churches` table where `active` is `true`
    but there is no corresponding link in `church_subscriptions` on `church_id`.
    """
    try:
        # Fetch unlinked active churches
        active_churches = supabase.table('churches')\
            .select('id, name, church_id')\
            .eq('active', True)\
            .execute()
        
        unlinked_churches = []
        for church in active_churches.data:
            # Check for linked subscriptions
            linked_subscriptions = supabase.table('church_subscriptions')\
                .select('id')\
                .eq('church_id', church['id'])\
                .execute()
            
            if not linked_subscriptions.data:
                unlinked_churches.append(church)

        # Deactivate unlinked churches
        for church in unlinked_churches:
            supabase.table('churches')\
                .update({'active': False})\
                .eq('id', church['id'])\
                .execute()
            print(f"Deactivated church: {church['name']} (ID: {church['id']})")

    except Exception as e:
        print(f"Error during deactivation process: {e}")



class ServiceDirectory(NamedTuple):
    """Container for service directory information"""
    path: str
    base_filename: str
    folder_name: str

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
            message = Mail(
                from_email=f'no-reply@{self.default_from_domain}',
                to_emails=recipient_email,
                subject=f'Preekanalyse - {church_name} - {date}',
                html_content=html_content
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                message
            )

            return True, "Email successfully sent"

        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _send_email(self, message: Mail) -> None:
        """Execute the actual email sending operation."""
        try:
            response = self.client.send(message)
            if response.status_code not in (200, 201, 202):
                raise Exception(f"SendGrid returned status code {response.status_code}")
        except Exception as e:
            self.logger.error(f"SendGrid API error: {str(e)}", exc_info=True)
            raise

class ChurchServiceProcessor:
    def __init__(self):
        """Initialize processor with API keys and configuration."""
        # API Keys and tokens
        self.deepgram_key = os.getenv('DEEPGRAM_API_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.kdg_token = os.getenv('AUTHORIZATION_TOKEN')
        
        # Supabase configuration
        self.supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        self.supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Directory setup
        self.output_dir = 'processed_services'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize AI services
        self.deepgram = DeepgramClient(self.deepgram_key)
        genai.configure(api_key=self.gemini_key)
        self.gemini = genai.GenerativeModel('gemini-1.5-flash-8b')

        # Store last API response for media lookup
        self.last_api_response = None

    def _create_service_directory(self, service_data: Dict, church_id: str, church_name: str) -> ServiceDirectory:
        """Create and return path to service directory with new structure."""
        # Get date and original title
        date = datetime.strptime(
            service_data['attributes']['start_at'], 
            '%Y-%m-%dT%H:%M:%S%z'
        ).strftime('%Y-%m-%d')
        
        # Get original sermon title from attributes
        original_title = service_data['attributes'].get('title', 'Untitled')
        # Remove any church name or date prefix that might be in the title
        title_parts = original_title.split(' - ', 1)
        sermon_title = title_parts[1] if len(title_parts) > 1 else title_parts[0]
        
        # Clean up the sermon title
        safe_title = self._sanitize_filename(sermon_title)
        safe_church_name = self._sanitize_filename(church_name)
        
        # Create base church directory
        church_dir = os.path.join(self.output_dir, f"{church_id}-{safe_church_name}")
        
        # Create sermon directory with date and original name
        sermon_folder_name = f"{date}_{safe_title}"
        service_dir = os.path.join(church_dir, sermon_folder_name)
        os.makedirs(service_dir, exist_ok=True)
        
        # Return directory info
        return ServiceDirectory(
            path=service_dir,
            base_filename=f"{date}_{church_id}_{safe_church_name}",
            folder_name=sermon_folder_name
        )

    
    async def get_latest_service(self, church_id: str) -> Optional[Dict]:
        """Fetch latest available service for a church."""
        url = f"https://api.kerkdienstgemist.nl/api/v2/stations/{church_id}/recordings"
        headers = {
            'Accept': 'application/vnd.api+json',
            'Authorization': f'Bearer {self.kdg_token}'
        }
        params = {'include': 'media', 'page': 1, 'size': 1}

        try:
            print(f"Fetching from URL: {url}")  # Debug log
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, headers=headers, params=params)
            )
            response.raise_for_status()
            data = response.json()
            
            # Store the full API response for media lookup
            self.last_api_response = data
            
            if not data.get('data'):
                print(f"No services found for church {church_id}")
                return None
            
            service = data['data'][0]
            return {
                'id': service['id'],
                'title': service['attributes'].get('title', 'Untitled'),
                'date': self._parse_date(service['attributes'].get('start_at', '')),
                'raw_data': {**service, 'included': data.get('included', [])}
            }
                
        except Exception as e:
            print(f"Error fetching service for church {church_id}: {e}")
            return None
    
    async def fetch_church_ids(self) -> List[Dict[str, str]]:
        """Fetch all active church IDs from Supabase."""
        try:
            response = self.supabase.table('churches')\
                .select('id, church_id, name, active')\
                .eq('active', True)\
                .execute()
            
            if response.data:
                # Return only the relevant fields for active churches
                return [
                    {'id': record['id'], 'church_id': record['church_id'], 'name': record['name']} 
                    for record in response.data
                ]
            else:
                print("No active churches found in database.")
                return []
        except Exception as e:
            print(f"Error fetching active church IDs: {e}")
            return []

            
    async def get_subscribers(self, church_id: str) -> List[Dict[str, str]]:
        """Get all subscribers for a church."""
        try:
            response = self.supabase.rpc('get_church_subscribers', {'church_uuid': church_id}).execute()
            
            if response.data:
                # Return a list of dictionaries with subscription ID and email
                return [{'subscription_id': subscriber['subscription_id'], 'email': subscriber['email']} for subscriber in response.data]
            return []
            
        except Exception as e:
            print(f"Error getting subscribers: {e}")
            return []


    async def send_report_to_subscribers(
        self, 
        church_id: str,
        church_name: str, 
        date: str, 
        report_path: str,
        sermon_id: str
    ) -> None:
        """Send report to all subscribers of a church."""
        try:
            subscribers = await self.get_subscribers(church_id)
            if not subscribers:
                print(f"No active subscribers found for church {church_name}")
                return

            email_service = EmailService(
                api_key=os.getenv('SENDGRID_API_KEY'),
                default_from_domain=os.getenv('EMAIL_DOMAIN', 'preeksamenvatting.nl')
            )

            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            successful_sends = []  # Track successful email sends

            for subscriber in subscribers:
                email = subscriber['email']
                subscription_id = subscriber['subscription_id']

                # Check if the sermon has already been sent to this subscriber
                delivery_exists = self.check_sermon_delivery(church_id, subscription_id, sermon_id)
                if delivery_exists:
                    print(f"Sermon {sermon_id} already sent to {email}, skipping.")
                    continue

                print(f"Sending report to {email}...")
                success, message = await email_service.send_report(
                    html_content=html_content,
                    recipient_email=email,
                    church_name=church_name,
                    date=date
                )
                
                if success:
                    print(f"Successfully sent report to {email}")
                    await self.update_subscriber_timestamp(email)
                    self.record_sermon_delivery(church_id, subscription_id, sermon_id)
                    successful_sends.append(email)
                else:
                    print(f"Failed to send report to {email}: {message}")

            # Send single Telegram message with summary
            if successful_sends:
                summary_message = (
                    f"🎯 Service Report Delivery Summary\n\n"
                    f"Church: {church_name}\n"
                    f"Date: {date}\n"
                    f"Successfully sent to {len(successful_sends)} subscriber(s):\n"
                    f"📧 {', '.join(successful_sends)}"
                )
                await send_telegram_message(summary_message)

        except Exception as e:
            print(f"Error sending reports: {e}")
            error_message = f"❌ Error sending reports for {church_name} on {date}: {str(e)}"
            await send_telegram_message(error_message)


    def check_sermon_delivery(self, church_id: str, subscription_id: str, sermon_id: str) -> bool:
        """Check if a sermon has already been sent to a subscriber."""
        try:
            response = self.supabase.table('sermon_deliveries')\
                .select('id')\
                .eq('church_id', church_id)\
                .eq('subscription_id', subscription_id)\
                .eq('sermon_id', sermon_id)\
                .execute()

            if response.data:
                return True
            return False
        except Exception as e:
            print(f"Error checking sermon delivery: {e}")
            return False


    def record_sermon_delivery(self, church_id: str, subscription_id: str, sermon_id: str) -> None:
        """Record that a sermon has been sent to a subscriber."""
        try:
            self.supabase.table('sermon_deliveries')\
                .insert({
                    'church_id': church_id,
                    'subscription_id': subscription_id,
                    'sermon_id': sermon_id
                }).execute()
        except Exception as e:
            print(f"Error recording sermon delivery: {e}")



    async def process_latest_service(self, church_id: str, service_data: Dict, church_name: str) -> Optional[Tuple[str, str]]:
        """Process a specific service."""
        try:
            # Create service directory with new structure
            dir_info = self._create_service_directory(
                service_data['raw_data'],
                church_id,  # Now passing the church UUID
                church_name
            )

            # Set up file paths with new naming convention
            mp3_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_sermon.mp3")
            transcript_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_transcript.txt")
            analysis_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_analysis.txt")
            report_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_report.html")

            # Check if analysis already exists
            if os.path.exists(analysis_path) and os.path.exists(report_path):
                print(f"Analysis already exists at {analysis_path}")
                return report_path, service_data['id']  # Return a tuple

            # Download and process media
            if not os.path.exists(mp3_path):
                media_path = await self._process_media(
                    service_data,
                    dir_info,
                    mp3_path
                )
                if not media_path:
                    return None

            # Generate transcription
            if not os.path.exists(transcript_path):
                print("Transcribing audio...")
                if not await self._transcribe_audio(mp3_path, transcript_path):
                    return None

            # Generate analysis
            if not os.path.exists(analysis_path):
                print("Generating analysis...")
                if not await self._generate_analysis(
                    transcript_path,
                    analysis_path,
                    church_name,
                    self._parse_date(service_data['raw_data']['attributes']['start_at'])
                ):
                    return None

            # Generate HTML report
            if not os.path.exists(report_path):
                print("Generating HTML report...")
                if not await self._generate_html_report(
                    analysis_path,
                    report_path,
                    service_data,
                    church_name  # Now correctly accepted by the method
                ):
                    return None

            # Return the path to the HTML report
            return report_path, service_data['id']

        except Exception as e:
            print(f"Error processing service: {e}")
            return None

    async def _process_media(
        self,
        service_data: Dict,
        dir_info: ServiceDirectory,
        final_mp3_path: str
    ) -> Optional[str]:
        """Download and process media file, returning path to MP3."""
        download_url, file_type = self._get_download_url(service_data)
        if not download_url:
            print("No download URL found in service data")
            return None

        # Define temporary download path
        temp_path = os.path.join(dir_info.path, f"temp_download.{file_type}")

        # Log file paths for debugging
        print(f"Checking for existing file at: {temp_path}")

        # Check if the temporary file already exists
        if os.path.exists(temp_path):
            print(f"File already exists: {temp_path}")
            # If it's a video file and the MP3 doesn't exist, convert it
            if file_type == 'mp4' and not os.path.exists(final_mp3_path):
                print(f"Converting existing file to MP3: {temp_path}")
                await self._convert_to_mp3(temp_path, final_mp3_path)
            # Return the MP3 path if it exists after conversion
            if os.path.exists(final_mp3_path):
                print(f"MP3 file already exists: {final_mp3_path}")
                return final_mp3_path
            else:
                print("MP3 conversion needed but failed to find the file.")
                return None

        try:
            # If no existing file, download it
            print("Downloading media file...")
            await self._download_file(download_url, temp_path)

            # Convert to MP3 if needed
            if file_type == 'mp4':
                print("Converting to MP3...")
                await self._convert_to_mp3(temp_path, final_mp3_path)
                os.remove(temp_path)
                print("Conversion complete")
            else:
                os.rename(temp_path, final_mp3_path)

            return final_mp3_path

        except Exception as e:
            print(f"Error processing media: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None




    async def _generate_analysis_and_report(
        self, 
        mp3_path: str, 
        dir_info: ServiceDirectory,
        service_data: Dict
    ) -> Optional[str]:
        """Generate analysis and HTML report with new file naming."""
        # Set up file paths
        transcript_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_transcript.txt")
        analysis_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_analysis.txt")
        report_path = os.path.join(dir_info.path, f"{dir_info.base_filename}_report.html")

        # Generate all files
        if not os.path.exists(transcript_path):
            print("Transcribing audio...")
            if not await self._transcribe_audio(mp3_path, transcript_path):
                return None

        if not os.path.exists(analysis_path):
            print("Generating analysis...")
            if not await self._generate_analysis(transcript_path, analysis_path):
                return None

        if not os.path.exists(report_path):
            print("Generating HTML report...")
            if not await self._generate_html_report(
                analysis_path,
                report_path,
                service_data
            ):
                return None

        return report_path

    
    def _get_download_url(self, service_data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract download URL and file type from service data."""
        try:
            raw_data = service_data['raw_data']
            included = raw_data.get('included', [])
            
            relationships = raw_data.get('relationships', {})
            media_data = relationships.get('media', {}).get('data', [])
            media_ids = {m['id'] for m in media_data}

            if not included and self.last_api_response:
                included = self.last_api_response.get('included', [])
            
            print(f"Looking for media IDs: {media_ids}")
            print(f"Available included media")
            
            for media_item in included:
                if media_item['id'] in media_ids:
                    attributes = media_item.get('attributes', {})
                    url = attributes.get('download_url')
                    if url:
                        file_type = 'mp3' if media_item['type'] == 'audio_files' else 'mp4'
                        print(f"Found media: {file_type} at {url}")
                        return url, file_type
                            
            print("No suitable media found in response")
            return None, None

        except Exception as e:
            print(f"Error extracting download URL: {str(e)}")
            return None, None


    


    async def _download_file(self, url: str, path: str) -> None:
        """Download file with reduced verbosity."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, stream=True)
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            last_printed_progress = -1  # To reduce the frequency of printed progress updates

            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            # Calculate progress percentage
                            percent = (downloaded / total_size) * 100
                            # Only print progress every 10% increase
                            if int(percent) // 10 > last_printed_progress:
                                print(f"Download progress: {int(percent)}%")
                                last_printed_progress = int(percent) // 10

            print("Download complete")
        except Exception as e:
            print(f"Error downloading file: {e}")
            if os.path.exists(path):
                os.remove(path)
            raise



    async def _convert_to_mp3(self, mp4_path: str, mp3_path: str) -> None:
        """Convert MP4 to MP3 using moviepy."""
        try:
            video = VideoFileClip(mp4_path)
            audio = video.audio
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: audio.write_audiofile(mp3_path, verbose=False, logger=None)
            )
            video.close()
            audio.close()
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            raise

    async def _transcribe_audio(self, audio_path: str, output_path: str) -> bool:
        """Transcribe audio using Deepgram."""
        print("Transcribing audio...")
        try:
            with open(audio_path, 'rb') as audio:
                options = PrerecordedOptions(
                    punctuate=True,
                    model="nova-2",
                    language="nl"
                )
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.deepgram.listen.prerecorded.v('1').transcribe_file(
                        {'buffer': audio},
                        options
                    )
                )
                
                transcript = response.to_dict()['results']['channels'][0]['alternatives'][0]['transcript']
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                return True
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return False
        
    async def _generate_analysis(self, transcript_path: str, output_path: str, church_name: str, date: str) -> bool:
        """Generate analysis using Gemini."""
        print("Generating analysis...")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()

            prompt = self._load_prompt('analysis_prompt.txt')
            if not prompt:
                return False

            # Include church_name and date in the prompt
            prompt_with_context = f"Church: {church_name}\nDate: {date}\n\n{prompt}\n\nTranscript:\n{transcript}"

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini.generate_content(prompt_with_context)
            )

            if result.text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.text)
                return True
            return False

        except Exception as e:
            print(f"Analysis generation error: {e}")
            return False


    async def _generate_html_report(
        self, 
        analysis_path: str, 
        output_path: str,
        service_data: Dict,
        church_name: str
    ) -> bool:
        """Generate HTML report using Gemini."""
        print("Generating HTML report...")
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis = f.read()

            prompt = self._load_prompt('html_report_prompt.txt')
            if not prompt:
                return False

            # Extract service metadata - handle both raw and processed data structures
            if 'raw_data' in service_data and 'attributes' in service_data['raw_data']:
                date = datetime.strptime(
                    service_data['raw_data']['attributes']['start_at'], 
                    '%Y-%m-%dT%H:%M:%S%z'
                ).strftime('%Y-%m-%d')
            elif 'attributes' in service_data:
                date = datetime.strptime(
                    service_data['attributes']['start_at'], 
                    '%Y-%m-%dT%H:%M:%S%z'
                ).strftime('%Y-%m-%d')
            else:
                # Fallback to date from service_data if available, or today's date
                date = service_data.get('date', datetime.now().strftime('%Y-%m-%d'))

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini.generate_content(
                    f"Church: {church_name}\nDate: {date}\n\nAnalysis Content:\n{analysis}\n\n{prompt}"
                )
            )

            if result.text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.text)
                
                # Call the `clean_html_report` function here
                clean_html_report(output_path)

                return True
            return False
                    
        except Exception as e:
            print(f"HTML report generation error: {e}")
            return False



    def _load_prompt(self, filename: str) -> Optional[str]:
        """Load prompt template from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt {filename}: {e}")
            return None

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem use."""
        # Replace biblical reference characters
        filename = filename.replace(':', '_').replace('/', '_')
        
        # Remove other invalid characters
        invalid_chars = '<>"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
            
        # Replace spaces and clean up
        filename = filename.replace(' ', '_')
        while '__' in filename:
            filename = filename.replace('__', '_')
            
        return filename.strip('_')



    async def process_all_churches(self):
        """Process latest services for all churches."""
        churches = await self.fetch_church_ids()

        if not churches:
            print("No churches found in database")
            return

        print(f"Found {len(churches)} churches to process")

        for church in churches:
            church_uuid = church['id']  # UUID from the 'id' field
            church_id = church['church_id']  # 'church_id' field (text)
            church_name = church['name']
            print(f"\nProcessing church: {church_name} (ID: {church_id})")

            service_info = await self.get_latest_service(church_id)
            if not service_info:
                print(f"No available services for {church_name}")
                continue

            print(f"Found service: {service_info['title']} from {service_info['date']}")

            result = await self.process_latest_service(
                church_uuid,  # Pass the church UUID
                service_info,
                church_name
            )

            if result:
                report_path, sermon_id = result
                print(f"✅ Successfully processed {church_name}")
                print(f"Report generated at: {report_path}")

                # Send reports to subscribers
                await self.send_report_to_subscribers(
                    church_id=church_uuid,  # Pass the church UUID
                    church_name=church_name,
                    date=service_info['date'],
                    report_path=report_path,
                    sermon_id=sermon_id
                )
            else:
                print(f"❌ Failed to process {church_name}")



    async def update_subscriber_timestamp(self, email: str) -> None:
        """Update the last_sent_at timestamp for a subscriber."""
        try:
            self.supabase.table('subscriptions')\
                .update({'last_sent_at': datetime.utcnow().isoformat()})\
                .eq('email', email)\
                .execute()
        except Exception as e:
            print(f"Error updating timestamp for {email}: {e}")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem use."""
        filename = filename.replace(':', '_').replace('/', '_')
        invalid_chars = '<>"|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        filename = filename.replace(' ', '_')
        while '__' in filename:
            filename = filename.replace('__', '_')
        return filename.strip('_')

    @staticmethod
    def _parse_date(start_time: str) -> str:
        """Parse date from API timestamp."""
        try:
            date_obj = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S%z')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return start_time[:10]

def generate_html_report(html_report_filename, church_name, date, analysis_content):
    """Generate HTML report using template."""
    try:
        prompt_template = load_html_report_prompt()
        if not prompt_template:
            print("❌ HTML report prompt not loaded. Skipping HTML report generation.")
            return
        
        prompt = f"Church: {church_name}\nDate: {date}\n\nAnalysis Content:\n{analysis_content}\n\n{prompt_template}"
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        result = model.generate_content(prompt)
        
        if result and result.text:
            with open(html_report_filename, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"✅ HTML report saved to {html_report_filename}")
            
            clean_html_report(html_report_filename)
        else:
            print("❌ Failed to generate HTML report.")
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")

def clean_html_report(file_path):
    """
    Cleans the HTML report file by removing the ```html at the start, the closing ```,
    and everything after the closing </html> tag, leaving valid HTML tags intact.
    
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
            
            # Remove everything after the closing </html> tag
            for i, line in enumerate(lines):
                if "</html>" in line.lower():  # Match regardless of case
                    lines = lines[:i + 1]  # Keep up to and including </html>
                    break
            
            # Write the cleaned lines back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            print(f"✅ HTML report cleaned and saved: {file_path}")
        else:
            print("❌ HTML report is empty.")
    except Exception as e:
        print(f"❌ Error cleaning HTML report: {e}")


async def main():
    """Main execution function."""
    try:
        # Send startup notification
        await send_telegram_message("🚀 Starting sermon processing script...")

        # Initialize processor
        processor = ChurchServiceProcessor()

        # Deactivate unlinked active churches
        print("Checking for unlinked active churches...")
        await deactivate_unlinked_churches(processor.supabase)
        
        # Process all churches
        await processor.process_all_churches()
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nProcessing complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        print("\nProcessing complete.")