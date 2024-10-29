import json
import requests
import os
from datetime import datetime
from moviepy.editor import VideoFileClip
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Retrieve variables from .env file
ROOT_FOLDER = os.path.join(os.getcwd(), 'recordings')
download_filter = True
recent_amount = int(os.getenv('RECENT_AMOUNT', 1))
transcription_service = os.getenv('TRANSCRIPTION_SERVICE', 'deepgram')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
authorization_token = os.getenv('AUTHORIZATION_TOKEN')


# List of URLs for different churches
API_URLS = {
    "Wijngaarden": "https://api.kerkdienstgemist.nl/api/v2/stations/1306/recordings",
    "Nieuwe Kerk Utrecht": "https://api.kerkdienstgemist.nl/api/v2/stations/1341/recordings"
}

def select_church():
    print("Select a church to download recordings from:")
    for idx, church in enumerate(API_URLS, start=1):
        print(f"{idx}. {church}")
    
    choice = int(input("Enter the number of the church: ")) - 1
    selected_church = list(API_URLS.keys())[choice]
    return API_URLS[selected_church], selected_church

def main():
    session = requests.Session()
    # Let the user select the church
    api_base_url, selected_church = select_church()
    print(f"Selected church: {selected_church}")


    print("📡 Using provided authorization token...")
    headers = {
        'Accept': 'application/vnd.api+json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                      'Chrome/130.0.0.0 Safari/537.36',
        'Authorization': f'Bearer {authorization_token}',
    }
    params = {
        'include': 'media',
        'page': 1,
        'size': 10
    }
    recordings_downloaded = 0
    
    if transcription_service == 'gemini':
        genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        while True:
            print(f"📄 Fetching recordings from API (Page {params['page']})...")
            api_response = session.get(api_base_url, headers=headers, params=params)
            api_response.raise_for_status()
            data = api_response.json()
            recordings = data.get('data', [])
            if not recordings:
                print("❌ No more recordings found. Stopping.")
                break

            included_media = data.get('included', [])
            for recording in recordings:
                if download_filter and recordings_downloaded >= recent_amount:
                    print(f"✅ Downloaded the last {recent_amount} recordings. Stopping.")
                    return

                attributes = recording.get('attributes', {})
                title = attributes.get('title', 'Untitled')
                start_time = attributes.get('start_at', '')

                relationships = recording.get('relationships', {})
                media = relationships.get('media', {})
                media_data = media.get('data', [])
                download_url = ''
                if media_data and included_media:
                    media_ids = [m['id'] for m in media_data]
                    for media_item in included_media:
                        if media_item['id'] in media_ids and media_item['type'] == 'video_files':
                            media_attributes = media_item.get('attributes', {})
                            download_url = media_attributes.get('download_url', '')
                            if download_url:
                                break

                try:
                    date_obj = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S%z')
                    date_str = date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    date_str = start_time[:10]

                folder_name = sanitize_filename(f"{date_str}_{title}")
                recording_folder = os.path.join(ROOT_FOLDER, folder_name)

                if not os.path.exists(recording_folder):
                    os.makedirs(recording_folder)
                    print(f"📁 Created folder: {recording_folder}")

                mp4_filename = os.path.join(recording_folder, f"{date_str}_{title.replace(' ', '_')}.mp4")
                mp3_filename = mp4_filename.replace('.mp4', '.mp3').replace(' ', '_')

                if not os.path.exists(mp4_filename):
                    if download_url:
                        print(f"🎬 Found new download URL. Downloading {mp4_filename}...")
                        download_file(session, download_url, mp4_filename)
                    else:
                        print(f"❌ No download URL found for {title}. Skipping.")
                        continue
                else:
                    print(f"ℹ️ File {mp4_filename} already exists. Skipping download.")

                if not os.path.exists(mp3_filename):
                    print(f"🎶 MP3 version not found for {mp4_filename}. Converting...")
                    mp3_filename = convert_to_mp3(mp4_filename)
                    print(f"🎶 MP3 version saved as {mp3_filename}")
                else:
                    print(f"ℹ️ MP3 file {mp3_filename} already exists. Skipping conversion.")

                transcription_filename = mp3_filename.replace('.mp3', '_full.txt')
                extracted_filename = transcription_filename.replace('_full.txt', '.txt')
                if not os.path.exists(transcription_filename):
                    print(f"📝 Transcription not found for {mp3_filename}. Transcribing...")
                    if transcription_service == 'gemini':
                        transcribe_with_gemini(mp3_filename, transcription_filename)
                    elif transcription_service == 'deepgram':
                        transcribe_with_deepgram(mp3_filename, transcription_filename, extracted_filename)
                else:
                    print(f"ℹ️ Transcription file {transcription_filename} already exists. Skipping transcription.")

                analysis_filename = os.path.join(recording_folder, 'analysis.txt')
                generate_analysis(transcription_filename, analysis_filename)
                recordings_downloaded += 1
            params['page'] += 1
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching page {params['page']}: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def transcribe_with_gemini(mp3_filename, transcription_filename):
    try:
        myfile = genai.upload_file(mp3_filename)
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        result = model.generate_content([myfile, "Transcribeer deze opname."])
        if result and result.text:
            with open(transcription_filename, 'w') as f:
                f.write(result.text)
            print(f"✅ Transcription saved to {transcription_filename}")
        else:
            print(f"❌ Failed to transcribe {mp3_filename} with Gemini.")
    except Exception as e:
        print(f"❌ Gemini transcription failed: {e}")

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
        model = genai.GenerativeModel('gemini-1.5-flash')
        result = model.generate_content(prompt)
        if result and result.text:
            with open(analysis_filename, 'w') as f:
                f.write(result.text)
            print(f"✅ Analysis saved to {analysis_filename}")
        else:
            print("❌ Failed to generate analysis.")
    except Exception as e:
        print(f"❌ Analysis generation failed: {e}")

def download_file(session, url, filename):
    try:
        print(f"⬇️ Downloading {filename} from {url}")
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")

def convert_to_mp3(mp4_filename):
    mp3_filename = mp4_filename.replace('.mp4', '.mp3')
    if not os.path.exists(mp3_filename):
        try:
            print(f"🔄 Converting {mp4_filename} to {mp3_filename}...")
            video_clip = VideoFileClip(mp4_filename)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(mp3_filename)
            audio_clip.close()
            video_clip.close()
            print(f"✅ Conversion complete: {mp3_filename}")
        except Exception as e:
            print(f"❌ Failed to convert {mp4_filename} to MP3: {e}")
    else:
        print(f"ℹ️ MP3 file {mp3_filename} already exists. Skipping conversion.")
    return mp3_filename

def sanitize_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    filename = filename.replace(' ', '_')
    return filename

if __name__ == '__main__':
    main()
