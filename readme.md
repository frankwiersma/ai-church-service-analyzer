# AI Church Service Analyzer

A tool to download, transcribe, and analyze church service recordings in a structured format. It uses AI to generate a comprehensive overview of each service, covering aspects like sermon themes, liturgy, community announcements, and key moments.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-church-service-analyzer.git
   cd ai-church-service-analyzer
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file with the following keys:
   ```plaintext
   DEEPGRAM_API_KEY=your_deepgram_api_key
   GEMINI_API_KEY=your_gemini_api_key
   AUTHORIZATION_TOKEN=your_auth_token
   RECENT_AMOUNT=1
   TRANSCRIPTION_SERVICE=deepgram  # or 'gemini'
   ```

## Usage

Run the main script to start downloading, transcribing, and analyzing recordings:

```bash
python transcribe.py
```

## Notes
- The script saves recordings in the `recordings/` folder, which is excluded from version control.
- All transcripts and analyses are structured according to the specified church service format.