import os
from transformers import pipeline
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def process_audio(input_file='output.mp3'):
    try:
        # Check if files already exist
        if os.path.exists('transcript.srt') and os.path.exists('transcript.txt'):
            logging.info("Transcript files already exist, skipping processing")
            return True

        # Check if input file exists
        if not os.path.exists(input_file):
            logging.error(f"Error: {input_file} not found")
            return False

        logging.info("Loading ASR pipeline...")
        asr_pipeline = pipeline("automatic-speech-recognition", 
                              model="openai/whisper-large-v2", 
                              device=0)
        
        logging.info("Processing audio file...")
        result = asr_pipeline(input_file, return_timestamps=True)
        
        # Write SRT file
        logging.info("Writing SRT file...")
        with open('transcript.srt', 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['chunks'], start=1):
                start = segment['timestamp'][0]
                end = segment['timestamp'][1]
                text = segment['text'].strip()
                f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n")

        # Write TXT file
        logging.info("Writing TXT file...")
        with open('transcript.txt', 'w', encoding='utf-8') as f:
            for segment in result['chunks']:
                start = segment['timestamp'][0]
                end = segment['timestamp'][1]
                text = segment['text'].strip()
                f.write(f"[{format_timestamp(start)} -> {format_timestamp(end)}] {text}\n")

        # Verify files were created
        if os.path.exists('transcript.srt') and os.path.exists('transcript.txt'):
            logging.info("Transcription completed successfully")
            return True
        else:
            logging.error("Error: Transcription files not created")
            return False

    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = process_audio()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
