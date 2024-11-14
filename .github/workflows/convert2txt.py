from transformers import pipeline

asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=0)
result = asr_pipeline("output.mp3", return_timestamps=True)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

with open('transcript.srt', 'w', encoding='utf-8') as f:
    for i, segment in enumerate(result['chunks'], start=1):
        start = segment['timestamp'][0]
        end = segment['timestamp'][1]
        text = segment['text'].strip()
        f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n")

with open('transcript.txt', 'w', encoding='utf-8') as f:
    for segment in result['chunks']:
        start = segment['timestamp'][0]
        end = segment['timestamp'][1]
        text = segment['text'].strip()
        f.write(f"[{format_timestamp(start)} -> {format_timestamp(end)}] {text}\n")

print("Transcription completed and saved.")
