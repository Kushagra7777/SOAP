import whisper
import os

# Path to your audio file (handle Unicode file names)
audio_path = r"audio/新規録音 5.m4a"

# Load the Whisper model (you can change to "small" or "large")
# model = whisper.load_model("medium")
model = whisper.load_model("medium").to("cuda")

# Transcribe in Japanese
print("Transcribing in Japanese...")
result_ja = model.transcribe(audio_path, language="Japanese")
japanese_text = result_ja["text"]

# Save Japanese transcript
with open("transcription_japanese.txt", "w", encoding="utf-8") as f:
    f.write(japanese_text)

# Translate to English
print("Translating to English...")
result_en = model.transcribe(audio_path, task="translate")
english_text = result_en["text"]

# Save English translation
with open("transcription_english.txt", "w", encoding="utf-8") as f:
    f.write(english_text)

print("✅ Done! Files saved as:")
print("- transcription_japanese.txt")
print("- transcription_english.txt")
