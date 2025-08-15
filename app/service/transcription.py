import whisper
from utils.logger import logger

# Load Whisper model globally to avoid reloading every time
model = whisper.load_model("base")

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    try:
        logger.info(f"Transcribing audio: {audio_path}, language: {language}")
        result = model.transcribe(audio_path, language=language)
        text = result.get("text", "").strip()
        if not text:
            raise ValueError("No transcription generated")
        return text
    except Exception as e:
        logger.exception("Transcription failed")
        raise
