from utils.logger import logger
from utils.config import settings
from pathlib import Path
from typing import Tuple, List
import whisper

# Load Whisper model (use 'base' or 'small' for CPU, 'medium' or 'large' if GPU available)
model = whisper.load_model(settings.WHISPER_MODEL)

def transcribe_with_durations(audio_path: Path) -> Tuple[str, List[float]]:
    """
    Transcribes the audio and extracts per-word durations using Whisper.
    Args:
        audio_path (Path): Path to the audio file.
    Returns:
        Tuple[str, List[float]]: Transcribed text and durations per segment.
    """
    logger.info(f"Transcribing audio: {audio_path}")

    result = model.transcribe(str(audio_path), word_timestamps=True, verbose=False)
    text = result["text"]
    durations = []

    for segment in result.get("segments", []):
        start = segment["start"]
        end = segment["end"]
        durations.append(end - start)

    logger.info(f"Transcription complete: {text}")
    return text, durations
