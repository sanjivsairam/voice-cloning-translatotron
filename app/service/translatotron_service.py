from utils.logger import logger
from utils.config import settings
from service.speech_to_text import transcribe_audio
from service.translator import translate_text
from service.duration_aligner import align_durations
from service.voice_generator import synthesize_voice
from pathlib import Path

def process_translation_pipeline(audio_path: Path, source_lang: str, target_lang: str) -> Path:
    """
    Full pipeline to convert input audio to translated output speech with voice retention.
    Args:
        audio_path (Path): Path to input .wav audio file
        source_lang (str): Source language code (e.g., "en", "ta")
        target_lang (str): Target language code (e.g., "es", "fr")
    Returns:
        Path: Path to output translated audio
    """
    try:
        logger.info("Step 1: Transcribing audio...")
        transcription, original_durations = transcribe_audio(audio_path, source_lang)
        logger.info(f"Transcription: {transcription}")

        logger.info("Step 2: Translating text...")
        translated_text = translate_text(transcription, source_lang, target_lang)
        logger.info(f"Translated: {translated_text}")

        logger.info("Step 3: Aligning durations...")
        aligned_durations = align_durations(original_durations, translated_text)

        logger.info("Step 4: Generating voice output...")
        output_path = synthesize_voice(
            translated_text,
            aligned_durations,
            source_audio_path=audio_path,
            source_lang=source_lang,
            target_lang=target_lang
        )

        logger.info(f"Pipeline complete. Output: {output_path}")
        return output_path

    except Exception as e:
        logger.exception("Failed in processing full pipeline")
        raise
