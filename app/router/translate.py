from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import shutil

from service.transcription import transcribe_audio
from service.translation import translate_text
from service.voice_clone import synthesize_voice
from utils.logger import logger
from config import TEMP_DIR, OUTPUT_DIR

router = APIRouter()


@router.post("/")
async def translate_and_clone(
        audio: UploadFile = File(...),
        source_lang: str = Form(...),
        target_lang: str = Form(...),
        speaker_ref: UploadFile = File(None),
):
    logger.info("Received request for translation and voice cloning.")

    # Create a unique temp directory for this request
    request_id = str(uuid.uuid4())
    request_dir = os.path.join(TEMP_DIR, request_id)
    os.makedirs(request_dir, exist_ok=True)

    try:
        # Save uploaded audio
        input_audio_path = os.path.join(request_dir, "input.wav")
        with open(input_audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Save reference voice if provided
        speaker_ref_path = None
        if speaker_ref:
            speaker_ref_path = os.path.join(request_dir, "ref.wav")
            with open(speaker_ref_path, "wb") as f:
                shutil.copyfileobj(speaker_ref.file, f)

        # Step 1: Transcribe
        transcript = transcribe_audio(input_audio_path, source_lang)
        logger.info(f"Transcription: {transcript}")

        # Step 2: Translate
        translated_text = translate_text(transcript, source_lang, target_lang)
        logger.info(f"Translated: {translated_text}")

        # Step 3: Voice Synthesis with speaker reference
        output_audio_path = os.path.join(OUTPUT_DIR, f"{request_id}_output.wav")
        synthesize_voice(translated_text, target_lang, speaker_ref_path, output_audio_path)

        logger.info(f"Successfully synthesized audio at {output_audio_path}")
        return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")

    except Exception as e:
        logger.exception("Error in translation and cloning pipeline")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary request directory
        shutil.rmtree(request_dir, ignore_errors=True)
