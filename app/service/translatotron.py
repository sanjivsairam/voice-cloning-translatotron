import os
import torch
import tempfile
import torchaudio
from utils.logger import logger
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.config import settings

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

def run_translatotron(audio_path: str) -> str:
    logger.info(f"Transcribing audio with Whisper: {audio_path}")

    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        speech_array = torchaudio.functional.resample(speech_array, sampling_rate, 16000)
    input_features = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt").input_features

    if torch.cuda.is_available():
        input_features = input_features.to("cuda")

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    logger.info(f"Transcription completed: {transcription}")
    return transcription
