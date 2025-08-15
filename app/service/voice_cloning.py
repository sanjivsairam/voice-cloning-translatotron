import os
import tempfile
import torch
import torchaudio
from service.translatotron import run_translatotron
from service.translator import translate_with_duration
from utils.logger import logger
from utils.config import settings

from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.utils import load_audio, save_audio, split_audio
from openvoice.module.models import load_voice_model, load_conversion_model

# Load OpenVoice models
base_tts = BaseSpeakerTTS(config_path="models/base_speaker/config.json", ckpt_path="models/base_speaker/ckpt.pth")
tone_converter = ToneColorConverter(config_path="models/converter/config.json", ckpt_path="models/converter/ckpt.pth")


# Load voice identity
def extract_voice_embedding(ref_audio_path: str, ref_lang: str) -> str:
    logger.info(f"Extracting voice embedding from {ref_audio_path} (lang={ref_lang})")
    source_se = tone_converter.get_speaker_embedding(audio_path=ref_audio_path, language=ref_lang)
    embed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pth").name
    torch.save(source_se, embed_path)
    return embed_path


def synthesize_speech(text: str, lang: str, embedding_path: str, output_path: str):
    logger.info(f"Generating audio for language {lang} using embedding {embedding_path}")
    base_tts.tts(text, speaker_embedding_path=embedding_path, language=lang, output_path=output_path)
    logger.info(f"Saved synthesized audio to {output_path}")


def clone_and_translate(source_audio: str, ref_audio: str, src_lang: str, tgt_lang: str) -> str:
    logger.info("Starting full voice cloning and translation pipeline")

    # Step 1: Transcribe original audio
    transcript = run_translatotron(source_audio)

    # Step 2: Translate and get durations
    translated_text, _ = translate_with_duration(transcript, src_lang, tgt_lang)

    # Step 3: Extract voice embedding
    embedding_path = extract_voice_embedding(ref_audio, ref_lang=src_lang)

    # Step 4: Synthesize speech
    final_output = os.path.join(settings.OUTPUT_DIR, "final_cloned.wav")
    synthesize_speech(translated_text, lang=tgt_lang, embedding_path=embedding_path, output_path=final_output)

    return final_output
