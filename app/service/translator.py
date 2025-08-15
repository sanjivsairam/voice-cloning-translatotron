from utils.logger import logger
from utils.config import settings
from typing import Tuple, List
from transformers import MarianMTModel, MarianTokenizer

# Load translation model
model_name = settings.TRANSLATION_MODEL
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_with_duration(source_text: str, src_lang: str, tgt_lang: str) -> Tuple[str, List[float]]:
    """
    Translates source text and approximates duration per sentence.
    Args:
        source_text (str): Original transcribed text.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
    Returns:
        Tuple[str, List[float]]: Translated text and duration approximations.
    """
    logger.info(f"Translating text from {src_lang} to {tgt_lang}")

    # Translation
    inputs = tokenizer.prepare_seq2seq_batch([source_text], return_tensors="pt")
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Approximate durations: assume constant time per word
    words = translated_text.split()
    avg_word_duration = settings.DEFAULT_WORD_DURATION  # e.g., 0.5s
    durations = [avg_word_duration for _ in words]

    logger.info(f"Translated text: {translated_text}")
    return translated_text, durations
