from transformers import MarianMTModel, MarianTokenizer
from utils.logger import logger
from functools import lru_cache

@lru_cache(maxsize=32)
def load_translation_model(src_lang: str, tgt_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    logger.info(f"Loading MarianMT model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text: str, src_lang: str = "en", tgt_lang: str = "fr") -> str:
    try:
        tokenizer, model = load_translation_model(src_lang, tgt_lang)
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        logger.info(f"Translated text: {tgt_text}")
        return tgt_text
    except Exception as e:
        logger.exception(f"Translation failed from {src_lang} to {tgt_lang}")
        raise
