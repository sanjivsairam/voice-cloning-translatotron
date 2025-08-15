from utils.logger import logger
import numpy as np

def align_durations(source_durations: list[float], target_text: str) -> list[float]:
    """
    Aligns the duration of the translated text with the original voice.

    Args:
        source_durations (list): List of word durations in seconds from source audio
        target_text (str): Translated text to align

    Returns:
        list: Scaled durations for each word in translated text
    """
    try:
        logger.info("Aligning durations...")
        src_total = sum(source_durations)
        tgt_word_count = len(target_text.strip().split())

        if tgt_word_count == 0:
            raise ValueError("Translated text has no words.")

        avg_duration = src_total / tgt_word_count
        aligned_durations = [avg_duration] * tgt_word_count

        logger.info(f"Source duration: {src_total:.2f}s, "
                    f"Target word count: {tgt_word_count}, "
                    f"Avg duration: {avg_duration:.2f}s")

        return aligned_durations
    except Exception as e:
        logger.exception("Duration alignment failed")
        raise
