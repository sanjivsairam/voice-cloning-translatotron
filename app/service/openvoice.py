import torch
import torchaudio
from TTS.api import TTS
import os
import tempfile
import librosa
import numpy as np
from utils.duration_matcher import match_duration

class OpenVoiceSynthesizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def generate(
        self,
        text: str,
        language: str,
        reference_audio_path: str,
        target_duration: float = None
    ) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=language,
            file_path=output_path
        )

        if target_duration:
            output_path = self._match_duration(output_path, target_duration)

        return output_path

    def _match_duration(self, audio_path: str, target_duration: float) -> str:
        audio, sr = librosa.load(audio_path, sr=None)
        current_duration = librosa.get_duration(y=audio, sr=sr)

        if abs(current_duration - target_duration) < 0.05:
            return audio_path

        stretched_audio = match_duration(audio, sr, target_duration)

        output_path = audio_path.replace(".wav", "_aligned.wav")
        torchaudio.save(output_path, torch.tensor([stretched_audio]), sample_rate=sr)
        return output_path
