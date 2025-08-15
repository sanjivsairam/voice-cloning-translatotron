import torch
from TTS.api import TTS
import tempfile
import os
import torchaudio
import librosa

class DurationEstimator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def estimate_duration(self, text: str, language: str, reference_audio_path: str) -> float:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            dummy_output_path = tmp_file.name

        self.tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio_path,
            language=language,
            file_path=dummy_output_path
        )

        waveform, sr = torchaudio.load(dummy_output_path)
        duration = librosa.get_duration(y=waveform.numpy().squeeze(), sr=sr)

        os.remove(dummy_output_path)
        return duration
