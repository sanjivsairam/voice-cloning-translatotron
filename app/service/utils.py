import os
import tempfile
import numpy as np
import torchaudio
import librosa
import soundfile as sf
from pydub import AudioSegment


def match_duration(reference_audio: np.ndarray, target_audio: np.ndarray, target_sr: int) -> np.ndarray:
    """
    Match the duration of the target audio to the reference audio using simple resampling or time stretching.
    """
    ref_duration = reference_audio.shape[-1] / target_sr
    target_duration = target_audio.shape[-1] / target_sr

    if np.isclose(ref_duration, target_duration, atol=0.01):
        return target_audio  # Already matching

    rate = ref_duration / target_duration
    stretched = librosa.effects.time_stretch(target_audio.astype(np.float32), rate)
    return stretched


def save_temp_wav(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Save a NumPy array as a temporary WAV file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sample_rate)
    return temp_file.name


def convert_mp3_to_wav(mp3_path: str) -> str:
    """
    Convert an MP3 file to WAV format and return the WAV file path.
    """
    sound = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    sound.export(wav_path, format="wav")
    return wav_path


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize the volume of the audio.
    """
    return librosa.util.normalize(audio)


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio with a target sample rate.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy(), target_sr


def delete_temp_file(path: str):
    """
    Delete a temporary file if it exists.
    """
    if os.path.exists(path):
        os.remove(path)
