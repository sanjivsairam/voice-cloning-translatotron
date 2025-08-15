import os
import librosa
import soundfile as sf
import tempfile
import torchaudio
from pydub import AudioSegment


class AudioUtils:
    @staticmethod
    def convert_audio_to_wav(input_path: str, output_path: str, target_sr: int = 16000):
        """Converts audio to WAV format with target sample rate."""
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")

    @staticmethod
    def trim_silence(input_path: str, output_path: str, top_db: int = 40):
        """Trims leading and trailing silence from an audio file."""
        y, sr = librosa.load(input_path, sr=None)
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        sf.write(output_path, yt, sr)

    @staticmethod
    def resample_audio(input_path: str, output_path: str, target_sr: int = 16000):
        """Resamples audio to a different sample rate."""
        y, sr = librosa.load(input_path, sr=None)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(output_path, y_resampled, target_sr)

    @staticmethod
    def extract_duration(file_path: str) -> float:
        """Extracts the duration of a WAV file."""
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)

    @staticmethod
    def convert_mp3_to_wav(input_path: str, target_sr: int = 16000) -> str:
        """Converts MP3 to WAV with a standard sample rate."""
        audio = AudioSegment.from_mp3(input_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(tmp_file.name, format="wav")
        return tmp_file.name
