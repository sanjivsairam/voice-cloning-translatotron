from fastapi import HTTPException, status


class TranscriptionError(HTTPException):
    def __init__(self, detail: str = "Failed to transcribe audio"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


class TranslationError(HTTPException):
    def __init__(self, detail: str = "Failed to translate text"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


class TTSGenerationError(HTTPException):
    def __init__(self, detail: str = "Failed to generate speech"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


class AudioProcessingError(HTTPException):
    def __init__(self, detail: str = "Audio processing error"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class VoiceRetentionError(HTTPException):
    def __init__(self, detail: str = "Voice retention alignment failed"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
