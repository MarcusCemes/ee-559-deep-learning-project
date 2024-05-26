from typing import Iterable

from faster_whisper.transcribe import Segment
from speech_recognition import (
    AudioData,
    AudioFile,
    Microphone,
    Recognizer,
    WaitTimeoutError,
)

DEFAULT_MODEL = "base.en"


class AudioRecorder:
    def __init__(self):
        self.recogniser = Recognizer()

    def record(self, timeout=2, phrase_timeout=10) -> AudioData | None:
        try:
            with Microphone() as source:
                return self.recogniser.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_timeout
                )

        except WaitTimeoutError:
            return None

    def calibrate(self):
        with Microphone() as source:
            self.recogniser.adjust_for_ambient_noise(source)

    def load_file(self, path: str) -> AudioData:
        with AudioFile(path) as source:
            return self.recogniser.record(source)

    @staticmethod
    def save_wav(data: AudioData, path: str):
        with open(path, "wb") as file:
            file.write(data.get_wav_data())


class AudioTransformer:
    def __init__(self, model=DEFAULT_MODEL):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cuda", compute_type="float16")

    def transcribe(self, path: str):
        return self.model.transcribe(path, beam_size=5)

    @staticmethod
    def join_segments(segments: Iterable[Segment]) -> str:
        return " ".join(segment.text for segment in segments)
