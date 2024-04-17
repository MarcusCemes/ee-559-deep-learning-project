from speech_recognition import AudioData, Microphone, Recognizer
from torch import Tensor
from whisper import (
    decode,
    DecodingOptions,
    DecodingResult,
    load_model,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)

DEFAULT_MODEL = "base"
DEFAULT_TIMEOUT = 2
DEFAULT_PHRASE_TIME_LIMIT = 5


class AudioRecorder:
    def __init__(self):
        self.recogniser = Recognizer()

    def record(
        self, timeout=DEFAULT_TIMEOUT, phrase_time_limit=DEFAULT_PHRASE_TIME_LIMIT
    ) -> AudioData:
        with Microphone() as source:
            return self.recogniser.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )

    @staticmethod
    def save_wav(data: AudioData, path: str):
        with open(path, "wb") as file:
            file.write(data.get_wav_data())


class AudioTransformer:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = load_model(model)

    def transcribe(self, path: str) -> tuple[DecodingResult, Tensor]:
        data = load_audio(path)
        data = pad_or_trim(data)

        mel = log_mel_spectrogram(data)

        options = DecodingOptions(fp16=False)
        result = decode(self.model, mel, options)

        assert isinstance(result, DecodingResult)
        return (result, mel)
