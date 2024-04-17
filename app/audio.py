from faster_whisper import WhisperModel
from speech_recognition import AudioData, Microphone, Recognizer

DEFAULT_MODEL = "base.en"


class AudioRecorder:
    def __init__(self):
        self.recogniser = Recognizer()

    def record(self) -> AudioData:
        with Microphone() as source:
            return self.recogniser.listen(source)

    @staticmethod
    def save_wav(data: AudioData, path: str):
        with open(path, "wb") as file:
            file.write(data.get_wav_data())


class AudioTransformer:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = WhisperModel(model, device="cuda", compute_type="float16")

    def transcribe(self, path: str):
        return self.model.transcribe(path, beam_size=5)
