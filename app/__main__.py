from .audio import AudioRecorder, AudioTransformer

SAVE_PATH = "tmp/recording.wav"


def main():

    audio_recorder = AudioRecorder()
    audio_transformer = AudioTransformer()

    print("Recording...")
    sample = audio_recorder.record()
    AudioRecorder.save_wav(sample, SAVE_PATH)

    print("Analysing audio...")
    result = audio_transformer.transcribe(SAVE_PATH)

    print(result)


if __name__ == "__main__":
    main()
