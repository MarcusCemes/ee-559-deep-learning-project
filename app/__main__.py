from time import time

from .audio import AudioRecorder, AudioTransformer

SAVE_PATH = "tmp/recording.wav"


def test_audio():

    audio_recorder = AudioRecorder()
    audio_transformer = AudioTransformer()

    print("Recording...", end="", flush=True)
    sample = audio_recorder.record()
    AudioRecorder.save_wav(sample, SAVE_PATH)
    print(" Done")

    print("Analysing...", end="", flush=True)
    start = time()
    segments, info = audio_transformer.transcribe(SAVE_PATH)
    print(" Done")
    print(f"Analysis took {time() - start:.2f}s")

    print(info, end="\n\n")

    for segment in segments:
        print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")


def main():
    test_audio()


if __name__ == "__main__":
    main()
