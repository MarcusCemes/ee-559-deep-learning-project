from .audio import AudioRecorder, AudioTransformer
from .robot import Robot

SAVE_PATH = "tmp/recording.wav"


class Context:
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.audio_transformer = AudioTransformer()
        self.robot = Robot()

    def __enter__(self):
        self.robot.__enter__()
        return self

    def __exit__(self, *args):
        self.robot.__exit__(*args)


def main():
    try:
        with Context() as ctx:
            while True:
                print("Press Enter to start recording...", end="", flush=True)
                input()
                run(ctx)

    except KeyboardInterrupt:
        pass


def run(ctx: Context):
    print("Recording...")
    sample = ctx.audio_recorder.record()
    AudioRecorder.save_wav(sample, SAVE_PATH)

    print("Analysing...")
    (segments, info) = ctx.audio_transformer.transcribe(SAVE_PATH)

    # TODO: text analysis
    text = AudioTransformer.join_segments(segments)

    good_response = True

    if good_response:
        ctx.robot.dance()
    else:
        ctx.robot.move_away()


if __name__ == "__main__":
    main()
