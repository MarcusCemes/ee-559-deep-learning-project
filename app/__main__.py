from asyncio import run, sleep
from threading import Thread

from .analysis import Analyser
from .audio import AudioRecorder, AudioTransformer
from .robot import Robot
from .server import Server


SAVE_PATH = "tmp/recording.wav"


class Context:
    def __init__(self):
        self.analyser = Analyser()
        self.audio_recorder = AudioRecorder()
        self.audio_transformer = AudioTransformer()
        self.robot = Robot()
        self.server = Server()

    async def __aenter__(self):
        self.robot.__enter__()
        await self.server.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.server.__aexit__(*args)
        self.robot.__exit__(*args)


async def main():
    try:
        async with Context() as ctx:
            while True:
                run_cycle(ctx)

    except EOFError:
        pass

    except KeyboardInterrupt:
        pass


def run_cycle(ctx: Context):
    print("Recording...")
    sample = ctx.audio_recorder.record()
    AudioRecorder.save_wav(sample, SAVE_PATH)

    print("Analysing...")
    (segments, _) = ctx.audio_transformer.transcribe(SAVE_PATH)

    text = AudioTransformer.join_segments(segments)
    print(text)
    result = ctx.analyser.classify(text)
    print(result)

    # Yeah... I know... I'm a bit lazy
    good_response = result[0][0] < 1
    response_type = "Good" if good_response else "Bad"

    print(f"Response: {response_type}\n")

    # if good_response:
    #     ctx.robot.dance()
    # else:
    #     ctx.robot.move_away()


if __name__ == "__main__":
    run(main())
