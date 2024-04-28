from asyncio import to_thread
from contextlib import AsyncExitStack
from signal import signal, SIGINT

from aioconsole import ainput

from .analysis import Analyser
from .audio import AudioRecorder, AudioTransformer
from .robot import Robot
from .server import Server

SAVE_PATH = "tmp/recording.wav"

should_stop = False


class Context:
    def __init__(self):
        self.analyser = Analyser()
        self.recorder = AudioRecorder()
        self.audio_transformer = AudioTransformer()
        self.robot = Robot()
        self.server = Server()

    async def __aenter__(self):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self.server)
            stack.enter_context(self.robot)
            self._stack = stack.pop_all()

        return self

    async def __aexit__(self, *args):
        print("Destroying context...")
        await self._stack.__aexit__(*args)


async def main():
    global should_stop
    signal(SIGINT, signal_handler)

    async with Context() as ctx:
        try:
            while not should_stop:
                command = await ainput("\nEnter command: ")

                match command:
                    case "run":
                        await run(ctx)

                    case "quit":
                        break

                    case cmd:
                        if cmd:
                            print(f"Unknown command: {cmd}")

                        print(f"Available commands: run, quit")

            print("Stopping event loop...")

        except (EOFError, KeyboardInterrupt):
            print("Cancelled!")

        except Exception as e:
            print(f"Error: {e}")
            raise e


async def run(ctx: Context):
    print("Recording...")
    await ctx.server.set_status("recording")
    await ctx.server.set_text(None)
    data = await to_thread(ctx.recorder.record)
    await to_thread(ctx.recorder.save_wav, data, SAVE_PATH)

    print("Transcribing and classifying...")
    await ctx.server.set_status("processing")
    (good, text) = await to_thread(transcribe_and_classify, ctx)

    await handle_response(good, text, ctx)


def transcribe_and_classify(ctx: Context):
    (segments, _) = ctx.audio_transformer.transcribe(SAVE_PATH)

    text = AudioTransformer.join_segments(segments)
    print(f"Transcription: {text}")

    classes = ctx.analyser.classify(text)
    good = True if (classes[0][0] < 1.0) else False

    return (good, text)


async def handle_response(good: bool, text: str, ctx: Context):
    type = "GOOD" if good else "BAD"
    print(f"Response: {type}")

    await ctx.server.set_text(text)

    if good:
        await ctx.server.set_status("good")
        await ctx.robot.dance()
    else:
        await ctx.server.set_status("bad")
        await ctx.robot.move_away()


def signal_handler(*_):
    global should_stop

    print("SIGINT received")
    should_stop = True
