from asyncio import to_thread
from contextlib import AsyncExitStack
from signal import signal, SIGINT

from aioconsole import ainput

from .analysis import Analyser
from .audio import AudioRecorder, AudioTransformer
from .robot import Robot
from .server import Server
from .state import state


SAVE_PATH = "tmp/recording.wav"
SERVER_PORT = 20000

should_stop = False


class Context:
    def __init__(self):
        self.analyser = Analyser()
        self.recorder = AudioRecorder()
        self.audio_transformer = AudioTransformer()
        self.robot = Robot()
        self.server = Server(SERVER_PORT)

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

                    case "calibrate":
                        print("Calibrating for ambient noise...")
                        ctx.recorder.calibrate()

                    case "prompt":
                        prompt = await ainput("Enter prompt: ")
                        await run(ctx, prompt)

                    case "quit":
                        break

                    case cmd:
                        if cmd:
                            print(f"Unknown command: {cmd}")

                        print(f"Available commands: run, calibrate, prompt, quit")

            print("Stopping event loop...")

        except (EOFError, KeyboardInterrupt):
            print("Cancelled!")

        except Exception as e:
            print(f"Error: {e}")
            raise e


async def run(ctx: Context, text: str | None = None):
    global should_stop

    if text is None:

        print("Recording...")
        state.status = "recording"
        state.text = ""
        await ctx.server.broadcast()

        data = await to_thread(ctx.recorder.record)

        if data is None:
            state.status = "idle"
            await ctx.server.broadcast()
            print("Recording timed out")
            return

        if should_stop:
            return

        await to_thread(ctx.recorder.save_wav, data, SAVE_PATH)

        if should_stop:
            return

        print("Transcribing...")
        state.status = "processing"
        await ctx.server.broadcast()

        text = await to_thread(transcribe, ctx)

    print("Classifying...")
    classes = await to_thread(classify, ctx, text)

    state.sentiments = classes
    await ctx.server.broadcast()

    await handle_response(text, classes, ctx)


def transcribe(ctx: Context) -> str:
    (segments, _) = ctx.audio_transformer.transcribe(SAVE_PATH)

    text = AudioTransformer.join_segments(segments)
    print(f"Transcription: {text}")

    return text


def classify(ctx: Context, text: str) -> dict[str, float]:
    classes = ctx.analyser.classify(text)
    print_classes(classes)

    return classes


async def handle_response(text: str, classes: dict[str, float], ctx: Context):
    positive = classes["respect"] >= 0.5

    state.status = "positive" if positive else "negative"
    state.text = text
    await ctx.server.broadcast()

    if positive:
        await ctx.robot.dance()
    else:
        await ctx.robot.move_away()


def print_classes(classes: dict[str, float]):
    for label, value in classes.items():
        print(f"{label}: {value:.2f}")


def signal_handler(*_):
    global should_stop

    print("SIGINT received")
    should_stop = True
