from asyncio import to_thread
from contextlib import AsyncExitStack
from signal import signal, SIGINT

from aioconsole import ainput
from torch import Tensor

from .analysis import Analyser
from .audio import AudioRecorder, AudioTransformer
from .robot import Robot
from .server import Server
from .state import state

CLASS_LABELS = [
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
]

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

                    case "calibrate":
                        print("Calibrating for ambient noise...")
                        ctx.recorder.calibrate()

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
    global should_stop

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

    print("Transcribing and classifying...")
    state.status = "processing"
    await ctx.server.broadcast()

    (text, classes) = await to_thread(transcribe_and_classify, ctx)

    state.sentiments = group_classes(classes)
    await ctx.server.broadcast()

    await handle_response(text, classes, ctx)


def transcribe_and_classify(ctx: Context) -> tuple[str, Tensor]:
    (segments, _) = ctx.audio_transformer.transcribe(SAVE_PATH)

    text = AudioTransformer.join_segments(segments)
    print(f"Transcription: {text}")

    classes = ctx.analyser.classify(text)
    return (text, classes)


def group_classes(classes: Tensor):
    [values] = classes.tolist()
    return [[l, v] for (l, v) in zip(CLASS_LABELS, values)]


async def handle_response(text: str, classes: Tensor, ctx: Context):
    positive = is_positive(classes)
    type = "positive" if positive else "negative"
    print(f"Response: {type}")

    state.status = type
    state.text = text
    await ctx.server.broadcast()

    if positive:
        await ctx.robot.dance()
    else:
        await ctx.robot.move_away()


def is_positive(classes: Tensor):
    return classes[0].mean() < 1.0


def signal_handler(*_):
    global should_stop

    print("SIGINT received")
    should_stop = True
