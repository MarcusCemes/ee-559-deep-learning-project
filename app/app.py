from asyncio import sleep, to_thread
from contextlib import AsyncExitStack
from signal import SIGINT, signal
from time import time

from aioconsole import ainput

from .analysis import Analyser
from .audio import AudioRecorder, AudioTransformer
from .robot import Color, Robot, Sound
from .server import Server
from .state import state

INTERPRET = True
PLAY_SOUNDS = True
SAVE_PATH = "tmp/recording.wav"
SERVER_PORT = 8080

should_stop = False


class Context:
    def __init__(self):
        self.analyser = Analyser(INTERPRET)
        self.recorder = AudioRecorder()
        self.audio_transformer = AudioTransformer()
        self.robot = Robot(PLAY_SOUNDS)
        self.server = Server(SERVER_PORT)

    async def __aenter__(self):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self.server)
            stack.enter_context(self.robot)
            self._stack = stack.pop_all()

        return self

    async def __aexit__(self, *args):
        print("💥 Destroying context...")
        await self._stack.__aexit__(*args)


async def main():
    global should_stop
    signal(SIGINT, signal_handler)

    async with Context() as ctx:
        try:
            await ctx.robot.prepare()

            while not should_stop:
                command = await ainput("\n🤖 Enter command: ")

                match command:
                    case "run":
                        await run(ctx)

                    case "calibrate":
                        print("⚒️ Calibrating for ambient noise...")
                        ctx.recorder.calibrate()

                    case "clear":
                        state.sentiments = {}
                        state.text = ""
                        await ctx.server.broadcast()

                    case "loop":
                        while not should_stop:
                            await run(ctx)

                    case "prompt":
                        prompt = await ainput("💬 Enter prompt: ")
                        await run(ctx, prompt)

                    case "warmup":
                        await to_thread(ctx.analyser.classify, "Test")
                        print("🔥 Classifier model warm!")

                    case "quit":
                        break

                    case cmd:
                        if cmd:
                            print(f"⚠️ Unknown command: {cmd}")

                        print(
                            f"❓ Available commands: calibrate, clear, loop, prompt, run, warmup, quit"
                        )

            print("🤚 Stopping event loop...")

        except (EOFError, KeyboardInterrupt):
            print("⛔ Cancelled!")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            raise e


async def run(ctx: Context, text: str | None = None):
    global should_stop

    if text is None:

        print("🔉 Recording...")
        data = to_thread(ctx.recorder.record)

        # There's a delay before the microphone is acquired
        await sleep(0.5)

        await ctx.robot.play_sound(Sound.Beep)
        await ctx.robot.circle(True)

        state.status = "recording"
        state.text = ""
        await ctx.server.broadcast()

        data = await data

        if data is None:
            state.status = "idle"
            await ctx.server.broadcast()
            print("⌛ Recording timed out")
            return

        if should_stop:
            return

        await to_thread(ctx.recorder.save_wav, data, SAVE_PATH)

        if should_stop:
            return

        print("👂 Transcribing...")
        await ctx.robot.play_sound(Sound.Confirm)
        await ctx.robot.circle(False)

        state.status = "processing"
        await ctx.server.broadcast()

        text = await to_thread(transcribe, ctx)

        print("🗣️ Transcription:", text)

    print("🤔 Classifying...")
    start = time()
    is_hate_speech, sentiments = await to_thread(classify, ctx, text)

    print("🔍 Interpreting words...")
    state.attributions = await to_thread(ctx.analyser.interpret, text)

    print(f"⏱️ Elapsed: {1000 * (time() - start):.0f}ms")
    if is_hate_speech:
        print(f"🚫 Hate speech detected")
    else:
        print(f"✅ No hate speech detected")

    print_classes(sentiments)

    state.sentiments = sentiments
    await ctx.server.broadcast()

    await handle_response(text, is_hate_speech, ctx)


def transcribe(ctx: Context) -> str:
    (segments, _) = ctx.audio_transformer.transcribe(SAVE_PATH)

    text = AudioTransformer.join_segments(segments)
    print(f"🗣️ Transcription: {text}")

    return text


def classify(ctx: Context, text: str) -> tuple[bool, dict[str, float]]:
    return ctx.analyser.classify(text)


async def handle_response(text: str, is_hate_speech: bool, ctx: Context):

    state.status = "negative" if is_hate_speech else "positive"
    state.text = text
    await ctx.server.broadcast()

    if is_hate_speech:
        await ctx.robot.top_led(Color.Red)
        await ctx.robot.play_sound(Sound.Shock)
        await ctx.robot.move_away()
    else:
        await ctx.robot.top_led(Color.Green)
        await ctx.robot.play_sound(Sound.Affirmative)
        await ctx.robot.dance()


def print_classes(classes: dict[str, float]):
    for label, value in classes.items():
        print(f"   {label}: {value:.2f}")


def signal_handler(*_):
    global should_stop

    print("🤚 SIGINT received")
    should_stop = True
