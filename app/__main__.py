from . import audio


def main():
    print("Listening...")
    sample = audio.record_sample()
    print("Processing...")


if __name__ == "__main__":
    main()
