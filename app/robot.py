from logging import error, info

from tdmclient import ClientAsync


class Robot:
    def __init__(self):
        self.connection = None

    def __enter__(self):
        try:
            info("Connecting to Thymio...")
            self.connection = ClientAsync()
            self.connection.__enter__()

        except ConnectionRefusedError:
            error("Unable to connect to Thymio")

    def __exit__(self, *args):
        if self.connection:
            self.connection.__exit__(*args)

    def dance(self):
        info("Dancing!")

    def move_away(self):
        print("Moving away!")
