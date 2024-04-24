from tdmclient import ClientAsync


class Robot:
    def __init__(self):
        self.connection = None

    def __enter__(self):
        try:
            print("Connecting to robot...")
            self.connection = ClientAsync()
            self.connection.__enter__()

        except ConnectionRefusedError:
            print("Unable to connect to robot")

    def __exit__(self, *args):
        if self.connection:
            self.connection.__exit__(*args)

    def dance(self): ...
    def move_away(self): ...
