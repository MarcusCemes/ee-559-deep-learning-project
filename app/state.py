from dataclasses import dataclass


@dataclass
class State:
    sentiments: dict[str, float]
    status: str
    text: str


state = State(sentiments={}, status="idle", text="")
