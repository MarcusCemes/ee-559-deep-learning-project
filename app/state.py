from dataclasses import dataclass


@dataclass
class State:
    attributions: dict[str, float]
    sentiments: dict[str, float]
    status: str
    text: str


state = State(attributions={}, sentiments={}, status="idle", text="")
