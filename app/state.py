from dataclasses import dataclass


@dataclass
class State:
    sentiments: list[list[str | float]]
    status: str
    text: str


state = State(sentiments=[], status="idle", text="")
