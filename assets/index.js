const Colours = {
  AMBER: "#d97706",
  BLACK: "#000",
  GREEN: "#16a34a",
  RED: "#dc2626",
  TEAL: "#0d9488",
  WHITE: "#fff",
};

const container = document.getElementById("container");
const title = document.getElementById("title");
const subtitle = document.getElementById("subtitle");

updateStatus("connecting");

const ws = new WebSocket(`ws://${window.location.host}/ws`);

ws.addEventListener("message", (event) => {
  console.log(event.data);
  onMessage(event.data);
});

ws.addEventListener("error", (event) => {
  updateStatus("disconnected");
  updateText("Error: " + event);
});

ws.addEventListener("close", () => {
  updateStatus("disconnected");
  updateText("Connection closed");
});

window.addEventListener("load", () => {
  document.body.addEventListener("click", () => {
    document.body.requestFullscreen();
  });
});

function onMessage(data) {
  const { status, text } = JSON.parse(data);

  if (status) {
    updateStatus(status);
  }

  if (text) {
    updateText(text);
  }
}

function updateStatus(status) {
  const [message, colour, background] = {
    connecting: ["Connecting", Colours.AMBER, Colours.BLACK],
    connected: ["Connected", Colours.AMBER, Colours.BLACK],
    idle: ["Idle", Colours.TEAL, Colours.BLACK],
    recording: ["Recording", Colours.RED, Colours.BLACK],
    processing: ["Thinking", Colours.AMBER, Colours.BLACK],
    good: ["Good job!", Colours.WHITE, Colours.GREEN],
    bad: ["Noooo!", Colours.WHITE, Colours.RED],
    disconnected: ["Disconnected", Colours.AMBER, Colours.BLACK],
  }[status] ?? ["Unknown", Colours.RED, Colours.BLACK];

  title.innerText = message;
  title.style.color = colour;
  container.style.background = background;
}

function updateText(text) {
  subtitle.innerText = text;
}
