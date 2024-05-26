export enum ConnectionState {
  Connecting,
  Connected,
  Disconnected,
  Error,
}

export interface Message {
  attributions: Record<string, number>;
  sentiments: Record<string, number>;
  status: string;
  text: string;
}

export type Connection = ReturnType<typeof connect>;

export function connect() {
  const { location } = window;

  const host = `ws${location.protocol.slice(4)}//${location.hostname}:8080/ws`;
  const socket = new WebSocket(host);

  let message = $state<Message>();
  let status = $state(ConnectionState.Connecting);

  socket.addEventListener("message", (event) => {
    message = JSON.parse(event.data);
  });

  socket.addEventListener("open", () => {
    status = ConnectionState.Connected;
  });

  socket.addEventListener("error", () => {
    status = ConnectionState.Error;
  });

  socket.addEventListener("close", () => {
    status = ConnectionState.Disconnected;
  });

  return {
    get message() {
      return message;
    },

    get status() {
      return status;
    },
  };
}
