<script lang="ts">
  import Message from "./lib/Message.svelte";
  import Sentiments from "./lib/Sentiments.svelte";
  import {
    ConnectionState,
    connect,
    type Connection,
  } from "./lib/server.svelte";

  interface Data {
    sentiments: [string, number][];
    status: string;
    text: string;
  }

  let connection = $state<Connection>();

  // let { sentiments, status, text } = $state<Data>({
  //   sentiments: [
  //     ["happy", 0.5],
  //     ["offensive", 2.0],
  //   ],
  //   status: "idle",
  //   text: "A message here",
  // });

  function isActive(connection: Connection) {
    return [ConnectionState.Connecting, ConnectionState.Connected].includes(
      connection.status
    );
  }

  function onclick() {
    connection = connect();
  }
</script>

<main>
  {#if connection}
    {#if connection.status === ConnectionState.Connecting}
      <h2>Connecting...</h2>
    {:else if connection.status === ConnectionState.Connected}
      {#if connection.message}
        <Message message={connection.message} />
      {/if}
    {:else if connection.status === ConnectionState.Disconnected}
      <h2>Disconnected</h2>
    {:else if connection.status === ConnectionState.Error}
      <h2>Error</h2>
    {/if}
  {/if}

  {#if !connection || [ConnectionState.Disconnected, ConnectionState.Error].includes(connection.status)}
    <button {onclick}>connect</button>
  {/if}
</main>

<style>
  main {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #efefef;
  }

  button {
    font-size: 2rem;
    padding: 1rem 2rem;
    background-color: #f00;
    color: #fff;
    border: none;
    border-radius: 0.25rem;
    font-family: "Jersey 15", system-ui, Avenir, Helvetica, Arial, sans-serif;
    cursor: pointer;
  }
</style>
