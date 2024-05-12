<script lang="ts">
  import Sentiments from "./Sentiments.svelte";
  import type { Message } from "./server.svelte";

  interface Props {
    message: Message;
  }

  let { message }: Props = $props();

  let { sentiments, status, text } = $derived(message);

  let color = $derived(getColour());

  function getColour() {
    switch (status) {
      case "positive":
        return "#a3e635";

      case "negative":
        return "#f87171";
    }
  }
</script>

<h1 style:color>{status}</h1>
<h2>{text}</h2>
<Sentiments {sentiments} />

<style>
  h1 {
    font-size: 6rem;
    margin-bottom: 0;
  }

  h2 {
    font-size: 2rem;
  }
</style>
