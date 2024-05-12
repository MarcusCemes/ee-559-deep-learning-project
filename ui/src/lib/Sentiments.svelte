<script lang="ts">
  import SentimentBar from "./SentimentBar.svelte";

  interface Props {
    sentiments: [string, number][];
  }

  let { sentiments }: Props = $props();

  let min = $derived(Math.min(...sentiments.map(([_, value]) => value)));
  let max = $derived(Math.max(...sentiments.map(([_, value]) => value)));
  let bounds = $derived([min, max] as [number, number]);

  function formatValue(value: number) {
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}`;
  }
</script>

<div class:grid={sentiments.length > 0}>
  {#each sentiments as [name, value]}
    <div class="name">{name}</div>

    <div class="bar">
      <SentimentBar {bounds} {value} />
    </div>
    <div class="value">{formatValue(value)}</div>
  {:else}
    <div class="none">No sentiments</div>
  {/each}
</div>

<style>
  .grid {
    display: inline-grid;
    grid-template-columns: auto auto auto;
    gap: 0.5rem 1rem;
  }

  .grid,
  .none {
    font-size: 1.5rem;
  }

  .name {
    text-align: right;
  }

  .bar {
    display: flex;
    align-items: center;
  }

  .value {
    font-size: 1rem;
    margin-left: 1rem;
    text-align: right;
  }
</style>
