<script lang="ts">
  interface Props {
    bounds?: [number, number];
    value: number;
  }

  let { bounds = [0, 1], value }: Props = $props();

  let [min, max] = $derived(bounds);

  let boundedValue = $derived(Math.max(min, Math.min(max, value)));

  let colour = $derived(computeColour());
  let { left, right } = $derived(computeSize());

  function computeColour() {
    let [h, s, l] =
      boundedValue >= 0
        ? [81, (81 * boundedValue) / max, 44]
        : [0, (84 * boundedValue) / min, 60];

    return `hsl(${h}, ${s}%, ${l}%)`;
  }

  function computeSize() {
    let l = 0.5;
    let r = 0.5;

    if (boundedValue >= 0) r -= Math.min(0.5, boundedValue / max);
    else l -= Math.min(0.5, boundedValue / min);

    return { left: `${100 * l}%`, right: `${100 * r}%` };
  }
</script>

<div class="container">
  <div class="bar" style:left style:right style:background-color={colour}></div>
</div>

<style>
  .container {
    position: relative;
    width: 4rem;
    height: 1rem;
  }

  .bar {
    position: absolute;
    top: 0;
    height: 100%;
  }
</style>
