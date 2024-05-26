<script lang="ts">
  interface Props {
    text: string;
    tones: Record<string, number>;
  }

  let { text, tones }: Props = $props();

  let highlighted = $derived(highlight());

  function highlight() {
    let result = text;

    for (const [word, tone] of Object.entries(tones)) {
      const colour = tone === 2 ? "#a3e635" : tone === 1 ? "#f87171" : "";

      result = result.replaceAll(
        word,
        `<span style="color: ${colour}">${word}</span>`
      );
    }

    return result;
  }
</script>

{@html highlighted}
