<script lang="ts">
  interface Props {
    text: string;
    tones: Record<string, number>;
  }

  const TEXT_COLOUR = "efefef";

  let { text, tones }: Props = $props();

  let maxTone = $derived(Math.max(...Object.values(tones)));
  let minTone = $derived(Math.min(...Object.values(tones)));

  let highlighted = $derived(highlight());

  function highlight() {
    return text
      .split(" ")
      .map((word) => {
        const tone = tones[word] ?? 0;

        const colour =
          tone > 0
            ? interpolate(TEXT_COLOUR, "a3e635", (maxTone - tone) / maxTone)
            : tone < 0
              ? interpolate(TEXT_COLOUR, "f87171", (tone - minTone) / minTone)
              : "";

        return `<span style="color: #${colour}">${word}</span>`;
      })
      .join(" ");
  }

  function interpolate(from: string, to: string, percent: number) {
    const rgb1 = parseInt(from, 16);
    const rgb2 = parseInt(to, 16);

    const [r1, g1, b1] = toArray(rgb1);
    const [r2, g2, b2] = toArray(rgb2);

    const q = 1 - percent;
    const rr = Math.round(r1 * percent + r2 * q);
    const rg = Math.round(g1 * percent + g2 * q);
    const rb = Math.round(b1 * percent + b2 * q);

    return Number((rr << 16) + (rg << 8) + rb).toString(16);
  }

  function toArray(rgb: number) {
    const r = rgb >> 16;
    const g = (rgb >> 8) % 256;
    const b = rgb % 256;

    return [r, g, b];
  }
</script>

{@html highlighted}
