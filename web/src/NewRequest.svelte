<script lang="ts">
  import { api, type PreviewItem, type Translator } from "./lib/api";

  let { onqueued }: { onqueued: () => void } = $props();

  interface Row extends PreviewItem {
    picked: boolean;
    series: string;
    season: number | null;
    episode: number | null;
  }

  let input = $state("");
  let filePath = $state("");
  let translator = $state<Translator>("qwen9b");
  let rows = $state<Row[]>([]);
  let listTitle = $state<string | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let hideShort = $state(true);

  const visible = $derived(rows.filter((r) => !hideShort || (r.duration ?? 0) >= 1200));
  const pickedCount = $derived(rows.filter((r) => r.picked).length);

  async function preview() {
    if (!input.trim()) return;
    busy = true;
    error = null;
    try {
      const res = await api.preview(input);
      listTitle = res.title;
      rows = res.items.map((it) => ({
        ...it,
        picked: false,
        series: it.parsed.series ?? "",
        season: it.parsed.season,
        episode: it.parsed.episode,
      }));
    } catch (e) {
      error = (e as Error).message;
      rows = [];
    } finally {
      busy = false;
    }
  }

  function pickAll(on: boolean) {
    for (const r of visible) r.picked = on;
  }

  async function queuePicked() {
    busy = true;
    error = null;
    try {
      const items = rows
        .filter((r) => r.picked)
        .map((r) => ({ url: r.url, title: r.title, series: r.series || null, season: r.season, episode: r.episode }));
      await api.queueUrls(items, translator);
      rows = [];
      onqueued();
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  async function queueFile() {
    busy = true;
    error = null;
    try {
      await api.queueFile(filePath.trim(), translator);
      filePath = "";
      onqueued();
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  function minutes(s: number | null): string {
    return s ? `${Math.round(s / 60)} min` : "";
  }
</script>

<section class="card">
  <h2>From YouTube (or any yt-dlp site)</h2>
  <form onsubmit={(e) => (e.preventDefault(), preview())}>
    <input
      bind:value={input}
      dir="auto"
      placeholder="Paste a video or playlist URL, or search: مسلسل ضيعة ضايعة الحلقة 3"
    />
    <button disabled={busy || !input.trim()}>{busy ? "Looking…" : "Preview"}</button>
  </form>

  <fieldset class="translator">
    <legend>Translator</legend>
    <label><input type="radio" bind:group={translator} value="qwen9b" /> Qwen 9B, fast (~20 min/episode)</label>
    <label><input type="radio" bind:group={translator} value="moe" /> Qwen 35B MoE, better idioms (~45 min/episode)</label>
  </fieldset>

  {#if error}<p class="error">{error}</p>{/if}

  {#if rows.length}
    <div class="toolbar">
      {#if listTitle}<strong dir="auto">{listTitle}</strong>{/if}
      <label><input type="checkbox" bind:checked={hideShort} /> Hide clips under 20 min</label>
      <button type="button" class="quiet" onclick={() => pickAll(true)}>Select all</button>
      <button type="button" class="quiet" onclick={() => pickAll(false)}>None</button>
      <button type="button" disabled={busy || pickedCount === 0} onclick={queuePicked}>
        Queue {pickedCount} episode{pickedCount === 1 ? "" : "s"}
      </button>
    </div>
    <table class="preview">
      <thead>
        <tr><th></th><th>Title</th><th>Series</th><th>S</th><th>E</th><th>Length</th></tr>
      </thead>
      <tbody>
        {#each visible as r (r.url)}
          <tr class:picked={r.picked}>
            <td><input type="checkbox" bind:checked={r.picked} /></td>
            <td dir="auto" class="t">{r.title}<br /><small>{r.channel ?? ""}</small></td>
            <td><input dir="auto" class="series" bind:value={r.series} /></td>
            <td><input type="number" min="1" class="num" bind:value={r.season} /></td>
            <td><input type="number" min="1" class="num" bind:value={r.episode} /></td>
            <td>{minutes(r.duration)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<section class="card">
  <h2>A file on the server</h2>
  <form onsubmit={(e) => (e.preventDefault(), queueFile())}>
    <input bind:value={filePath} placeholder="/home/hamza/Videos/Bab Al Hara S01E05.mkv" />
    <button disabled={busy || !filePath.trim()}>Queue file</button>
  </form>
</section>
