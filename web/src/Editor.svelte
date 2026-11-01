<script lang="ts">
  import { onMount, tick } from "svelte";
  import WaveSurfer from "wavesurfer.js";
  import RegionsPlugin, { type Region } from "wavesurfer.js/plugins/regions";
  import TimelinePlugin from "wavesurfer.js/plugins/timeline";
  import { editorApi, subscribe, type CuePatch, type CueRow, type EditorData, type GlossaryEntry } from "./lib/api";

  let { id }: { id: number } = $props();

  let data = $state<EditorData | null>(null);
  let cues = $state<CueRow[]>([]);
  let glossary = $state<GlossaryEntry[]>([]);
  let error = $state<string | null>(null);
  let notice = $state<string | null>(null);
  let video = $state<HTMLVideoElement>();
  let waveEl = $state<HTMLDivElement>();
  let listEl = $state<HTMLOListElement>();
  let active = $state(-1);
  let follow = $state(true);
  let showDeleted = $state(false);
  let newAr = $state("");
  let newEn = $state("");
  let ws: WaveSurfer | null = null;
  let regions: ReturnType<typeof RegionsPlugin.create> | null = null;

  const series = $derived(data?.episode.series ?? null);
  const shown = $derived(cues.filter((c) => showDeleted || c.status !== "deleted"));
  const counts = $derived({
    verified: cues.filter((c) => c.status === "verified").length,
    edited: cues.filter((c) => c.status === "edited").length,
    total: cues.filter((c) => c.status !== "deleted").length,
  });

  onMount(() => {
    load();
    const unsubscribe = subscribe((e) => {
      if (e.type === "media" && e.id === id) load();
      if (e.type === "episode" && e.episode.id === id && e.episode.state === "done" && data && data.episode.state !== "done")
        load();
    });
    addEventListener("keydown", onKey);
    return () => {
      unsubscribe();
      removeEventListener("keydown", onKey);
      ws?.destroy();
    };
  });

  async function load() {
    try {
      data = await editorApi.load(id);
      cues = data.cues;
      glossary = data.glossary;
      error = null;
      if (data.media.proxy && data.media.peaks) {
        await tick();
        await setupWave();
      } else if (!data.preparing) {
        await editorApi.prepare(id);
        data.preparing = true;
      }
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function setupWave() {
    if (ws || !video || !waveEl) return;
    const p = await editorApi.peaks(id);
    // Scale to the loudest moment (ignoring the top 0.5% of spikes) so quiet
    // dialogue is still visible.
    const sorted = [...p.peaks].sort((a, b) => a - b);
    const loud = Math.max(sorted[Math.floor(sorted.length * 0.995)] ?? 1, 1);
    regions = RegionsPlugin.create();
    ws = WaveSurfer.create({
      container: waveEl,
      media: video,
      peaks: [Float32Array.from(p.peaks, (v) => Math.min(v / loud, 1))],
      duration: p.duration,
      height: 90,
      minPxPerSec: 60,
      barWidth: 2,
      barGap: 1,
      autoScroll: true,
      autoCenter: true,
      waveColor: "#7f8a96",
      progressColor: "#3fc1a1",
      cursorColor: "#f0a24a",
      plugins: [regions, TimelinePlugin.create({ height: 14 })],
    });
    ws.on("ready", () => {
      for (const c of cues) if (c.status !== "deleted") addRegion(c);
    });
    regions.on("region-updated", (r: Region) => {
      const c = cues.find((x) => String(x.id) === r.id);
      if (c) save(c, { start_ms: Math.round(r.start * 1000), end_ms: Math.round(r.end * 1000) });
    });
    regions.on("region-clicked", (r: Region, e: MouseEvent) => {
      e.stopPropagation();
      const i = cues.findIndex((x) => String(x.id) === r.id);
      if (i >= 0) seekTo(i);
    });
  }

  function color(c: CueRow, isActive = false): string {
    if (isActive) return "rgba(240, 162, 74, 0.35)";
    if (c.status === "verified") return "rgba(63, 193, 161, 0.25)";
    if (c.status === "edited") return "rgba(120, 160, 255, 0.25)";
    return c.uncertain ? "rgba(255, 122, 112, 0.22)" : "rgba(160, 160, 170, 0.18)";
  }

  function addRegion(c: CueRow) {
    regions?.addRegion({ id: String(c.id), start: c.start_ms / 1000, end: c.end_ms / 1000, color: color(c), drag: true, resize: true });
  }

  function region(c: CueRow): Region | undefined {
    return regions?.getRegions().find((r) => r.id === String(c.id));
  }

  function onTime() {
    if (!video) return;
    const t = video.currentTime * 1000;
    let i = cues.findIndex((c) => c.status !== "deleted" && t >= c.start_ms && t < c.end_ms);
    if (i === active) return;
    if (active >= 0 && cues[active]) region(cues[active])?.setOptions({ color: color(cues[active]) });
    active = i;
    if (i >= 0) {
      region(cues[i])?.setOptions({ color: color(cues[i], true) });
      const editing = document.activeElement?.tagName === "TEXTAREA";
      if (follow && !editing) scrollToCue(cues[i]);
    }
  }

  function seekTo(i: number, play = false) {
    if (!video || !cues[i]) return;
    video.currentTime = cues[i].start_ms / 1000 + 0.001;
    if (play) video.play();
    scrollToCue(cues[i]);
  }

  /** Scrolls the cue list (not the page) so the cue sits in its upper third. */
  function scrollToCue(c: CueRow) {
    const row = document.getElementById(`cue-${c.id}`);
    if (!row || !listEl) return;
    const top = row.offsetTop; // the list is position: relative
    if (top < listEl.scrollTop || top + row.offsetHeight > listEl.scrollTop + listEl.clientHeight)
      listEl.scrollTo({ top: top - listEl.clientHeight / 3, behavior: "smooth" });
  }

  async function save(c: CueRow, patch: CuePatch) {
    try {
      const updated = await editorApi.patch(c.id, patch);
      const i = cues.findIndex((x) => x.id === c.id);
      cues[i] = updated;
      const r = region(updated);
      if (updated.status === "deleted") r?.remove();
      else if (r) r.setOptions({ color: color(updated, i === active) });
      else addRegion(updated);
    } catch (e) {
      error = (e as Error).message;
    }
  }

  function saveText(c: CueRow, field: "ar_text" | "en_text", value: string) {
    if (value !== c[field]) save(c, { [field]: value });
  }

  /** Ctrl/Cmd+Enter: save both fields, mark verified, go to the next cue. */
  function verifyRow(c: CueRow, row: HTMLElement) {
    const ar = (row.querySelector("textarea.ar") as HTMLTextAreaElement).value;
    const en = (row.querySelector("textarea.en") as HTMLTextAreaElement).value;
    const patch: CuePatch = { status: "verified" };
    if (ar !== c.ar_text) patch.ar_text = ar;
    if (en !== c.en_text) patch.en_text = en;
    save(c, patch);
    const next = row.nextElementSibling?.querySelector("textarea.en") as HTMLTextAreaElement | null;
    next?.focus();
    const i = cues.findIndex((x) => x.id === c.id);
    if (cues[i + 1]) seekTo(i + 1);
  }

  function onRowKey(e: KeyboardEvent, c: CueRow) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      verifyRow(c, (e.currentTarget as HTMLElement).closest("li")!);
    } else if (e.key === "Escape") {
      (e.currentTarget as HTMLElement).blur();
    }
  }

  function onKey(e: KeyboardEvent) {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || !video) return;
    if (e.key === " ") {
      e.preventDefault();
      if (video.paused) video.play();
      else video.pause();
    } else if (e.key === "j" || e.key === "k") {
      const t = video.currentTime * 1000;
      const live = cues.map((c, i) => [c, i] as const).filter(([c]) => c.status !== "deleted");
      const target =
        e.key === "k"
          ? live.find(([c]) => c.start_ms > t + 50)
          : [...live].reverse().find(([c]) => c.start_ms < t - 300);
      if (target) seekTo(target[1]);
    } else if (e.key === "r" && active >= 0) {
      seekTo(active, true);
    }
  }

  async function publish() {
    try {
      const r = await editorApi.publish(id);
      notice = `Published: ${r.en_cues} English and ${r.ar_cues} Arabic cues written to the .srt files.`;
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function retranslate() {
    const untouched = cues.filter((c) => c.status === "auto").length;
    const examples = counts.verified + counts.edited;
    if (!confirm(`Retranslate the ${untouched} lines you haven't touched, using ${glossary.length} glossary terms and ${examples} lines you fixed or approved? Your edits are kept.`))
      return;
    try {
      await editorApi.retranslate(id);
      notice = "Retranslation queued. This page reloads when it's done.";
      if (data) data.episode.state = "queued";
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function addTerm() {
    if (!series) return;
    try {
      glossary = await editorApi.addGlossary(series, newAr, newEn);
      newAr = "";
      newEn = "";
    } catch (e) {
      error = (e as Error).message;
    }
  }

  async function removeTerm(g: GlossaryEntry) {
    await editorApi.deleteGlossary(g.id);
    glossary = glossary.filter((x) => x.id !== g.id);
  }

  function useSelection() {
    const text = window.getSelection()?.toString().trim();
    if (text) newAr = text;
  }

  function time(ms: number): string {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}.${Math.floor((ms % 1000) / 100)}`;
  }
</script>

{#if error}<p class="error">{error}</p>{/if}

{#if data}
  <div class="editor-head">
    <a href="#queue">← Queue</a>
    <strong dir="auto">{data.episode.series ?? data.episode.title}{data.episode.episode ? ` · E${data.episode.episode}` : ""}</strong>
    <span class="meta">{counts.verified} verified · {counts.edited} edited · {counts.total} lines</span>
    <span class="spacer"></span>
    <button class="quiet" onclick={retranslate} disabled={data.episode.state !== "done"}>Retranslate untouched lines</button>
    <button onclick={publish}>Publish .srt</button>
  </div>
  {#if notice}<p class="notice">{notice}</p>{/if}

  <div class="editor-grid">
    <div class="player">
      {#if data.media.proxy && data.media.peaks}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video bind:this={video} src={`/api/media/${id}/proxy.mp4`} controls preload="auto" ontimeupdate={onTime}></video>
        <div class="wave" bind:this={waveEl}></div>
        <p class="hint">
          Space play/pause · J/K previous/next line · R replay line · drag region edges to retime ·
          Ctrl/⌘+Enter verify & next
        </p>
      {:else}
        <p class="empty">Preparing the video and waveform for the editor (a minute or two)…</p>
      {/if}
    </div>

    <aside class="glossary">
      <h3>Glossary{series ? ` · ${series}` : ""}</h3>
      {#if series}
        <p class="hint">Used for every future translation of this show, with the lines you fix or verify.</p>
        <form onsubmit={(e) => (e.preventDefault(), addTerm())}>
          <input dir="rtl" bind:value={newAr} placeholder="بلا هالموشح" />
          <input bind:value={newEn} placeholder="spare me the lecture" />
          <button disabled={!newAr.trim() || !newEn.trim()}>Add</button>
        </form>
        <button class="quiet small" type="button" onclick={useSelection}>Use selected Arabic text</button>
        <ul>
          {#each glossary as g (g.id)}
            <li><span dir="rtl">{g.ar}</span> → {g.en} <button class="link" onclick={() => removeTerm(g)}>×</button></li>
          {/each}
        </ul>
      {:else}
        <p class="hint">This episode has no series name, so it can't have a glossary.</p>
      {/if}
      <label><input type="checkbox" bind:checked={follow} /> Follow playback</label>
      <label><input type="checkbox" bind:checked={showDeleted} /> Show deleted lines</label>
    </aside>
  </div>

  <ol class="cues" bind:this={listEl}>
    {#each shown as c (c.id)}
      {@const i = cues.indexOf(c)}
      <li id={`cue-${c.id}`} class={`cue ${c.status}`} class:active={i === active} class:uncertain={c.uncertain}>
        <button class="time" onclick={() => seekTo(i, true)} title="Play from here">{time(c.start_ms)}</button>
        <div class="texts">
          <textarea class="ar" dir="rtl" rows="2" value={c.ar_text}
            onblur={(e) => saveText(c, "ar_text", e.currentTarget.value)}
            onkeydown={(e) => onRowKey(e, c)}></textarea>
          <textarea class="en" rows="2" value={c.en_text}
            onblur={(e) => saveText(c, "en_text", e.currentTarget.value)}
            onkeydown={(e) => onRowKey(e, c)}></textarea>
          {#if c.en_text !== c.en_mt}<small class="was">machine: {c.en_mt}</small>{/if}
        </div>
        <div class="row-actions">
          <span class="status">{c.status}{c.uncertain ? " ?" : ""}</span>
          {#if c.status === "deleted"}
            <button class="quiet small" onclick={() => save(c, { status: "auto" })}>Restore</button>
          {:else}
            <button class="small" onclick={() => save(c, { status: "verified" })} disabled={c.status === "verified"}>✓</button>
            <button class="quiet small" onclick={() => save(c, { status: "deleted" })} title="Delete line">🗑</button>
          {/if}
        </div>
      </li>
    {/each}
  </ol>
{:else if !error}
  <p class="empty">Loading…</p>
{/if}
