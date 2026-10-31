<script lang="ts">
  import { api, STAGES, type Episode, type Live } from "./lib/api";

  let { episodes, live }: { episodes: Episode[]; live: Record<number, Live> } = $props();

  function label(e: Episode): string {
    if (e.series && e.episode !== null) {
      const s = e.season !== null ? `S${String(e.season).padStart(2, "0")}` : "";
      return `${e.series} ${s}E${String(e.episode).padStart(2, "0")}`;
    }
    return e.title;
  }

  function status(e: Episode, stage: string): string {
    return e.stages.find((s) => s.stage === stage)?.status ?? "pending";
  }

  function failure(e: Episode): string | null {
    return e.error ?? e.stages.find((s) => s.error && s.status !== "done")?.error ?? null;
  }

  async function act(fn: () => Promise<void>) {
    try {
      await fn();
    } catch (err) {
      alert((err as Error).message);
    }
  }
</script>

{#if episodes.length === 0}
  <p class="empty">Nothing queued yet. <a href="#new">Request an episode</a>.</p>
{/if}

<ul class="queue">
  {#each episodes as e (e.id)}
    {@const now = live[e.id]}
    <li class="episode {e.state}">
      <div class="title">
        <strong dir="auto">{label(e)}</strong>
        <span class="meta">
          #{e.id} · {e.translator === "moe" ? "MoE (quality)" : "Qwen 9B"} · {e.state}
        </span>
      </div>

      <ol class="stages">
        {#each STAGES as stage}
          {@const st = status(e, stage)}
          <li class={st} title={stage}>
            <span>{stage}</span>
            {#if now && now.stage === stage}
              <progress max="100" value={now.pct}></progress>
            {/if}
          </li>
        {/each}
      </ol>

      {#if now}
        <p class="live">{now.stage}: {now.pct.toFixed(0)}% {now.msg}</p>
      {/if}
      {#if e.state === "failed" || (e.state !== "done" && failure(e))}
        <p class="error">{failure(e)}</p>
      {/if}

      <div class="actions">
        {#if e.state === "done"}
          <a href={`/api/episodes/${e.id}/subs/en`} download>English .srt</a>
          <a href={`/api/episodes/${e.id}/subs/ar`} download>Arabic .srt</a>
        {/if}
        {#if e.state === "failed" || e.state === "cancelled"}
          <button onclick={() => act(() => api.retry(e.id))}>Retry</button>
        {/if}
        {#if e.state === "queued" || e.state === "running"}
          <button class="quiet" onclick={() => act(() => api.cancel(e.id))}>Cancel</button>
        {/if}
      </div>
    </li>
  {/each}
</ul>
