<script lang="ts">
  import { onMount } from "svelte";
  import { api, subscribe, type Episode, type GpuStatus, type Live } from "./lib/api";
  import Editor from "./Editor.svelte";
  import NewRequest from "./NewRequest.svelte";
  import QueueView from "./QueueView.svelte";

  let route = $state(location.hash.slice(1) || "queue");
  let episodes = $state<Episode[]>([]);
  let live = $state<Record<number, Live>>({});
  let gpu = $state<GpuStatus>({ paused: false, holding: null, vram_used_mb: null });
  let error = $state<string | null>(null);

  async function refresh() {
    try {
      const data = await api.episodes();
      episodes = data.episodes;
      live = Object.fromEntries(Object.entries(data.live).map(([k, v]) => [Number(k), v]));
      gpu = data.gpu;
      error = null;
    } catch (e) {
      error = (e as Error).message;
    }
  }

  onMount(() => {
    const onHash = () => (route = location.hash.slice(1) || "queue");
    addEventListener("hashchange", onHash);
    refresh();
    const unsubscribe = subscribe((e) => {
      if (e.type === "episode") {
        const i = episodes.findIndex((x) => x.id === e.episode.id);
        if (i >= 0) episodes[i] = e.episode;
        else episodes = [e.episode, ...episodes];
        setLive(e.episode.id, e.live);
      } else if (e.type === "progress") {
        setLive(e.id, e.live);
      } else if (e.type === "gpu") {
        gpu = e.gpu;
      }
    });
    return () => {
      removeEventListener("hashchange", onHash);
      unsubscribe();
    };
  });

  function setLive(id: number, value: Live | null) {
    if (value) live[id] = value;
    else delete live[id];
  }

  async function togglePause() {
    gpu = await api.setPaused(!gpu.paused);
  }

  const active = $derived(episodes.filter((e) => e.state === "queued" || e.state === "running").length);
</script>

<header>
  <h1><span lang="ar">شو قال</span> shu-2al</h1>
  <nav>
    <a href="#queue" class:current={route === "queue"}>Queue{active ? ` (${active})` : ""}</a>
    <a href="#new" class:current={route === "new"}>New request</a>
  </nav>
  <div class="gpu" class:holding={gpu.holding}>
    <span>
      GPU {gpu.holding ? `on hold: ${gpu.holding}` : "available"}
      {#if gpu.vram_used_mb !== null}· {gpu.vram_used_mb} MiB used{/if}
    </span>
    <button onclick={togglePause}>{gpu.paused ? "Resume GPU work" : "Pause GPU (gaming)"}</button>
  </div>
</header>

{#if error}<p class="error">Can't reach the server: {error}</p>{/if}

<main class:wide={route.startsWith("episode/")}>
  {#if route.startsWith("episode/")}
    {#key route}<Editor id={Number(route.split("/")[1])} />{/key}
  {:else if route === "new"}
    <NewRequest onqueued={() => (location.hash = "queue")} />
  {:else}
    <QueueView {episodes} {live} />
  {/if}
</main>
