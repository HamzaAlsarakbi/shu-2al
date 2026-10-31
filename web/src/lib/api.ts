// Typed client for the shu2al server (crates/server/src/api.rs).

export const STAGES = ["fetch", "extract", "asr", "align", "cues", "translate", "render"] as const;
export type StageName = (typeof STAGES)[number];

export interface StageRun {
  stage: StageName;
  status: "pending" | "running" | "done" | "failed";
  attempts: number;
  error: string | null;
  stats: Record<string, unknown> | null;
}

export interface Episode {
  id: number;
  title: string;
  source_kind: "url" | "file";
  source: string;
  series: string | null;
  season: number | null;
  episode: number | null;
  translator: Translator;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  error: string | null;
  created_at: string;
  stages: StageRun[];
}

export interface Live {
  stage: StageName;
  pct: number;
  msg: string;
}

export interface GpuStatus {
  paused: boolean;
  holding: string | null;
  vram_used_mb: number | null;
}

export type Translator = "qwen9b" | "moe";

export interface PreviewItem {
  url: string;
  title: string;
  duration: number | null;
  channel: string | null;
  parsed: { series: string | null; season: number | null; episode: number | null };
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error ?? res.statusText);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  episodes: () =>
    call<{ episodes: Episode[]; live: Record<string, Live>; gpu: GpuStatus }>("/api/episodes"),
  preview: (input: string) =>
    call<{ title: string | null; items: PreviewItem[] }>("/api/preview", {
      method: "POST",
      body: JSON.stringify({ input }),
    }),
  queueUrls: (
    items: { url: string; title: string; series?: string | null; season?: number | null; episode?: number | null }[],
    translator: Translator,
  ) => call<{ ids: number[] }>("/api/episodes", { method: "POST", body: JSON.stringify({ items, translator }) }),
  queueFile: (path: string, translator: Translator) =>
    call<{ ids: number[] }>("/api/files", { method: "POST", body: JSON.stringify({ path, translator }) }),
  retry: (id: number) => call<void>(`/api/episodes/${id}/retry`, { method: "POST" }),
  cancel: (id: number) => call<void>(`/api/episodes/${id}/cancel`, { method: "POST" }),
  setPaused: (paused: boolean) =>
    call<GpuStatus>("/api/gpu", { method: "POST", body: JSON.stringify({ paused }) }),
};

export type ServerEvent =
  | { type: "episode"; episode: Episode; live: Live | null }
  | { type: "progress"; id: number; live: Live | null }
  | { type: "gpu"; gpu: GpuStatus };

/** Subscribes to server-sent events; reconnects automatically. Returns an unsubscribe function. */
export function subscribe(onEvent: (e: ServerEvent) => void): () => void {
  const source = new EventSource("/api/events");
  source.onmessage = (msg) => onEvent(JSON.parse(msg.data));
  return () => source.close();
}
