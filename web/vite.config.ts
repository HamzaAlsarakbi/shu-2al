import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [svelte()],
  // `npm run dev` talks to a local `shu2al serve`.
  server: { proxy: { "/api": "http://127.0.0.1:8080" } },
});
