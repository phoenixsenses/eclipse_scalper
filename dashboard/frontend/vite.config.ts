import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function resolveProxyTarget(env: Record<string, string>, cwd: string): string {
  if (env.VITE_PROXY_TARGET) {
    return env.VITE_PROXY_TARGET;
  }

  const runtimePath = path.resolve(cwd, "../runtime/dashboard_backend.json");
  try {
    if (fs.existsSync(runtimePath)) {
      const raw = fs.readFileSync(runtimePath, "utf-8");
      const parsed = JSON.parse(raw) as { host?: string; port?: number };
      const host = parsed.host || "127.0.0.1";
      const port = parsed.port || 8765;
      return `http://${host}:${port}`;
    }
  } catch {
    // ignore pointer parse errors and use default target
  }

  return "http://127.0.0.1:8765";
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = resolveProxyTarget(env, process.cwd());
  const proxyAgent = new http.Agent({ keepAlive: true, maxSockets: 64 });

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
          agent: proxyAgent,
        },
      },
    },
  };
});
