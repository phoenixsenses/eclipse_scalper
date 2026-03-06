import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ConfigEntry } from "../api/types";
import { useDashboardAuth } from "../context/AuthContext";
import PageGuide from "../components/PageGuide";

export default function Settings() {
  const { auth, setAuth } = useDashboardAuth();
  const [entries, setEntries] = useState<ConfigEntry[]>([]);
  const [health, setHealth] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.config()
      .then(setEntries)
      .catch((e) => setErr(String(e)));
    api.health()
      .then((d) => setHealth(d.status))
      .catch(() => setHealth("error"));
  }, []);

  const envFile = entries.filter((e) => e.source === "env_file");
  const runtime = entries.filter((e) => e.source === "runtime_env");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 700 }}>
      <PageGuide
        icon="⚙️"
        titleTr="Ayarlar ve Erisim"
        titleEn="Settings & Access"
        subtitleTr="API anahtari, operator kimligi ve runtime override bilgilerini buradan yonet."
        subtitleEn="Manage API key, operator identity, and runtime overrides here."
        items={[
          {
            icon: "🔑",
            titleTr: "Kimlik",
            titleEn: "Identity",
            descTr: "API key + operator basliklari write endpoint erisimini belirler.",
            descEn: "API key + operator headers control write endpoint access.",
          },
          {
            icon: "🧱",
            titleTr: "Env Kaynagi",
            titleEn: "Env Source",
            descTr: ".env dosyasi ile runtime override degerlerini karsilastir.",
            descEn: "Compare .env values against runtime overrides.",
          },
          {
            icon: "❤️",
            titleTr: "Backend Sagligi",
            titleEn: "Backend Health",
            descTr: "Status ok degilse diger sayfalarda veri eksik gelebilir.",
            descEn: "If status is not ok, other pages can show stale or missing data.",
          },
        ]}
      />

      <div className="card">
        <div className="card-title">Backend Health</div>
        {health === null
          ? <span style={{ color: "var(--muted)" }}>Checking...</span>
          : health === "ok"
            ? <span className="badge badge-green">OK</span>
            : <span className="badge badge-red">{health}</span>
        }
      </div>

      {err && (
        <div className="card" style={{ borderColor: "var(--red)" }}>
          <div style={{ color: "var(--red)" }}>{err}</div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Dashboard Auth Context</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            API Key
            <input
              type="password"
              value={auth.apiKey}
              onChange={(e) => setAuth((p) => ({ ...p, apiKey: e.target.value }))}
              placeholder="X-Api-Key"
              style={{ marginLeft: 6, width: 260, background: "transparent", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 6px" }}
            />
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            Operator
            <input
              value={auth.operator}
              onChange={(e) => setAuth((p) => ({ ...p, operator: e.target.value }))}
              placeholder="X-Operator"
              style={{ marginLeft: 6, width: 180, background: "transparent", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 6px" }}
            />
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12 }}>
            Role
            <select
              value={auth.role}
              onChange={(e) => setAuth((p) => ({ ...p, role: e.target.value as "viewer" | "operator" | "admin" }))}
              style={{ marginLeft: 6, background: "transparent", color: "var(--text)", border: "1px solid var(--border)" }}
            >
              <option value="viewer">viewer</option>
              <option value="operator">operator</option>
              <option value="admin">admin</option>
            </select>
          </label>
          <label style={{ color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={auth.idempotency}
              onChange={(e) => setAuth((p) => ({ ...p, idempotency: e.target.checked }))}
            />
            Auto idempotency key for write requests
          </label>
        </div>
      </div>

      <div className="card">
        <div className="card-title">.env Configuration</div>
        {envFile.length === 0
          ? <div style={{ color: "var(--muted)" }}>No .env file found</div>
          : (
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Value</th>
                  <th>Sensitive</th>
                </tr>
              </thead>
              <tbody>
                {envFile.map((e) => (
                  <tr key={e.key}>
                    <td style={{ fontWeight: 600 }}>{e.key}</td>
                    <td style={{ color: e.sensitive ? "var(--muted)" : "var(--text)" }}>
                      {e.value || <span style={{ color: "var(--muted)" }}>(empty)</span>}
                    </td>
                    <td>
                      {e.sensitive
                        ? <span className="badge badge-yellow">masked</span>
                        : <span className="badge badge-gray">no</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      {runtime.length > 0 && (
        <div className="card">
          <div className="card-title">Runtime Overrides</div>
          <table>
            <thead>
              <tr>
                <th>Key</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {runtime.map((e) => (
                <tr key={e.key}>
                  <td style={{ fontWeight: 600 }}>{e.key}</td>
                  <td>{e.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
