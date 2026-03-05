# Frontend Design Plan (TR/EN) - Eclipse Dashboard

## Goal
Dashboard'u teknik detaya hakim olmayan bir operatorun da hizli okuyabilecegi hale getirmek:
- nereden baslayacagini bilsin,
- terimleri TR/EN gorebilsin,
- kritik sinyalleri renk ve ikonla ayirsin,
- debug akisini adim adim takip etsin.

## Design Principles
1. Durum once gelir: LIVE/DEGRADED/STALE bir bakista okunur.
2. Metin iki dilli: TR once, EN destek satiri.
3. Kullanim yolu gorunur: her sayfada signpost kartlari.
4. Hata ayiklama adimlari sabit: Guide -> Action -> Evidence.
5. Operasyon guvenligi: write aksiyonlari role-based net gorunur.

## Phase 1 - Visual Foundation
Scope:
- Renk tokenlarini operasyon odakli hale getir.
- Header/nav iki dilli ve ikonlu olsun.
- Global legend satiri ekle.

Done:
- `src/index.css`: yeni palette, card vurgulari, legend/signpost stilleri.
- `src/components/Layout.tsx`: TR/EN nav + global legend + security chip.

## Phase 2 - Page Guides & Signposts
Scope:
- Her ana sayfaya "Nasil kullanirim" rehberi ekle.
- 3 kartlik mini akis: neye bak, nasil yorumla, ne aksiyon al.

Done:
- `src/components/PageGuide.tsx` eklendi.
- `src/pages/Overview.tsx`, `src/pages/Logs.tsx`, `src/pages/Trades.tsx`, `src/pages/Debug.tsx`, `src/pages/Settings.tsx` guide ile guncellendi.

## Phase 3 - Terminology & Operator Help
Scope:
- Teknik terimleri plain-language acikla.
- Table basliklarinda kritik alanlarin anlami net olsun.

Status:
- Global legend aktif.
- Logs/Overview/Settings metinleri sadeleştirildi.
- Tablo baslik tooltipleri eklendi (`TermTip`):
  - Overview: hit rate, baseline delta, confidence, regime.
  - Trades: signal/stability/quality kolonlari.
  - Logs: level filtresi aciklamasi.

## Phase 4 - Debug Session UX
Scope:
- Guided debug ve incident triage akisini daha az adimla yonet.
- Role lock durumunu daha belirgin yap.

Status:
- Debug sayfasinda signpost + guided session akisi var.
- Sonraki adim: "Recommended next action" auto-focus (planned).

## Phase 5 - Validation & Accessibility
Checks:
- Frontend typecheck pass
- smoke tests pass
- 1366x768 + mobile responsive check

Operational checklist:
1. Overview'de Runtime Status karti dolu mu?
2. Logs'da dosya secip satirlar gorunuyor mu?
3. Debug'da guided runbook butonlari role'e gore kilitleniyor mu?
4. Settings'te API key/operator/role degisikligi uygulanabiliyor mu?

## Next Iteration Backlog
1. Table column tooltips (TR/EN micro explanations)
2. Guided empty states ("once su endpoint'i calistir")
3. Severity-driven highlight lanes for logs
4. Keyboard shortcuts (`/` search, `g o` overview, `g l` logs)
