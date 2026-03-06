# Eclipse Scalper - Basit Kullanim Notlari (TR)

Bu dosya "hizli calistir / durdur / kontrol et" icin hazirlandi.

## 0) Klasore gir
```powershell
cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
```

## 1) Paper Trade Baslat
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_paper_trading.ps1
```

Beklenen:
- Environment validation `PASS`
- Preflight `PASS`
- Bootstrap loglari akmaya baslar

## 2) Paper Trade Durdur
- Calistigi pencerede: `Ctrl + C`

Gerekirse zorla kapat:
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "execution\.bootstrap|tools\.collection_watchdog" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 3) Dashboard (Frontend + Backend birlikte)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_dashboard.ps1
```

Ac:
- `http://localhost:5173/`

Not:
- Bu komut backend + frontend birlikte acmaya calisir.

## 4) Sadece Dashboard Backend
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_dashboard_backend.ps1
```

Health kontrol:
```powershell
Invoke-WebRequest http://127.0.0.1:8765/api/health -UseBasicParsing
```

## 5) Sadece Dashboard Frontend
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_dashboard_frontend.ps1
```

## 6) Microstructure / Data Collection Calisiyor mu?
Collector log son satir:
```powershell
Get-Content .\logs\microstructure_collector.log -Tail 20
```

DB yaziliyor mu:
```powershell
Get-Item .\data\microstructure.db | Select-Object LastWriteTime,Length
```

Son veri kac saniye eski:
```powershell
@"
import sqlite3, time
con=sqlite3.connect("data/microstructure.db"); cur=con.cursor()
now=int(time.time()*1000)
for t in ["agg_trades","mark_prices"]:
    mx=cur.execute(f"select max(ts_ms) from {t}").fetchone()[0]
    print(t, "age_sec=", round((now-mx)/1000,3))
con.close()
"@ | python -
```

Yorum:
- `age_sec` genelde birkac saniye ise saglikli.

## 7) Cift process var mi? (Normal / anormal)
Detay gor:
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "execution\.bootstrap|tools\.collection_watchdog" } | Select-Object ProcessId,ParentProcessId,CommandLine
```

Leaf (gercek worker) gor:
```powershell
$targets = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "execution\.bootstrap|tools\.collection_watchdog" }
$targets | Where-Object { -not ($targets.ParentProcessId -contains $_.ProcessId) } | Select-Object ProcessId,ParentProcessId,CommandLine
```

Not:
- Parent + child birlikte 4 satir gorunebilir.
- Leaf tarafinda 2 worker gorunmesi (bootstrap worker + watchdog worker) normaldir.

## 8) Sik Hata - Hizli Cozum

### A) `Backend unavailable` / `ECONNREFUSED 127.0.0.1:8765`
1. Backend'i ac:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_dashboard_backend.ps1
```
2. Sonra dashboard:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_dashboard.ps1
```

### B) PowerShell path hatasi (`Windows 11` bosluk sorunu)
Her zaman tirnak kullan:
```powershell
cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
```

### C) Bot acilmiyor / processler karisik
Temiz kapat:
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "execution\.bootstrap|tools\.collection_watchdog" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```
Sonra tekrar baslat:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_paper_trading.ps1
```

## 9) Gunluk Basit Rutin
1. Paper trade ac
2. Dashboard ac (`run_dashboard.ps1`)
3. Overview + Control Tower'da health kontrol et
4. Gun sonunda log/rapor kontrol et

## 10) Live Monitor Test Paketi (tek komut)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_live_monitor_tests.ps1
```

Bu komut sirayla:
- backend `py_compile`
- backend live metrics pytest
- frontend typecheck
- frontend `LiveMonitor` smoke test

Hepsi gecerse `ALL PASS` yazar.
