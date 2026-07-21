$ErrorActionPreference = "Continue"
Set-Location "D:\eclipse_scalper"

& "C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe" `
  -u "tools\s34_shadow_paper_runner.py" `
  --loop `
  --interval-sec 60 `
  --regime-filter-enabled `
  --regime-min-trend-pct 1.0 `
  --regime-min-range-pct 2.5 `
  --regime-min-buy-liq-notional 5000000 `
  --regime-min-agg-trade-count 250000 `
  --quality-gate-enabled `
  --quality-gate-min-eclipse 42.0 `
  *>> "logs\s34_shadow_paper_runner.wrapper.log"
