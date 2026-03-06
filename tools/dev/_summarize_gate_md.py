import re, pathlib

paths = [
  "reports/PV_NO_GATE_fee1_adv1.md",
  "reports/PV_GATE_int3250_fee1_adv1.md",
  "reports/PV_GATE_int3500_fee1_adv1.md",
  "reports/PV_GATE_spread0p0004_fee1_adv1.md",
  "reports/PV_GATE_spread0p0003_fee1_adv1.md",
]

KEYS = [
  "pass_rate", "pass_count",
  "median_net_per_attempt", "median_filled_avg_net",
  "attempt_fill_rate_median", "attempts_per_min_median",
  "insufficient_fill_rate",
  "val_attempts_before_gate", "val_attempts_after_gate", "val_filled_after_gate",
  "net_per_attempt_after_gate",
]

def find_numbers(text, key):
  # matches: key: 0.123  OR key = 0.123 OR key|0.123 (markdown tables)
  pat = re.compile(rf"{re.escape(key)}[^0-9eE+\-]*([+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)", re.IGNORECASE)
  m = pat.search(text)
  return float(m.group(1)) if m else None

for p in paths:
  fp = pathlib.Path(p)
  if not fp.exists():
    continue
  txt = fp.read_text(encoding="utf-8", errors="ignore")
  print("\n===", p, "===")
  for k in KEYS:
    v = find_numbers(txt, k)
    if v is not None:
      print(f"{k}: {v}")
