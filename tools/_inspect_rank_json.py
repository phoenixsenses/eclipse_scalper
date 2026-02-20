import json, pathlib
p = pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_fastpass.json")
d = json.loads(p.read_text(encoding="utf-8"))

print("Top-level keys:", sorted(d.keys())[:50])

lst_key = None
lst = None
for k, v in d.items():
    if isinstance(v, list) and v and isinstance(v[0], dict):
        lst_key = k
        lst = v
        break

print("Candidate list key:", lst_key, "len=", len(lst) if lst else None)
if lst:
    print("Sample item keys:", sorted(lst[0].keys())[:120])
