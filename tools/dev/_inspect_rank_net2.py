import json, pathlib, re

p = pathlib.Path("reports/PASSIVE_POCKET_RANKING_V2_7D_fastpass.json")
d = json.loads(p.read_text(encoding="utf-8"))
r = d["ranking"][0]

def walk(obj, prefix=""):
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            out += walk(v, prefix + "." + str(k) if prefix else str(k))
    elif isinstance(obj, list):
        for i,v in enumerate(obj[:5]):
            out += walk(v, prefix + f"[{i}]")
    else:
        if isinstance(obj,(int,float)) and re.search(r"net|pnl|profit|edge|bps", prefix.lower()):
            out.append((prefix, obj))
    return out

hits = walk(r.get("evals"))
print("net/pnl-like numeric fields found (first pocket):")
for path,val in hits[:80]:
    print(path, "=", val)
