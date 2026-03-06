import json, pathlib, pprint

p = pathlib.Path("reports").glob("*forward*.json")
p = next(p, None)

if not p:
    print("No forward validation JSON found")
else:
    print("Inspecting:", p)
    d=json.loads(p.read_text(encoding="utf-8"))

    # walk entire tree looking for net_per_attempt
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k,v in obj.items():
                if "net_per_attempt" in k or "attempt" in k:
                    print(prefix+k, "=", v)
                walk(v, prefix+k+".")
        elif isinstance(obj, list):
            for i,v in enumerate(obj[:3]):
                walk(v, prefix+f"[{i}].")

    walk(d)
