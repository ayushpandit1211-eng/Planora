"""Compare two contract_harness outputs. Server-generated request ids (qs-<hex>) are normalised."""
import difflib
import json
import re
import sys


def norm(c):
    path = re.sub(r"qs-[0-9a-f]{16}", "qs-<request_id>", c["path"])
    return (c["m"], path, json.dumps(c["json"], sort_keys=True), c["content_bytes"], tuple(c["headers"]))


a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = True
for k in a:
    x, y = [norm(c) for c in a[k]["calls"]], [norm(c) for c in b[k]["calls"]]
    same = x == y and not b[k]["exceptions"]
    ok &= same
    print(f"{k:15s} calls before={len(x):4d} after={len(y):4d} identical={same} exceptions={b[k]['exceptions']}")
    if x != y:
        for line in list(difflib.unified_diff(list(map(str, x)), list(map(str, y)), lineterm=""))[:20]:
            print("   ", line)
print("CONTRACTS IDENTICAL" if ok else "CONTRACT DIFFERENCES FOUND")
sys.exit(0 if ok else 1)
