"""Report real dependency status and exit nonzero when the core demo is incomplete."""
import argparse
import json
import os
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    headers = {"x-api-key": os.environ["QS_API_KEY"]} if os.environ.get("QS_API_KEY") else {}
    try:
        with urlopen(Request(args.url.rstrip("/") + "/api/v1/status", headers=headers), timeout=20) as r:
            status = json.load(r)
    except Exception as exc:
        print("Backend unavailable:", type(exc).__name__)
        return 1
    checks = {name: bool(status.get(name, {}).get("ok")) for name in
              ("postgresql", "hypopg", "pg_stat_statements", "privacy_vault", "langgraph")}
    checks["trained GNN"] = status.get("gnn", {}).get("mode") == "trained"
    checks["pg_stat_statements"] &= bool(status.get("pg_stat_statements", {}).get("preloaded"))
    checks["trained RL"] = status.get("rl", {}).get("mode") == "trained_policy"
    for name, ok in checks.items():
        print(("PASS" if ok else "UNAVAILABLE") + "  " + name)
    print("LLM (optional):", "enabled" if status.get("llm", {}).get("enabled") else "disabled")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
