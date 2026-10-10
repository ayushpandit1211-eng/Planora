"""Run backend and dashboard against an already configured local PostgreSQL server."""
from pathlib import Path
import os
import subprocess
import sys
import time

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = {**{k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}, **os.environ}
    env.setdefault("POSTGRES_HOST", "127.0.0.1")
    env.setdefault("BACKEND_URL", "http://127.0.0.1:8000")
    env["PYTHONUTF8"] = "1"
    commands = [
        ["-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"],
        ["-m", "streamlit", "run", "dashboard/app.py", "--server.address", "127.0.0.1", "--server.port", "8501"],
    ]
    children = []
    try:
        for args in commands:
            children.append(subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env))
        print("Dashboard: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs", flush=True)
        print("Press Ctrl+C to stop both services.", flush=True)
        while all(p.poll() is None for p in children):
            time.sleep(1)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == "__main__":
    raise SystemExit(main())
