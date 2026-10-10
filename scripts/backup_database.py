"""Back up the production database to a private local PostgreSQL custom-format dump."""
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "--env-file", ".env.production", "-f", "compose.production.yml"]


def main():
    directory = ROOT / "backups"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / ("planora-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".dump")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            subprocess.run([*COMPOSE, "exec", "-T", "db", "pg_dump", "-U", "postgres", "-d",
                            "querysentinel", "--format=custom"], cwd=ROOT, stdout=f, check=True)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print("Database backup saved:", path)
    print("Also retain .env.production securely; copy backups off the server.")


if __name__ == "__main__":
    main()
