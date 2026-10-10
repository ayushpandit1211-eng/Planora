"""Restore a dump into a new disposable database; never replace the live database."""
import argparse
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "--env-file", ".env.production", "-f", "compose.production.yml",
           "exec", "-T", "db"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()
    if not args.dump.is_file():
        raise SystemExit("Backup file not found")
    name = "qs_restore_" + uuid.uuid4().hex
    subprocess.run([*COMPOSE, "createdb", "-U", "postgres", name], cwd=ROOT, check=True)
    try:
        with args.dump.open("rb") as f:
            subprocess.run([*COMPOSE, "pg_restore", "-U", "postgres", "-d", name, "--exit-on-error"],
                           cwd=ROOT, stdin=f, check=True)
        subprocess.run([*COMPOSE, "psql", "-U", "postgres", "-d", name, "-v", "ON_ERROR_STOP=1",
                        "-c", "SELECT count(*) AS restored_orders FROM orders;"], cwd=ROOT, check=True)
        print("PASS: backup restored into a disposable database")
    finally:
        subprocess.run([*COMPOSE, "dropdb", "-U", "postgres", name], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
