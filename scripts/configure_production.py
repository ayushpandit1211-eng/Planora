"""Create private production configuration without printing credentials."""
import argparse
import getpass
import os
from pathlib import Path
import re
import secrets


def configuration(domain, email, user, password):
    import bcrypt
    if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", domain) or "." not in domain:
        raise ValueError("Use a DNS hostname without scheme, path, port or whitespace")
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email):
        raise ValueError("Enter a valid certificate contact email")
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", user):
        raise ValueError("Username must contain 3-40 letters, digits, underscores or hyphens")
    if not 16 <= len(password.encode()) <= 72:
        raise ValueError("Use a unique login password of 16-72 UTF-8 bytes")
    return {"APP_DOMAIN": domain, "ACME_EMAIL": email, "DEMO_USER": user,
            "DEMO_PASSWORD_HASH": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            "POSTGRES_PASSWORD": secrets.token_hex(32), "QS_APP_PASSWORD": secrets.token_hex(32),
            "PRIVACY_HMAC_KEY": secrets.token_hex(32), "QS_API_KEY": secrets.token_hex(32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--user", default="reviewer")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / ".env.production"
    if path.exists():
        raise SystemExit(".env.production already exists; it was preserved")
    password = getpass.getpass("Private demo login password (16-72 bytes): ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords differ; nothing written")
    values = configuration(args.domain, args.email, args.user, password)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf8", newline="\n") as f:
        for key, value in values.items():
            f.write(f"{key}='{value}'\n")
    print("Created .env.production. Keep it private and back it up securely.")


if __name__ == "__main__":
    main()
