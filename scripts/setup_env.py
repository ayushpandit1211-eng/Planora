"""Create local credentials without overwriting an existing .env."""
from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print("Existing .env preserved.")
        return
    text = (root / ".env.example").read_text(encoding="utf-8")
    text = text.replace("POSTGRES_PASSWORD=change-me-postgres", "POSTGRES_PASSWORD=" + secrets.token_hex(24))
    text = text.replace("QS_APP_PASSWORD=change-me-app", "QS_APP_PASSWORD=" + secrets.token_hex(24))
    text = text.replace("PRIVACY_HMAC_KEY=\n", "PRIVACY_HMAC_KEY=" + secrets.token_hex(32) + "\n")
    with target.open("x", encoding="utf-8") as f:
        f.write(text)
    print("Created .env with local credentials. Keep it out of Git.")


if __name__ == "__main__":
    main()
