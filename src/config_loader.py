"""Tiny .env loader (stdlib‑only).

Reads `KEY=VALUE` lines from `.env` in the project root and pushes them
into `os.environ` only if they aren't already set there (so a real
shell‑exported value always wins).

Idempotent – safe to call once at startup.
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # existing env var wins
        os.environ.setdefault(k, v)

# load on import
load_env()