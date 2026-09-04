#!/usr/bin/env python3
"""
run_dev.py - one-command local launcher for the AURA command center.

Why this exists: jarvis_server.py reads its config (GROQ_API_KEY, and the
AURA_DEV_USER dev-mode switch) from real environment variables, but for local
use it's tedious to export them every time. This script:

  1. loads key=value pairs from the .env file next to it (without overriding
     anything already set in your real environment -- setdefault, so a real
     deployment's env always wins),
  2. turns on local dev mode (AURA_DEV_USER=local) unless you've configured
     Supabase auth, so the dashboard works with no login/token, and
  3. starts the server.

Run it from anywhere:  python run_dev.py
Then open:             http://localhost:8787/

This is a DEV convenience only. In production you run jarvis_server.py under
gunicorn with SUPABASE_JWT_SECRET set, which disables dev mode entirely.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            # Real environment wins over .env, matching normal dotenv behavior.
            os.environ.setdefault(key, val)


# Order matters: load .env BEFORE importing the server, because jarvis.py reads
# GROQ_API_KEY into a module global at import time.
_load_dotenv()

# Default to local, no-auth dev mode -- but only when Supabase isn't configured.
if not os.environ.get("SUPABASE_JWT_SECRET"):
    os.environ.setdefault("AURA_DEV_USER", "local")

# cd into this folder so relative paths (aura_memory.db, dashboard.html,
# knowledge/, chroma_db/) resolve the same way they do for jarvis_server.py.
os.chdir(HERE)

import jarvis_server as s  # noqa: E402  (must follow the env setup above)

if __name__ == "__main__":
    dev = getattr(s, "DEV_MODE", False)
    print("AURA command center (dev launcher)")
    print(f"  dev mode:  {dev}  (auth {'bypassed as user local' if dev else 'required'})")
    print(f"  groq key:  {'set' if s.core.GROQ_API_KEY else 'MISSING -- chat will error'}")
    print("  open:      http://localhost:8787/\n")
    s.app.run(host="127.0.0.1", port=8787, debug=False)
