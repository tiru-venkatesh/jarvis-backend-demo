#!/usr/bin/env python3
"""
run_dev.py - one-command local launcher for the AURA command center.

Why this exists: jarvis_server.py reads its config (GROQ_API_KEY, AURA_USER,
etc.) from real environment variables, but for local use it's tedious to
export them every time. This script:

  1. loads key=value pairs from the .env file next to it (without overriding
     anything already set in your real environment -- setdefault, so a real
     deployment's env always wins), and
  2. starts the server.

AURA is single-user with no login system, so there's no auth mode to switch
on here -- jarvis_server.py always serves the one local user (AURA_USER,
default "local").

Run it from anywhere:  python run_dev.py
Then open:             http://localhost:8787/

In production you'd still typically run jarvis_server.py under gunicorn
rather than this launcher's app.run(), but there's no auth to configure.
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

# cd into this folder so relative paths (aura_memory.db, dashboard.html,
# knowledge/, chroma_db/) resolve the same way they do for jarvis_server.py.
os.chdir(HERE)

import jarvis_server as s  # noqa: E402  (must follow the env setup above)

if __name__ == "__main__":
    print("AURA command center (dev launcher)")
    print(f"  user:      {s.LOCAL_USER}  (single-user, no login)")
    print(f"  groq key:  {'set' if s.core.GROQ_API_KEY else 'MISSING -- chat will error'}")
    print("  open:      http://localhost:8787/\n")
    s.app.run(host="127.0.0.1", port=8787, debug=False)
