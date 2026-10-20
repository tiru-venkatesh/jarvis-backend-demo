#!/usr/bin/env python3
"""
make_gmail_token.py - run this ONCE on your own computer to create token.json,
which lets the Render server send Gmail over HTTPS (the Gmail API).

Steps:
  1. Put your OAuth client file (Desktop app type) next to this script as
     credentials.json.
  2. pip install google-auth-oauthlib google-api-python-client
  3. python make_gmail_token.py     -> a browser opens, sign in, click Allow
  4. Upload BOTH credentials.json and token.json to Render as Secret Files.
Never commit either file (they're already in .gitignore).
"""
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
HERE = os.path.dirname(os.path.abspath(__file__))
CRED = os.path.join(HERE, "credentials.json")
OUT = os.path.join(HERE, "token.json")

if not os.path.exists(CRED):
    sys.exit(f"credentials.json not found at {CRED}. Download it from Google Cloud "
             "Console -> APIs & Services -> Credentials (OAuth client, Desktop app).")

flow = InstalledAppFlow.from_client_secrets_file(CRED, SCOPES)
creds = flow.run_local_server(port=0)
with open(OUT, "w") as f:
    f.write(creds.to_json())
print(f"Saved {OUT}. Upload it (and credentials.json) to Render -> Environment -> Secret Files.")
