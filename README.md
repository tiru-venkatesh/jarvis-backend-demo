# AURA -- personal AI command center

AURA is a personal AI assistant with persistent memory, a per-user knowledge base
(RAG), an Intent Engine (goals -> executable plans), tool-calling (email, search,
memory, plans), and tiered email permissions. It runs as a CLI (`jarvis.py`) or a
web server + browser command center (`jarvis_server.py` + `dashboard.html`).

## Quick start (local command center)
```bash
pip install -r requirements.txt   # flask, flask-cors, PyJWT, chromadb, groq, etc.
python run_dev.py                 # loads .env, turns on local dev mode, starts server
```
Then open **http://localhost:8787/** -- the two-column command center: chat on the
left, Active Plans / Memory / Email Permission on the right.

`run_dev.py` reads `GROQ_API_KEY` from `.env` and enables a no-login **dev mode**
(`AURA_DEV_USER=local`) so the dashboard works without a token. Dev mode only turns
on when `SUPABASE_JWT_SECRET` is *not* set, so production (which sets it) always
requires real auth.

CLI instead of the web UI:
```bash
python jarvis.py                  # same agent + tools in the terminal
```

## New capabilities
- **Per-user knowledge base** (`rag.py`): each user gets a physically separate
  Chroma collection (`kb_<id>`), so one user's docs can never surface for another.
  Build/query: `python rag.py build [user_id] [--force]`, `python rag.py query [user_id] "..."`.
- **Intent Engine** (`memory_store.py` plans layer): turns a goal into a persisted
  step checklist. Tools `create_plan` / `complete_plan_step`; CLI `/plans` and
  `/plan <id>`; API `GET /api/plans`, `GET/DELETE /api/plans/<id>`. A plan
  auto-completes when all steps are done/skipped.
- **Command center** (`dashboard.html`): served at `/`, same-origin (no CORS/token
  in dev mode). Confirm cards for email sends, confidence rings for memory facts,
  0-4 email permission control.

> **Note:** `aura_memory.db` may contain seeded demo data for user `local`
> (1 plan, 2 facts). Clear it from the UI (trash / "Clear all" buttons) or delete
> `aura_memory.db` to start fresh.

---

# Jarvis outreach package -- ready to run

## Files
| File | What it's for |
|---|---|
| `jarvis.py` | Main agent: chat, email drafting/sending, `/bulk`, `/bulk-role` |
| `rag.py` | Local knowledge base (ChromaDB + sentence-transformers) that Jarvis searches for context |
| `templates.json` | Role-matched middle paragraphs for `/bulk-role` (ml, nlp, cv, systems, data, robotics) |
| `recipients.csv` | IIT KGP professor list with names + matched research type, ready for `/bulk-role` |
| `emails.txt` | Plain address list, for `/bulk` (same email to everyone) |
| `outreach_body.txt` | Short-form email body, for `/bulk emails.txt \| Subject \| @outreach_body.txt` |
| `knowledge/about_tiru.txt` | Starter RAG doc -- add resume.txt, projects_detailed.txt etc. alongside it |
| `requirements.txt` | Python deps |

## One-time setup
1. `pip install -r requirements.txt`
2. `export GROQ_API_KEY="gsk_..."` (console.groq.com/keys)
3. Drop `tiru-venkatesh.pdf` and `credentials.json` (Gmail OAuth client secret) next to `jarvis.py`
4. `python rag.py build` -- indexes `knowledge/`
5. `python jarvis.py` -- first send opens a browser for Gmail OAuth, caches `token.json`

## Sending outreach
- **One email:** just ask Jarvis in chat ("email Prof. Rudra about an internship") -- it drafts, shows a preview with the resume attached, and asks y/N before sending.
- **Same email to a list:** `/bulk emails.txt | Subject line | @outreach_body.txt`
- **Role-matched, personalized per professor:** `/bulk-role recipients.csv` -- picks the right paragraph per person from `templates.json`, rotates between the 2 variants per type so consecutive same-type recipients don't get identical wording.

## About `recipients.csv`
Types were matched from each professor's public research page:

| Email | Matched type | Why |
|---|---|---|
| adway@cai.iitkgp.ac.in | data | Responsible AI, AI for social good, healthcare/data-driven modeling |
| mahesh.mohan@cai.iitkgp.ac.in | cv | Signal processing + deep learning + computer vision |
| krudra@cai.iitkgp.ac.in | nlp | Information retrieval, responsible AI, NLP |
| ksreddy@ai.iitkgp.ac.in | ml | Multi-armed bandits, reinforcement learning, information theory |
| jiaul@ai.iitkgp.ac.in | nlp | Information retrieval / NLP |
| amaiti@ai.iitkgp.ac.in | systems | **Couldn't verify this one's research area -- fix before sending** |
| shreya.ghosh@ai.iitkgp.ac.in | cv | Computer vision, affective computing, generative AI |
| somdyuti@cai.iitkgp.ac.in | cv | Video/image quality, signal processing |

`aravahimesh@gmail.com` from your `emails.txt` was left out of `recipients.csv` -- it's a personal Gmail, not an IITKGP faculty address, so it doesn't fit the professor template. Send it separately if it's someone specific, or tell me who it is and I'll fold it in.

## Safety net
Nothing sends without your explicit `y` at the confirmation prompt -- single sends, `/bulk`, and `/bulk-role` all show a full preview (recipients, subject, body, whether the resume attached) first.
