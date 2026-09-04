#!/usr/bin/env python3
"""
memory_store.py - persistent, per-user memory for AURA.

This is what turns Jarvis from "a chatbot that forgets everything on
restart" into "a personal system that remembers." Two things live here,
both scoped by user_id so it stays multi-tenant:

  1. Conversation history - so a server restart or redeploy doesn't wipe
     everyone's chat, the way the old in-memory _user_sessions dict did.
  2. Long-term facts - things worth remembering *across* separate
     conversations (preferences, ongoing projects, recurring contacts).
     Added via a remember_fact tool the model can call, same pattern as
     send_email and search_knowledge already use.

Storage is one SQLite file (aura_memory.db by default) - no new infra,
works fine on Render's free tier, and the table shapes map directly onto
Postgres columns later if/when this needs to run with more than one
worker process.
"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("AURA_DB_PATH", "aura_memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT,
    tool_call_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.7,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_confirmed REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, id);

CREATE TABLE IF NOT EXISTS permissions (
    user_id TEXT PRIMARY KEY,
    send_email_level INTEGER NOT NULL DEFAULT 2
);

CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_user ON plans(user_id, id);

CREATE TABLE IF NOT EXISTS plan_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    description TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT 'manual',
    status TEXT NOT NULL DEFAULT 'pending',
    note TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id, position);
"""

# Permission levels for tools that act in the real world (item 11 in the
# AURA spec). Only send_email uses this so far -- extend the same pattern
# to future tools (calendar writes, file deletes, etc.) rather than
# inventing a new scheme per tool.
#   0 = disabled        tool refuses, tells the user why
#   1 = draft-only       tool composes but never sends -- always returns
#                        the draft as text, no confirm card, nothing goes out
#   2 = confirm required  (default, current behavior) -- confirm card, user
#                        clicks Send
#   3 = auto-send        sends immediately for a *single* recipient, then
#                        tells the user what it sent (a log, not a question)
#   4 = same as 3 today; kept as a distinct level so a future "fully
#       autonomous, don't even tell me until the daily briefing" mode has
#       somewhere to live without renumbering everything else
#
# NOTE: bulk sends (/bulk, /bulk-role) do NOT read this table -- they
# always go through the confirm-card path regardless of level. Spam/
# deliverability/consent risk from a bulk send isn't something a user
# should be able to configure away by accident, so it's a hard rule in
# code, not a setting.
DEFAULT_SEND_EMAIL_LEVEL = 2



@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(SCHEMA)


# --- conversation persistence ----------------------------------------------

def append_message(user_id, role, content, tool_calls=None, tool_call_id=None):
    """tool_calls, if present, is the raw list from the Groq response --
    stored as JSON so a reload can hand it straight back to the API."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, tool_calls, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                role,
                content or "",
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                time.time(),
            ),
        )


def load_recent_messages(user_id, limit=40):
    """Most recent `limit` turns, oldest first -- ready to drop straight
    into the messages[] list a Groq chat-completions call expects."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    rows = list(reversed(rows))
    out = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["tool_calls"]:
            msg["tool_calls"] = json.loads(r["tool_calls"])
        if r["tool_call_id"]:
            msg["tool_call_id"] = r["tool_call_id"]
        out.append(msg)
    return out


def clear_conversation(user_id):
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))


def pop_last_message(user_id):
    """Delete the most recent message row for a user -- used to roll back
    a user turn that was persisted right before a Groq call failed, so a
    retry doesn't end up duplicating it in history."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))


# --- long-term facts, with evidence tracking -------------------------------
#
# A flat list of strings can't tell "mentioned once in passing" from
# "confirmed independently five times" apart -- both just look like a
# line in the prompt. Confidence + evidence_count fix that: a fact you
# say once starts at 0.7 confidence; saying something that reinforces it
# again nudges it up (diminishing returns, capped at 0.97) instead of
# creating a duplicate row. Nothing here does semantic similarity -- match
# is deliberately simple (normalized substring) so it only reinforces
# near-identical restatements, not "vaguely related" facts. That's a
# reasonable v1: false negatives (two facts that should merge but don't)
# are harmless, false positives (two different facts merged into one)
# would quietly corrupt memory.

def _normalize(s):
    return " ".join(s.lower().split())


def remember(user_id, fact):
    fact = (fact or "").strip()
    if not fact:
        return
    norm = _normalize(fact)
    now = time.time()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT id, confidence, evidence_count FROM facts WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        for row in existing:
            # exact match after normalization, or one string fully contains
            # the other (catches "likes Python" said again as "really likes
            # Python for backend work") -- reinforce instead of duplicating.
            stored = conn.execute("SELECT fact FROM facts WHERE id = ?", (row["id"],)).fetchone()["fact"]
            stored_norm = _normalize(stored)
            if norm == stored_norm or norm in stored_norm or stored_norm in norm:
                new_confidence = min(0.97, row["confidence"] + (1 - row["confidence"]) * 0.35)
                conn.execute(
                    "UPDATE facts SET confidence = ?, evidence_count = evidence_count + 1, "
                    "last_confirmed = ? WHERE id = ?",
                    (new_confidence, now, row["id"]),
                )
                return
        conn.execute(
            "INSERT INTO facts (user_id, fact, confidence, evidence_count, last_confirmed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, fact, 0.7, 1, now, now),
        )


def list_facts(user_id, limit=200):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, fact, confidence, evidence_count, last_confirmed, created_at "
            "FROM facts WHERE user_id = ? ORDER BY confidence DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def forget(user_id, fact_id):
    """Returns True if a row was actually deleted -- lets the caller tell
    the user 'already gone' vs 'forgotten' accurately."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM facts WHERE user_id = ? AND id = ?", (user_id, fact_id)
        )
    return cur.rowcount > 0


def clear_facts(user_id):
    with _conn() as conn:
        conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))


def facts_as_context(user_id, limit=25):
    """Rendered as a block to prepend to the system prompt, so the model
    has these facts on every turn without a separate tool round-trip.
    Ordered by confidence (list_facts already sorts that way), capped at
    `limit` so this can't quietly balloon the prompt for a long-lived
    user -- swap for a relevance-ranked subset (embed + top-k) once
    someone has more than ~25 facts worth keeping."""
    facts = list_facts(user_id, limit=limit)
    if not facts:
        return ""
    lines = []
    for f in facts:
        tag = " (confirmed multiple times)" if f["evidence_count"] >= 3 else ""
        lines.append(f"- {f['fact']}{tag}")
    return "\n\nThings you already know about this user, from past conversations:\n" + "\n".join(lines)


# --- permission levels ------------------------------------------------

def get_permission(user_id, tool="send_email"):
    if tool != "send_email":
        return DEFAULT_SEND_EMAIL_LEVEL
    with _conn() as conn:
        row = conn.execute(
            "SELECT send_email_level FROM permissions WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["send_email_level"] if row else DEFAULT_SEND_EMAIL_LEVEL


def set_permission(user_id, level, tool="send_email"):
    if tool != "send_email":
        raise ValueError(f"No permission setting for tool '{tool}' yet.")
    level = int(level)
    if level not in (0, 1, 2, 3, 4):
        raise ValueError("Permission level must be 0-4.")
    with _conn() as conn:
        conn.execute(
            "INSERT INTO permissions (user_id, send_email_level) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET send_email_level = excluded.send_email_level",
            (user_id, level),
        )
    return level


# --- plans (Intent Engine) --------------------------------------------
#
# The Intent Engine's job is to turn a fuzzy goal ("I need internships
# somehow") into a structured, ordered, persisted plan the user and AURA
# share as a living checklist -- intent in, steps out, progress tracked
# across conversations. This is deliberately a DATA layer only: it stores
# and mutates plans/steps and nothing here sends an email, searches, or
# calls a model. The model decomposes the goal (via the create_plan tool)
# and marks steps done (via complete_plan_step) as it works through them
# using the SAME tools and permission gates that already exist -- so a plan
# step that "sends an email" still goes through the level 0-4 permission
# check and the confirm card. No hidden parallel executor, no autonomy the
# user didn't grant. Everything is scoped by user_id, like facts and
# conversations.
#
# A step's `action` is a hint about what kind of work it represents, so the
# frontend can badge it and the model knows which tool to reach for:
#   'search'   -> search_knowledge
#   'email'    -> send_email (still permission-gated + confirm card)
#   'remember' -> remember_fact
#   'manual'   -> something the human does offline (default)
VALID_STEP_ACTIONS = ("search", "email", "remember", "manual")
VALID_PLAN_STATUS = ("active", "done", "abandoned")
VALID_STEP_STATUS = ("pending", "done", "skipped")


def create_plan(user_id, goal, steps):
    """Persist a new plan and its ordered steps in one shot.

    `steps` is a list of either plain strings (description, action defaults
    to 'manual') or dicts {"description": ..., "action": ...}. Returns the
    new plan's id. Steps with a blank description are skipped rather than
    stored empty; an unrecognized action falls back to 'manual' instead of
    raising, so a slightly-off tool call from the model still produces a
    usable plan rather than an error the user sees."""
    goal = (goal or "").strip()
    if not goal:
        raise ValueError("A plan needs a non-empty goal.")
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO plans (user_id, goal, status, created_at, updated_at) "
            "VALUES (?, ?, 'active', ?, ?)",
            (user_id, goal, now, now),
        )
        plan_id = cur.lastrowid
        position = 0
        for step in steps or []:
            if isinstance(step, dict):
                desc = (step.get("description") or "").strip()
                action = (step.get("action") or "manual").strip().lower()
            else:
                desc = str(step or "").strip()
                action = "manual"
            if not desc:
                continue
            if action not in VALID_STEP_ACTIONS:
                action = "manual"
            conn.execute(
                "INSERT INTO plan_steps (plan_id, user_id, position, description, action, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (plan_id, user_id, position, desc, action, now, now),
            )
            position += 1
    return plan_id


def _plan_row(conn, user_id, plan_id):
    return conn.execute(
        "SELECT id, goal, status, created_at, updated_at FROM plans "
        "WHERE user_id = ? AND id = ?",
        (user_id, plan_id),
    ).fetchone()


def get_plan(user_id, plan_id):
    """Return one plan with its steps, or None if it doesn't belong to this
    user / doesn't exist. The user_id check is what keeps plans isolated --
    same rule as facts and knowledge."""
    with _conn() as conn:
        plan = _plan_row(conn, user_id, plan_id)
        if not plan:
            return None
        steps = conn.execute(
            "SELECT id, position, description, action, status, note, created_at, updated_at "
            "FROM plan_steps WHERE plan_id = ? AND user_id = ? ORDER BY position, id",
            (plan_id, user_id),
        ).fetchall()
    out = dict(plan)
    out["steps"] = [dict(s) for s in steps]
    total = len(out["steps"])
    done = sum(1 for s in out["steps"] if s["status"] == "done")
    out["progress"] = {"done": done, "total": total, "pct": round(100 * done / total) if total else 0}
    return out


def list_plans(user_id, include_done=True, limit=100):
    """Plans newest-first, each with a lightweight progress summary but not
    the full step list -- for a plans overview. Use get_plan for detail."""
    query = (
        "SELECT p.id, p.goal, p.status, p.created_at, p.updated_at, "
        "COUNT(s.id) AS total, "
        "SUM(CASE WHEN s.status = 'done' THEN 1 ELSE 0 END) AS done "
        "FROM plans p LEFT JOIN plan_steps s ON s.plan_id = p.id "
        "WHERE p.user_id = ? "
    )
    if not include_done:
        query += "AND p.status = 'active' "
    query += "GROUP BY p.id ORDER BY p.id DESC LIMIT ?"
    with _conn() as conn:
        rows = conn.execute(query, (user_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        total = d.pop("total") or 0
        done = d.pop("done") or 0
        d["progress"] = {"done": done, "total": total, "pct": round(100 * done / total) if total else 0}
        out.append(d)
    return out


def _touch_plan(conn, plan_id, now):
    conn.execute("UPDATE plans SET updated_at = ? WHERE id = ?", (now, plan_id))


def _maybe_autocomplete_plan(conn, user_id, plan_id, now):
    """If every step of a plan is resolved (done or skipped) and at least one
    is actually done, flip the plan itself to 'done'. Keeps plan status
    honest without the model having to remember a separate 'close the plan'
    call. A plan of all-skipped steps is left active -- nothing was actually
    accomplished, so silently marking it done would be misleading."""
    rows = conn.execute(
        "SELECT status FROM plan_steps WHERE plan_id = ? AND user_id = ?",
        (plan_id, user_id),
    ).fetchall()
    if not rows:
        return
    statuses = [r["status"] for r in rows]
    if all(s in ("done", "skipped") for s in statuses) and any(s == "done" for s in statuses):
        conn.execute("UPDATE plans SET status = 'done', updated_at = ? WHERE id = ?", (now, plan_id))


def update_step_status(user_id, step_id, status, note=None):
    """Mark a single step done/skipped/pending. Returns the updated step as a
    dict, or None if the step doesn't exist or isn't this user's. Rolls the
    parent plan to 'done' automatically once all steps are resolved."""
    status = (status or "").strip().lower()
    if status not in VALID_STEP_STATUS:
        raise ValueError(f"Step status must be one of {VALID_STEP_STATUS}.")
    now = time.time()
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, plan_id FROM plan_steps WHERE user_id = ? AND id = ?",
            (user_id, step_id),
        ).fetchone()
        if not row:
            return None
        if note is not None:
            conn.execute(
                "UPDATE plan_steps SET status = ?, note = ?, updated_at = ? WHERE id = ?",
                (status, note, now, step_id),
            )
        else:
            conn.execute(
                "UPDATE plan_steps SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, step_id),
            )
        _touch_plan(conn, row["plan_id"], now)
        _maybe_autocomplete_plan(conn, user_id, row["plan_id"], now)
        updated = conn.execute(
            "SELECT id, plan_id, position, description, action, status, note, updated_at "
            "FROM plan_steps WHERE id = ?",
            (step_id,),
        ).fetchone()
    return dict(updated) if updated else None


def complete_next_step(user_id, plan_id, note=None):
    """Convenience for the model: mark the first still-pending step of a plan
    done, in position order. Returns the step that was completed (dict), or
    None if the plan has no pending steps left / isn't this user's. Lets the
    model say 'done with the next thing' without tracking step ids itself."""
    with _conn() as conn:
        if not _plan_row(conn, user_id, plan_id):
            return None
        nxt = conn.execute(
            "SELECT id FROM plan_steps WHERE plan_id = ? AND user_id = ? AND status = 'pending' "
            "ORDER BY position, id LIMIT 1",
            (plan_id, user_id),
        ).fetchone()
    if not nxt:
        return None
    return update_step_status(user_id, nxt["id"], "done", note=note)


def set_plan_status(user_id, plan_id, status):
    """Explicitly set a plan's status (active/done/abandoned). Returns True if
    a row was updated. 'abandoned' is how a user drops a goal without deleting
    its history."""
    status = (status or "").strip().lower()
    if status not in VALID_PLAN_STATUS:
        raise ValueError(f"Plan status must be one of {VALID_PLAN_STATUS}.")
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE plans SET status = ?, updated_at = ? WHERE user_id = ? AND id = ?",
            (status, now, user_id, plan_id),
        )
    return cur.rowcount > 0


def delete_plan(user_id, plan_id):
    """Delete a plan and its steps. Returns True if the plan existed. The
    step delete is scoped by user_id too (defense in depth -- a plan_id alone
    should never reach across users, but don't rely on the join to enforce
    it)."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM plans WHERE user_id = ? AND id = ?", (user_id, plan_id)
        )
        if cur.rowcount:
            conn.execute(
                "DELETE FROM plan_steps WHERE user_id = ? AND plan_id = ?",
                (user_id, plan_id),
            )
    return cur.rowcount > 0


def active_plans_as_context(user_id, limit=5):
    """A compact summary of the user's active plans to fold into the system
    prompt, so AURA knows what goals are in flight without a tool round-trip
    -- same idea as facts_as_context. Capped tight (goal + next pending step
    only) so it can't balloon the prompt."""
    plans = list_plans(user_id, include_done=False, limit=limit)
    if not plans:
        return ""
    lines = []
    for p in plans:
        prog = p["progress"]
        detail = get_plan(user_id, p["id"])
        nxt = next((s["description"] for s in detail["steps"] if s["status"] == "pending"), None)
        line = f"- Goal: {p['goal']} ({prog['done']}/{prog['total']} steps done)"
        if nxt:
            line += f"; next step: {nxt}"
        lines.append(line)
    return (
        "\n\nActive plans you're helping this user work through "
        "(use complete_plan_step as each gets done):\n" + "\n".join(lines)
    )
