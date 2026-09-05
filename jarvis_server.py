#!/usr/bin/env python3
"""
jarvis_server.py - local HTTP bridge between the Jarvis HUD (jarvis.html)
and the existing jarvis.py agent.

jarvis.py's tool loop was built for a terminal: it calls input() to get
"Send this email? [y/N]" confirmation, which just isn't possible from a
browser. This server keeps everything else about jarvis.py identical --
same Groq model/system prompt, same tools, same Gmail send, same RAG
search -- but splits the send_email flow into two HTTP calls so the
frontend can show a preview with Send/Cancel buttons instead:

    1. POST /api/chat      -> if the model wants to send an email, the
                               server PAUSES and returns a preview instead
                               of sending anything.
    2. POST /api/confirm   -> the browser calls this once the user clicks
                               Send or Cancel; only then does send_gmail()
                               actually run.

Setup (in the same folder as jarvis.py, rag.py, credentials.json, etc.):
    pip install -r requirements.txt        # now includes flask, flask-cors
    export GROQ_API_KEY="gsk_..."           # (or setx / $env: on Windows)
    python jarvis_server.py

Then open jarvis.html in a browser. It talks to http://localhost:8787 by
default -- change SERVER_URL near the top of jarvis.html's <script> if you
run this on a different host/port.

Note: this is a small local dev server for a single user (you) -- it holds
one conversation in memory and isn't meant to be exposed to the internet.
"""

import json
import os
import uuid
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests

import jarvis as core  # reuses GROQ config, SYSTEM_PROMPT, ALL_TOOLS, send_gmail, etc.
import memory_store

memory_store.init_db()

app = Flask(__name__)
CORS(app)  # allow the browser (file:// or a local dev server) to call this API

# --- Single-user mode (no auth) -------------------------------------------
# AURA is a personal, single-user tool: one person, one machine, no login
# system. Every request runs as the same fixed local user. AURA_USER lets
# you rename that account if you want (e.g. to keep memory_store rows under
# a specific name); it defaults to "local" and matches jarvis.py's
# CLI_USER_ID so the CLI and the server share the same memory/knowledge.
LOCAL_USER = os.environ.get("AURA_USER", "local").strip() or "local"


def require_auth(f):
    """Attaches a fixed request.user_id / request.user_email to every
    request. Kept as a decorator (rather than inlining LOCAL_USER in each
    route) so reintroducing real multi-user auth later only means editing
    this one function."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        request.user_id = LOCAL_USER
        request.user_email = f"{LOCAL_USER}@local.dev"
        return f(*args, **kwargs)
    return wrapper


# --- Per-user session state ----------------------------------------------
# `messages` now hydrates from memory_store (SQLite) on first touch per
# process instead of always starting empty, so a redeploy or restart no
# longer wipes everyone's conversation -- that was the biggest gap between
# "chatbot" and "personal system that remembers." Every message appended
# to a session also gets written through to memory_store immediately (see
# _add_message below), so the in-memory dict here is a cache, not the
# source of truth.
#
# pending_sends stays in-memory only -- an unconfirmed email/bulk preview
# is meant to be short-lived and re-askable, not something worth
# persisting across a restart.
#
# Note: this cache is still per-process, so keep WEB_CONCURRENCY=1 on
# Render until _user_sessions itself is replaced with a Redis-backed (or
# purely memory_store-backed) lookup -- otherwise two workers would each
# hold a different in-memory copy of the same user's pending_sends.
_user_sessions = {}


def _build_system_content(user_id):
    """System prompt + everything AURA should know up front for this user
    without a tool round-trip: long-term facts and any active plans in
    flight. Kept in one place so /api/chat hydration and /api/reset can't
    drift apart."""
    return (
        core.SYSTEM_PROMPT
        + memory_store.facts_as_context(user_id)
        + memory_store.active_plans_as_context(user_id)
    )


def _get_session(user_id):
    if user_id not in _user_sessions:
        system_content = _build_system_content(user_id)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(memory_store.load_recent_messages(user_id))
        _user_sessions[user_id] = {"messages": messages, "pending_sends": {}}
    return _user_sessions[user_id]


def _add_message(user_id, sess, role, content, tool_calls=None, tool_call_id=None):
    """Append to both the live in-memory list (what the next Groq call
    sees) and memory_store (what survives a restart)."""
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    sess["messages"].append(msg)
    memory_store.append_message(user_id, role, content, tool_calls=tool_calls, tool_call_id=tool_call_id)


def groq_chat(msgs):
    """Same call as jarvis.stream_chat(), minus the terminal print()s."""
    if not core.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server.")

    headers = {
        "Authorization": f"Bearer {core.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": core.MODEL, "messages": core.trim_for_budget(msgs), "tools": core.ALL_TOOLS}
    resp = requests.post(core.GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls", []) or []

    if not tool_calls:
        fallback = core._extract_fallback_tool_call(content)
        if fallback is not None:
            tool_calls, content = fallback, ""

    return content, tool_calls


def run_followup(user_id, sess):
    """Once tool result(s) are in this user's `messages`, get Jarvis's
    natural-language follow-up (e.g. "Email sent.") and store it."""
    followup, _ = groq_chat(sess["messages"])
    if followup:
        _add_message(user_id, sess, "assistant", followup)
    return followup


def _parse_args(call):
    fn = call.get("function", {})
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return fn.get("name"), args


# --- /bulk and /bulk-role slash commands ------------------------------
# main()'s CLI loop intercepts these before they ever reach the model (see
# jarvis.py). The server has to do the same thing itself, or a command like
# "/bulk-role recipients.csv" just gets forwarded to Groq as plain chat text
# and the model (reasonably) treats it like a one-off email request instead
# of a templated batch send.

def _build_bulk_role_preview(path):
    """Mirrors jarvis.bulk_role_send(), minus the print()s/input()."""
    templates = core.load_templates()
    recipients = core.load_recipients_csv(path)
    if not recipients:
        return None, f"No recipients found in {path}."

    unknown_types = sorted({r["type"] for r in recipients if r["type"] not in templates})
    if unknown_types:
        return None, (
            f"Unknown type(s) in CSV: {', '.join(unknown_types)}. "
            f"Valid types: {', '.join(templates.keys())}"
        )

    counters = {t: core.itertools.cycle(range(len(templates[t]["variants"]))) for t in templates}
    attach_path = core.RESUME_PATH if core.resume_available() else None

    items = []
    for r in recipients:
        variant_idx = next(counters[r["type"]])
        subject, body = core.compose_role_email(templates, r["name"], r["type"], variant_idx)
        items.append({"to": r["email"], "subject": subject, "body": body, "type": r["type"]})

    return {"items": items, "attach_path": attach_path}, None


def _build_bulk_preview(path, subject, body):
    """Mirrors jarvis.bulk_send(), minus the print()s/input()."""
    if body.startswith("@"):
        body_file = body[1:].strip()
        try:
            with open(body_file, "r", encoding="utf-8") as bf:
                body = bf.read()
        except OSError as e:
            return None, f"Couldn't read body file '{body_file}': {e}"

    recipients = core.load_recipients(path)
    if not recipients:
        return None, f"No addresses found in {path}."

    attach_path = core.RESUME_PATH if core.resume_available() else None
    items = [{"to": addr, "subject": subject, "body": body} for addr in recipients]
    return {"items": items, "attach_path": attach_path}, None


def _make_bulk_confirm_response(sess, data, kind):
    pending_id = uuid.uuid4().hex
    sess["pending_sends"][pending_id] = {
        "type": "bulk",
        "items": data["items"],
        "attach_path": data["attach_path"],
    }
    will_attach = bool(data["attach_path"])
    role_note = "role-matched " if kind == "bulk-role" else ""
    attach_note = (
        " with your resume attached" if will_attach
        else f" (no resume attached -- {core.RESUME_PATH} not found)"
    )
    summary = f"Ready to send {len(data['items'])} {role_note}emails{attach_note}."

    return jsonify({
        "type": "confirm_bulk",
        "pending_id": pending_id,
        "text": summary,
        "preview": {
            "count": len(data["items"]),
            "recipients": [it["to"] for it in data["items"]],
            "sample_subject": data["items"][0]["subject"] if data["items"] else "",
            "sample_body": data["items"][0]["body"] if data["items"] else "",
            "attachment": os.path.basename(data["attach_path"]) if will_attach else None,
        },
    })


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    sess = _get_session(request.user_id)
    messages = sess["messages"]
    pending_sends = sess["pending_sends"]

    user_text = (request.json or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"type": "error", "text": "Empty message."}), 400

    lower = user_text.lower()

    # /bulk-role <recipients.csv> -- must be checked before /bulk, since
    # "/bulk-role ...".startswith("/bulk") is also true.
    if lower.startswith("/bulk-role"):
        rest = user_text[len("/bulk-role"):].strip()
        if not rest:
            return jsonify({"type": "text", "text": "Usage: /bulk-role recipients.csv"})
        try:
            data, err = _build_bulk_role_preview(rest)
        except RuntimeError as e:
            data, err = None, str(e)
        if err:
            return jsonify({"type": "text", "text": f"[bulk-role] {err}"})
        return _make_bulk_confirm_response(sess, data, kind="bulk-role")

    # /bulk <file.txt> [| subject | body]
    if lower.startswith("/bulk"):
        rest = user_text[len("/bulk"):].strip()
        parts = [p.strip() for p in rest.split("|")]
        file_path = parts[0] if parts and parts[0] else None
        subject = parts[1] if len(parts) > 1 and parts[1] else "Hello from Tiru Jarvis"
        body = parts[2] if len(parts) > 2 and parts[2] else "Hello from Tiru Jarvis!"
        if not file_path:
            return jsonify({
                "type": "text",
                "text": "Usage: /bulk emails.txt  OR  /bulk emails.txt | Subject | Body text",
            })
        try:
            data, err = _build_bulk_preview(file_path, subject, body)
        except RuntimeError as e:
            data, err = None, str(e)
        if err:
            return jsonify({"type": "text", "text": f"[bulk] {err}"})
        return _make_bulk_confirm_response(sess, data, kind="bulk")

    _add_message(request.user_id, sess, "user", user_text)

    try:
        content, tool_calls = groq_chat(messages)
    except Exception as e:
        # don't leave a dangling user turn on failure -- pop from both the
        # live list and memory_store so a retry doesn't duplicate it.
        messages.pop()
        memory_store.pop_last_message(request.user_id)
        return jsonify({"type": "error", "text": str(e)}), 500

    _add_message(request.user_id, sess, "assistant", content, tool_calls=tool_calls or None)

    if not tool_calls:
        return jsonify({"type": "text", "text": content})

    for call in tool_calls:
        name, args = _parse_args(call)
        tool_call_id = call.get("id", "")

        if name == "search_knowledge":
            import rag
            try:
                result = rag.query_as_context(request.user_id, args.get("query", ""))
            except RuntimeError as e:
                result = f"Knowledge search unavailable: {e}"
            _add_message(request.user_id, sess, "tool", result, tool_call_id=tool_call_id)

        elif name == "remember_fact":
            fact = args.get("fact", "")
            memory_store.remember(request.user_id, fact)
            _add_message(request.user_id, sess, "tool", f"Remembered: {fact}", tool_call_id=tool_call_id)

        elif name == "create_plan":
            try:
                plan_id = memory_store.create_plan(
                    request.user_id, args.get("goal", ""), args.get("steps", [])
                )
                plan = memory_store.get_plan(request.user_id, plan_id)
                lines = [f"Created plan #{plan_id}: {plan['goal']}"]
                for s in plan["steps"]:
                    lines.append(f"  {s['position'] + 1}. [{s['action']}] {s['description']}")
                tool_result = "\n".join(lines)
            except ValueError as e:
                tool_result = f"Couldn't create plan: {e}"
            _add_message(request.user_id, sess, "tool", tool_result, tool_call_id=tool_call_id)

        elif name == "complete_plan_step":
            step = memory_store.complete_next_step(
                request.user_id, args.get("plan_id"), note=args.get("note")
            )
            if step is None:
                tool_result = "No pending step to complete on that plan (it may be finished or not exist)."
            else:
                plan = memory_store.get_plan(request.user_id, step["plan_id"])
                prog = plan["progress"]
                tool_result = (
                    f"Marked done: {step['description']} "
                    f"({prog['done']}/{prog['total']} steps complete)."
                )
                if plan["status"] == "done":
                    tool_result += " That was the last step -- plan complete."
            _add_message(request.user_id, sess, "tool", tool_result, tool_call_id=tool_call_id)

        elif name == "send_email":
            level = memory_store.get_permission(request.user_id, "send_email")

            if level == 0:
                msg = "Email sending is turned off at your current permission level. Change it in settings to draft or send."
                _add_message(request.user_id, sess, "tool", msg, tool_call_id=tool_call_id)
                continue

            if level == 1:
                draft = f"Draft (not sent -- your permission level is draft-only):\n\nTo: {args.get('to','')}\nSubject: {args.get('subject','')}\n\n{args.get('body','')}"
                _add_message(request.user_id, sess, "tool", "Drafted, not sent (draft-only permission level).", tool_call_id=tool_call_id)
                return jsonify({"type": "text", "text": draft})

            want_resume = args.get("attach_resume", True)
            attach_path = core.RESUME_PATH if want_resume else None
            will_attach = bool(attach_path and os.path.exists(attach_path))

            if level >= 3:
                # Auto-send: this is the user's own configured permission
                # level acting on a *single* recipient they're already
                # mid-conversation about -- not the same risk shape as a
                # bulk send, which always confirms regardless of this
                # setting (see the note on DEFAULT_SEND_EMAIL_LEVEL above).
                try:
                    core.send_gmail(
                        to=args.get("to", ""),
                        subject=args.get("subject", ""),
                        body=args.get("body", ""),
                        attach_path=attach_path if will_attach else None,
                    )
                    result = f"Sent to {args.get('to','')} automatically (permission level {level})."
                except Exception as e:
                    result = f"Auto-send failed: {e}"
                _add_message(request.user_id, sess, "tool", result, tool_call_id=tool_call_id)
                continue

            # level == 2, the default: existing confirm-card flow
            pending_id = uuid.uuid4().hex
            pending_sends[pending_id] = {
                "call_id": tool_call_id,
                "args": args,
                "will_attach": will_attach,
                "attach_path": attach_path,
            }
            return jsonify({
                "type": "confirm_email",
                "pending_id": pending_id,
                "text": content or "I've drafted an email -- take a look before I send it.",
                "preview": {
                    "to": args.get("to", ""),
                    "subject": args.get("subject", ""),
                    "body": args.get("body", ""),
                    "attachment": os.path.basename(attach_path) if will_attach else None,
                },
            })

        else:
            _add_message(request.user_id, sess, "tool", f"Unknown tool '{name}'.", tool_call_id=tool_call_id)

    try:
        followup = run_followup(request.user_id, sess)
    except Exception as e:
        return jsonify({"type": "error", "text": str(e)}), 500
    return jsonify({"type": "text", "text": followup})


@app.route("/api/confirm", methods=["POST"])
@require_auth
def confirm():
    sess = _get_session(request.user_id)
    messages = sess["messages"]
    pending_sends = sess["pending_sends"]

    body = request.json or {}
    pending_id = body.get("pending_id")
    approved = bool(body.get("approved"))
    pending = pending_sends.pop(pending_id, None)
    if not pending:
        return jsonify({"type": "error", "text": "That confirmation has expired -- ask me again."}), 404

    if pending.get("type") == "bulk":
        if not approved:
            return jsonify({"type": "text", "text": "Bulk send cancelled."})

        sent, failed = 0, []
        for item in pending["items"]:
            try:
                core.send_gmail(
                    item["to"], item["subject"], item["body"],
                    attach_path=pending["attach_path"],
                )
                sent += 1
            except Exception as e:
                failed.append(f"{item['to']}: {e}")

        summary = f"Bulk send done. Sent {sent}/{len(pending['items'])}."
        if failed:
            summary += " Failed -- " + "; ".join(failed)
        # Persisted as an assistant turn purely for the audit trail -- a
        # bulk send used to leave zero trace in the conversation, which
        # made "what did I already send, and to whom" unanswerable later.
        _add_message(request.user_id, sess, "assistant", summary)
        return jsonify({"type": "text", "text": summary})

    args = pending["args"]
    if approved:
        try:
            msg_id = core.send_gmail(
                args["to"], args["subject"], args["body"],
                attach_path=pending["attach_path"] if pending["will_attach"] else None,
            )
            note = " with the resume attached" if pending["will_attach"] else ""
            result = f"Email sent successfully to {args.get('to')}{note} (id: {msg_id})."
        except Exception as e:
            result = f"Failed to send email: {e}"
    else:
        result = "The user declined to send this email."

    _add_message(request.user_id, sess, "tool", result, tool_call_id=pending["call_id"])
    try:
        followup = run_followup(request.user_id, sess)
    except Exception as e:
        followup = None
    return jsonify({"type": "text", "text": followup or result})


@app.route("/api/reset", methods=["POST"])
@require_auth
def reset():
    """Clears the live conversation (both the in-memory cache and
    memory_store) but leaves long-term facts alone -- "start a fresh
    conversation" and "forget what you know about me" are different asks,
    handled by /api/reset and /api/memories respectively."""
    memory_store.clear_conversation(request.user_id)
    _user_sessions[request.user_id] = {
        "messages": [{"role": "system", "content": _build_system_content(request.user_id)}],
        "pending_sends": {},
    }
    return jsonify({"type": "ok"})


@app.route("/api/memories", methods=["GET"])
@require_auth
def list_memories():
    """Powers a 'what does AURA remember about me' view in the frontend --
    the user-control side of item 18 in the AURA spec (remember/forget/
    delete all), not just something the model writes to silently."""
    return jsonify({"type": "ok", "memories": memory_store.list_facts(request.user_id)})


@app.route("/api/memories/<int:fact_id>", methods=["DELETE"])
@require_auth
def delete_memory(fact_id):
    ok = memory_store.forget(request.user_id, fact_id)
    if not ok:
        return jsonify({"type": "error", "text": "No memory with that id."}), 404
    return jsonify({"type": "ok"})


@app.route("/api/memories", methods=["DELETE"])
@require_auth
def clear_memories():
    memory_store.clear_facts(request.user_id)
    return jsonify({"type": "ok"})


@app.route("/api/plans", methods=["GET"])
@require_auth
def list_plans():
    """Plans overview for the frontend's command-center view -- each with a
    progress summary but not full steps. ?include_done=0 to hide finished
    ones."""
    include_done = request.args.get("include_done", "1") != "0"
    return jsonify({
        "type": "ok",
        "plans": memory_store.list_plans(request.user_id, include_done=include_done),
    })


@app.route("/api/plans/<int:plan_id>", methods=["GET"])
@require_auth
def get_plan(plan_id):
    plan = memory_store.get_plan(request.user_id, plan_id)
    if not plan:
        return jsonify({"type": "error", "text": "No plan with that id."}), 404
    return jsonify({"type": "ok", "plan": plan})


@app.route("/api/plans/<int:plan_id>", methods=["DELETE"])
@require_auth
def delete_plan(plan_id):
    ok = memory_store.delete_plan(request.user_id, plan_id)
    if not ok:
        return jsonify({"type": "error", "text": "No plan with that id."}), 404
    return jsonify({"type": "ok"})


@app.route("/api/permissions", methods=["GET"])
@require_auth
def get_permissions():
    """Levels 0-4 -- see the comment above DEFAULT_SEND_EMAIL_LEVEL in
    memory_store.py for what each one does. Frontend should render this
    as a slider/segmented control, not a raw number."""
    return jsonify({"type": "ok", "send_email_level": memory_store.get_permission(request.user_id, "send_email")})


@app.route("/api/permissions", methods=["PUT"])
@require_auth
def set_permissions():
    body = request.json or {}
    if "send_email_level" not in body:
        return jsonify({"type": "error", "text": "Missing send_email_level."}), 400
    try:
        level = memory_store.set_permission(request.user_id, body["send_email_level"], "send_email")
    except ValueError as e:
        return jsonify({"type": "error", "text": str(e)}), 400
    return jsonify({"type": "ok", "send_email_level": level})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"type": "ok", "model": core.MODEL, "groq_key_set": bool(core.GROQ_API_KEY), "user": LOCAL_USER})


# Serve the command-center dashboard from the same origin as the API, so the
# browser has no CORS boundary to cross and (in dev mode) needs no token.
@app.route("/", methods=["GET"])
def dashboard():
    here = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(here, "dashboard.html")


if __name__ == "__main__":
    core.check_groq_setup()
    print(f"Jarvis server -- model: {core.MODEL}")
    print(f"Single-user mode: all requests run as '{LOCAL_USER}'. No login required.")
    print("\nOpen the command center:  http://localhost:8787/")
    print("API base:                 http://localhost:8787/api\n")
    app.run(host="127.0.0.1", port=8787, debug=False)
