#!/usr/bin/env python3
"""
clean_recipients.py - turn the scraper's CSV into a send-ready recipients CSV.

    python clean_recipients.py recipients_iit_all_clean.csv

Writes:
    recipients_final.csv     email,name,type   (name = "Prof. First Last" or "Professor")
    recipients_rejected.csv  email,name,reason (what was dropped / downgraded)
"""
import csv
import re
import sys

ROLE_LOCAL = re.compile(
    r"(^|[._-])(ahead|a?dean|hod|head|fic|ccs\w*|(cse|ee|me|ce)?off(ice)?|admin|registrar|dugc|dpgc|"
    r"convener|chair|info|contact|webmaster|helpdesk|support|dept|placement|training|intern\w*|"
    r"career\w*|alumni|office|faculty|staff|lab|sysadmin|sysad|noc|pro|vice\w*|help\w*|urbic|firstname|"
    r"cairoffice|gate\w*|jam|cdc|hostel|mess|library|nss|sports)([._-]|\d|$)", re.I)
ROLE_EXACT = {"cse", "csea", "csed", "ee", "me", "ce", "cs", "it", "ai", "dept", "hod"}

JUNK_NAME = re.compile(
    r"personal website|find a location|home ?page|website|click|read more|view|profile|email|"
    r"contact|location|department|faculty$|people|more", re.I)

TITLE_PREFIX = re.compile(r"^(prof(essor)?\.?|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", re.I)
DESIG = re.compile(
    r"\b((associate|assistant|adjunct|visiting|emeritus|institute)\s+)?"
    r"(head|dean|professor|chair|convener|coordinator|registrar)"
    r"(\s+of\s+(the\s+)?(department|dept)(\s+of\s+[\w &]+)?)?\b"
    r"|\bfaculty\s+in[- ]?charge\b.*|\bprofessor\s+of\s+practice\b.*",
    re.I)


def clean_name(raw):
    n = " ".join((raw or "").split())
    if not n or JUNK_NAME.search(n):
        return ""
    for _ in range(3):
        n = TITLE_PREFIX.sub("", n).strip()
    for _ in range(3):
        m = DESIG.search(n)
        if not m:
            break
        n = n[m.end():] if m.start() == 0 else n[:m.start()]
        n = TITLE_PREFIX.sub("", " ".join(n.split())).strip(" ,.-")
    n = TITLE_PREFIX.sub("", " ".join(n.split())).strip(" ,.-")
    cut = re.search(r"\b(assistant|associate|visiting|adjunct|vice|ramanujan|professo|faculty|fellow|chair\w*|"
                    r"in-charge|incharge|head|dean|emeritus|institute)\b", n, re.I)
    if cut and cut.start() > 0:
        n = n[:cut.start()].strip(" ,.-")
    parts = [p for p in n.split() if re.fullmatch(r"[A-Za-z][A-Za-z.\-']*", p)]
    if not (2 <= len(parts) <= 5):
        return ""
    return " ".join(parts)


def name_fits_email(name, email):
    local = re.sub(r"[^a-z]", "", email.split("@")[0].lower())
    toks = [re.sub(r"[^a-z]", "", t.lower()) for t in name.split()]
    toks = [t for t in toks if t]
    if not local or not toks:
        return False
    if any(len(t) >= 3 and (t in local or local in t) for t in toks):
        return True
    initials = "".join(t[0] for t in toks)
    if local == initials or (len(local) >= 2 and initials.startswith(local)):
        return True
    joined = " ".join(toks)
    return any(local.startswith(t[0]) and local[1:] in joined for t in toks) and len(local) > 3


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python clean_recipients.py recipients_iit_all_clean.csv")
    final, rejected, seen = [], [], set()
    with open(sys.argv[1], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            email = (r.get("email") or "").strip().lower()
            raw = (r.get("name") or "").strip()
            typ = (r.get("type") or "ml").strip().lower()
            if not email or email in seen:
                continue
            seen.add(email)
            local = email.split("@")[0]
            if local in ROLE_EXACT or ROLE_LOCAL.search(local):
                rejected.append((email, raw, "role/office alias"))
                continue
            name = clean_name(raw)
            if name and name_fits_email(name, email):
                greeting = f"Prof. {name}"
            else:
                greeting = "Professor"
                if name:
                    rejected.append((email, raw, f"kept, greeting 'Professor' (name '{name}' doesn't fit email)"))
            final.append({"email": email, "name": greeting, "type": typ})

    # same person name under different IIT domains => one of the emails is likely a mis-pair
    def inst(e):
        d = e.split("@")[1]
        return re.sub(r"^(cse|www|cs)\.", "", d)
    by_name = {}
    for r in final:
        if r["name"] != "Professor":
            by_name.setdefault(r["name"].lower(), []).append(r)
    review = []
    for nm, rs in by_name.items():
        if len({inst(x["email"]) for x in rs}) > 1:
            for x in rs:
                review.append(x)
    review_emails = {x["email"] for x in review}
    final = [r for r in final if r["email"] not in review_emails]
    with open("recipients_review.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["email", "name", "type"])
        w.writeheader()
        w.writerows(review)

    with open("recipients_final.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["email", "name", "type"])
        w.writeheader()
        w.writerows(final)
    with open("recipients_rejected.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["email", "name", "reason"])
        w.writerows(rejected)

    named = sum(1 for r in final if r["name"] != "Professor")
    print(f"final: {len(final)} recipients ({named} with names, {len(final) - named} plain 'Professor')")
    print(f"rejected/flagged: {len(rejected)} -> recipients_rejected.csv")
    print(f"needs manual check (same name at 2+ IITs): {len(review)} -> recipients_review.csv")


if __name__ == "__main__":
    main()
