#!/usr/bin/env python3
"""
iit_faculty_scraper.py - build a recipients CSV (email,name,type) of IIT CS/AI
faculty for the JARVIS outreach pipeline.

Run on YOUR machine (needs internet):
    pip install requests beautifulsoup4
    python iit_faculty_scraper.py --exclude recipients_iitm.csv receipts.txt emails.txt -o recipients_iit_all.csv

Only emails that a faculty page actually publishes are collected. Nothing is guessed.
Handles: mailto links, Cloudflare data-cfemail, "name [at] iitb [dot] ac [dot] in",
"(at)", "{at}", " AT " variants.

Options:
    --seeds seeds.csv     override/add start pages: institute,domain,url
    --ai-only             keep only profs whose block text mentions AI/ML/NLP/CV/etc.
    --max-pages N         page cap per institute (default 40)
    --insecure            skip SSL verification (some IIT sites have broken certs)
    --js                  render pages with Playwright (for JS-built faculty pages)
                          pip install playwright && python -m playwright install chromium
    --only NAME           run one institute, e.g. --only Bombay
    --debug               print the faculty/CS links found on each page
Always spot-check the output. Name/type extraction is heuristic.
"""
import argparse, csv, re, sys, time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# institute -> (email domain, start url). Start URLs are institute roots; the crawler
# looks for CS/CSE faculty pages from there. Override with --seeds for exact pages.
INSTITUTES = {
    "IIT Bombay":      ("iitb.ac.in",     "https://www.cse.iitb.ac.in"),
    "IIT Delhi":       ("iitd.ac.in",     "https://cse.iitd.ac.in"),
    "IIT Kanpur":      ("iitk.ac.in",     "https://www.cse.iitk.ac.in"),
    "IIT Madras":      ("iitm.ac.in",     "https://www.cse.iitm.ac.in"),
    "IIT Kharagpur":   ("iitkgp.ac.in",   "https://cse.iitkgp.ac.in"),
    "IIT Roorkee":     ("iitr.ac.in",     "https://iitr.ac.in"),
    "IIT Guwahati":    ("iitg.ac.in",     "https://www.iitg.ac.in/cse"),
    "IIT Hyderabad":   ("iith.ac.in",     "https://cse.iith.ac.in"),
    "IIT BHU":         ("iitbhu.ac.in",   "https://www.iitbhu.ac.in"),
    "IIT Indore":      ("iiti.ac.in",     "https://cse.iiti.ac.in"),
    "IIT Gandhinagar": ("iitgn.ac.in",    "https://cse.iitgn.ac.in"),
    "IIT Jodhpur":     ("iitj.ac.in",     "https://www.iitj.ac.in"),
    "IIT Patna":       ("iitp.ac.in",     "https://www.iitp.ac.in"),
    "IIT Ropar":       ("iitrpr.ac.in",   "https://cse.iitrpr.ac.in"),
    "IIT Mandi":       ("iitmandi.ac.in", "https://www.iitmandi.ac.in"),
    "IIT Tirupati":    ("iittp.ac.in",    "https://www.iittp.ac.in"),
    "IIT Palakkad":    ("iitpkd.ac.in",   "https://iitpkd.ac.in"),
    "IIT Dharwad":     ("iitdh.ac.in",    "https://www.iitdh.ac.in"),
    "IIT Bhilai":      ("iitbhilai.ac.in","https://www.iitbhilai.ac.in"),
    "IIT Goa":         ("iitgoa.ac.in",   "https://www.iitgoa.ac.in"),
    "IIT Jammu":       ("iitjammu.ac.in", "https://www.iitjammu.ac.in"),
    "IIT Bhubaneswar": ("iitbbs.ac.in",   "https://www.iitbbs.ac.in"),
    "IIT (ISM) Dhanbad": ("iitism.ac.in", "https://www.iitism.ac.in"),
}

BAD_LOCALS = re.compile(
    r"^(info|admin|office|registrar|admission[s]?|contact|webmaster|dean|hod|head|"
    r"director|noreply|no-reply|helpdesk|support|academic|acad|ugoffice|pgoffice|"
    r"dugc|dpgc|placement|cse|ccc|ee|me|ce)\b", re.I)

# keyword -> template type (keys in templates.json). First match wins.
TYPE_RULES = [
    ("robotics", r"robot|autonomous (vehicle|system)|control systems|manipulat"),
    ("nlp",      r"natural language|\bnlp\b|computational linguistics|language model|speech"),
    ("cv",       r"computer vision|image processing|image analysis|video|visual|\bvision\b"),
    ("data",     r"data mining|data science|big data|database|information retrieval|bioinformatics"),
    ("systems",  r"operating system|distributed|networks?\b|architecture|compiler|security|"
                 r"cloud|embedded|systems\b|parallel|storage|cryptograph"),
    ("ml",       r"machine learning|deep learning|reinforcement|\bai\b|artificial intelligence|"
                 r"neural|generative|probabilistic|optimization|learning"),
]
AI_RE = re.compile(
    r"machine learning|deep learning|\bai\b|artificial intelligence|nlp|natural language|"
    r"computer vision|reinforcement|neural|language model|data mining|robot|generative", re.I)

INSECURE = False
STRICT = False
# pages that list students/staff/openings, never faculty: skip crawling AND extracting
URL_SKIP_RE = re.compile(
    r"student|alumni|alumnus|scholar|phd|ph-d|ph\.d|mtech|m-tech|btech|b-tech|ms-by|ms-research|"
    r"staff|intern|join|recruit|opening|vacanc|career|postdoc|post-doc|project-?(staff|associate)|"
    r"/ms/|/mtech|/phd", re.I)
CS_ONLY = False
CS_RE = re.compile(r"computer|\bcse?\b|\bcs\b|information technology|\bai\b|artificial intelligence|"
                   r"data science|machine learning|informatics|computing|software", re.I)
DROPS = {}
USE_JS = False
DEBUG = False
_pw = {}
STUDENT_RE = re.compile(
    r"research scholar|ph\.?d\.? scholar|ph\.?d\.? student|research associate|project associate|"
    r"project engineer|\bstaff\b|alumn|post-?doc|\bintern\b|teaching assistant", re.I)
PROF_RE = re.compile(r"\bprof(essor|\.)?\b|\bdr\.?\s", re.I)

HEADERS = {"User-Agent": "Mozilla/5.0 (research-outreach; contact: tiruvenkatesh123@gmail.com)"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
OBF_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)\s*[\[\(\{<]?\s*(?:at|AT)\s*[\]\)\}>]?\s*"
    r"([A-Za-z0-9\-]+(?:\s*[\[\(\{<]?\s*(?:dot|DOT)\s*[\]\)\}>]?\s*[A-Za-z0-9\-]+)+)")


def decode_cf(hexstr):
    try:
        k = int(hexstr[:2], 16)
        return "".join(chr(int(hexstr[i:i + 2], 16) ^ k) for i in range(2, len(hexstr), 2))
    except Exception:
        return ""


def clean(email):
    return email.strip().strip(".,;:()<>[]").lower()


def emails_in_block(block):
    found = set()
    for a in block.find_all("a", href=True):
        h = a["href"]
        if h.lower().startswith("mailto:"):
            found.update(clean(e) for e in EMAIL_RE.findall(h[7:]))
        if "email-protection#" in h:
            found.add(clean(decode_cf(h.split("#")[-1])))
    for el in block.find_all(attrs={"data-cfemail": True}):
        found.add(clean(decode_cf(el["data-cfemail"])))
    text = block.get_text(" ", strip=True)
    found.update(clean(e) for e in EMAIL_RE.findall(text))
    for m in OBF_RE.finditer(text):
        dom = re.sub(r"\s*[\[\(\{<]?\s*(?:dot|DOT)\s*[\]\)\}>]?\s*", ".", m.group(2))
        found.add(clean(f"{m.group(1)}@{dom}"))
    return {e for e in found if EMAIL_RE.fullmatch(e)}


def card_for(el):
    """Climb to the smallest ancestor that looks like one person's card."""
    cur = el
    for _ in range(6):
        par = cur.parent
        if par is None or par.name in ("body", "html"):
            break
        txt = par.get_text(" ", strip=True)
        if len(txt) > 900:
            break
        cur = par
    return cur


def guess_name(card, email):
    for tag in card.find_all(["h1", "h2", "h3", "h4", "h5", "strong", "b", "a"]):
        t = " ".join(tag.get_text(" ", strip=True).split())
        if t and 4 <= len(t) <= 60 and "@" not in t and not re.search(
                r"read more|view|profile|homepage|email|home page|click|www", t, re.I):
            if re.match(r"^(prof\.?|dr\.?|mr\.?|ms\.?|mrs\.?)\s", t, re.I) or \
               re.fullmatch(r"[A-Z][A-Za-z.\- ']+( [A-Z][A-Za-z.\-']+){1,4}", t):
                return t
    m = re.search(r"((?:Prof\.?|Dr\.?)\s+[A-Z][A-Za-z.\- ']{3,40})", card.get_text(" ", strip=True))
    return m.group(1).strip() if m else "Professor"


def guess_type(text):
    for key, pat in TYPE_RULES:
        if re.search(pat, text, re.I):
            return key
    return "ml"


def render_js(url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   ! --js needs: pip install playwright && python -m playwright install chromium",
              file=sys.stderr)
        return None
    if "pw" not in _pw:
        _pw["ctx"] = sync_playwright().start()
        _pw["pw"] = _pw["ctx"].chromium.launch()
    try:
        page = _pw["pw"].new_page(ignore_https_errors=True)
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3000)   # let JS build the list
        html = page.content()
        page.close()
        return html
    except Exception as e:
        print(f"   ! js render {url}: {e}", file=sys.stderr)
        return None


def _host_alive(url):
    """Cheap DNS check so dead hosts (people.iitgn.ac.in/~x etc.) don't burn 20s each."""
    import socket
    try:
        socket.gethostbyname(urlparse(url).hostname)
        return True
    except Exception:
        return False


def get(url, session):
    html = None
    for verify in ([True, False] if INSECURE else [True]):
        try:
            r = session.get(url, headers=HEADERS, timeout=20, verify=verify)
            if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                html = r.text
            break
        except requests.exceptions.SSLError as e:
            if not INSECURE or not verify:
                print(f"   ! {url}: SSL error (retry with --insecure)", file=sys.stderr)
                break
        except requests.RequestException as e:
            print(f"   ! {url}: {type(e).__name__}", file=sys.stderr)
            break
    # JS only when raw HTML shows no emails (listing built by JS, or email injected on profiles)
    has_mail = bool(html and (EMAIL_RE.search(html) or re.search(r"data-cfemail|\bat\b.{0,8}\bdot\b", html, re.I)))
    looks_faculty = bool(re.search(r"faculty|people|profile|prof|staff", url, re.I))
    need_js = USE_JS and (
        html is None
        or len(BeautifulSoup(html, "html.parser").find_all("a")) < 15
        or (looks_faculty and not has_mail))
    if need_js and not (html is None and not _host_alive(url)):
        rendered = render_js(url)
        if rendered:
            html = rendered
    return html


def crawl(start, domain, max_pages, session, delay):
    """Fetch the start page, follow likely CS faculty/people links, return pages."""
    host_root = ".".join(domain.split(".")[-3:])
    queue, seen, pages = [(start, 0)], set(), []
    link_pat = re.compile(r"faculty|people|professor|staff|member", re.I)
    cs_pat = re.compile(r"cse|computer|\bcs\b|\bai\b|artificial|data", re.I)
    dept_pat = re.compile(r"computer science|\bcse\b|\bcs\b|computing|artificial intelligence", re.I)
    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        url = url.split("#")[0]
        if url in seen:
            continue
        seen.add(url)
        html = get(url, session)
        time.sleep(delay)
        if not html:
            continue
        pages.append((url, html))
        soup = BeautifulSoup(html, "html.parser")
        cands = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0]
            p = urlparse(href)
            if not p.netloc.endswith(host_root) or href in seen or href.lower().endswith(".pdf"):
                continue
            if URL_SKIP_RE.search(p.path + "?" + p.query):
                continue
            text = a.get_text(" ", strip=True)
            blob = f"{href} {text}"
            if link_pat.search(blob) and (cs_pat.search(blob) or cs_pat.search(url)):
                queue.append((href, depth + 1)); cands.append(href)
            elif depth == 0 and dept_pat.search(text):
                queue.append((href, depth + 1)); cands.append(href)   # find the CS dept first
            elif re.search(r"[?&]page=\d+|/page/\d+", href) and link_pat.search(url):
                queue.append((href, depth + 1)); cands.append(href)
        if DEBUG:
            print(f"   [debug] {url} -> {len(cands)} candidate links", file=sys.stderr)
            for c in cands[:8]:
                print(f"           {c}", file=sys.stderr)
    return pages


def extract(pages, domain, ai_only):
    out = {}
    for url, html in pages:
        if URL_SKIP_RE.search(urlparse(url).path):
            DROPS["student/staff_page"] = DROPS.get("student/staff_page", 0) + 1
            continue
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            if tag.has_attr("data-cfemail"):
                continue
        seen_cards = set()
        cands = list(soup.select('a[href*="mailto:"], [data-cfemail], a[href*="email-protection"]'))
        for t in soup.find_all(string=re.compile(r"@|\bat\b\W{0,3}[\w\s\[\]\(\)\{\}]{0,30}\bdot\b", re.I)):
            if t.parent and t.parent.name not in ("script", "style"):
                cands.append(t.parent)
        for el in cands:
            card = card_for(el)
            if id(card) in seen_cards:
                continue
            emails = emails_in_block(card)
            if not emails:
                continue
            seen_cards.add(id(card))
            txt = card.get_text(" ", strip=True)
            if CS_ONLY and not (CS_RE.search(txt) or CS_RE.search(url)):
                DROPS["not_cs_dept"] = DROPS.get("not_cs_dept", 0) + 1
                continue
            if STRICT and not PROF_RE.search(txt):
                DROPS["no_prof_title"] = DROPS.get("no_prof_title", 0) + 1
                continue
            if STUDENT_RE.search(txt) and not re.search(r"(assistant|associate)?\s*professor", txt, re.I):
                DROPS["student/staff"] = DROPS.get("student/staff", 0) + 1
                continue
            for e in emails:
                local, _, dom = e.partition("@")
                if not dom.endswith(domain.split(".")[-3] + ".ac.in") and not dom.endswith(domain):
                    DROPS["other_domain"] = DROPS.get("other_domain", 0) + 1
                    continue
                if BAD_LOCALS.match(local):
                    DROPS["office_address"] = DROPS.get("office_address", 0) + 1
                    continue
                # ai-only: judge only when the card actually describes research (>=150 chars)
                if ai_only and len(txt) >= 150 and not AI_RE.search(txt):
                    DROPS["not_ai"] = DROPS.get("not_ai", 0) + 1
                    continue
                if e not in out:
                    out[e] = {"email": e, "name": guess_name(card, e), "type": guess_type(txt),
                              "source": url, "snippet": txt[:140]}
    return out


def load_excluded(paths):
    ex = set()
    for p in paths or []:
        try:
            with open(p, encoding="utf-8") as f:
                ex.update(clean(e) for e in EMAIL_RE.findall(f.read()))
        except OSError as e:
            print(f"! cannot read exclude file {p}: {e}", file=sys.stderr)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="recipients_iit_all.csv")
    ap.add_argument("--seeds", help="CSV: institute,domain,url (adds/overrides)")
    ap.add_argument("--exclude", nargs="*", help="files with already-contacted emails")
    ap.add_argument("--ai-only", action="store_true")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--only", help="substring filter, e.g. 'Bombay'")
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--cs-only", action="store_true",
                    help="keep only cards/pages that mention CS/CSE/AI (drops other departments)")
    ap.add_argument("--strict", action="store_true",
                    help="require Prof/Dr in the card (cleaner, but can drop real faculty)")
    ap.add_argument("--js", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    global INSECURE, USE_JS, DEBUG, STRICT, CS_ONLY
    INSECURE, USE_JS, DEBUG, STRICT, CS_ONLY = args.insecure, args.js, args.debug, args.strict, args.cs_only
    if INSECURE:
        import urllib3
        urllib3.disable_warnings()

    insts = dict(INSTITUTES)
    if args.seeds:
        with open(args.seeds, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                insts[r["institute"].strip()] = (r["domain"].strip(), r["url"].strip())

    excluded = load_excluded(args.exclude)
    session = requests.Session()
    results, report = {}, []
    def save():
        rows = sorted(results.values(), key=lambda r: r["email"])
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["email", "name", "type"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        review = args.out.rsplit(".", 1)[0] + "_review.csv"   # for skimming before you send
        with open(review, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["email", "name", "type", "source", "snippet"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    try:
        for name, (domain, start) in insts.items():
            if args.only and args.only.lower() not in name.lower():
                continue
            print(f"-> {name} ({start})", flush=True)
            pages = crawl(start, domain, args.max_pages, session, args.delay)
            DROPS.clear()
            found = extract(pages, domain, args.ai_only)
            new = {e: v for e, v in found.items() if e not in excluded and e not in results}
            results.update(new)
            report.append((name, len(pages), len(found), len(new)))
            print(f"   pages={len(pages)} emails={len(found)} new={len(new)}"
                  + (f"  dropped={dict(DROPS)}" if DROPS else ""), flush=True)
            save()   # progress is kept even if you stop the run
    except KeyboardInterrupt:
        print("\nStopped by you. Saving what we have...")
    finally:
        try:
            if "ctx" in _pw:
                _pw["pw"].close(); _pw["ctx"].stop()
        except Exception:
            pass

    save()

    print(f"\nWrote {len(results)} recipients -> {args.out}")
    zero = [n for n, p, f_, _ in report if f_ == 0]
    if zero:
        print("0 emails (JS-rendered or wrong start URL; pass exact faculty URLs via --seeds):")
        for n in zero:
            print("  -", n)


if __name__ == "__main__":
    main()
