"""D8 `sec_suspensions`: SEC trading suspensions -> weakly labelled daily BarFrames (one per suspended ticker).

Source   https://www.sec.gov/enforcement-litigation/trading-suspensions  (HTML list, paginated) and .../trading-suspensions/rss
         (public domain, US government work). The list gives release number (34-xxxxx), date and respondent names; the
         TICKERS ARE ONLY INSIDE THE PDF ORDERS, so each order is read (pypdf if importable, else a pre-extracted .txt with
         the same stem) and the ticker is pulled out with the regexes documented at `extract_ticker`.
Prices   daily OHLCV from Yahoo Finance through yfinance (Yahoo terms of use: personal / research use, do not redistribute).
         Most suspended OTC issuers are no longer served by Yahoo: every ticker without prices is recorded in
         meta["survivorship_dropped"] (never synthesised), so the surviving sample is biased towards issuers that kept trading.
Label    WEAK: a suspension is not proof of a pump (many orders cite delinquent filings or doubts about public information).
         Episode(family "pump", difficulty "real", t0 = suspension - label_d (30) days, t1 = suspension day); the frame covers
         calendar days [suspension - pre_d (120), suspension + post_d (20)], one bar per trading day present in the price file.
Features Only ret (close-to-close log return), rv (log(high / low), a range proxy for intraday volatility) and vol_trade
         (volume / median volume of the bars BEFORE the label window, i.e. a causal normaliser) are available; the other
         LOB-lite columns are 0 with feat_mask False.

Local layout under root (data_raw/sec_suspensions/):
    list*.html, rss*.xml            saved list pages / feed (any number of files)
    orders/<name>.pdf | .txt        orders named as in the list URL (or containing the release number)
    prices/<TICKER>.csv             Date,Open,High,Low,Close,Volume (yfinance history CSV)
    prices/_unavailable.txt         tickers for which Yahoo returned nothing (one per line), written by the price fetch
"""
from __future__ import annotations
import glob
import html as _html
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..schema import BarFrame, Episode
from .common import register_loader, missing, NS

KEY = "sec_suspensions"
LICENCE = "public domain (SEC list and orders); prices: Yahoo Finance terms, not redistributable"
LIST_URL = "https://www.sec.gov/enforcement-litigation/trading-suspensions"
RSS_URL = LIST_URL + "/rss"
DEFAULT_ROOT = os.path.join("data_raw", KEY)
DAY_NS = 86400 * NS
DROPPED: List[Dict[str, Any]] = []   # survivorship / parsing drops of the last loader call

RECIPE = """SEC trading suspensions (public domain). sec.gov rejects clients without a declared User-Agent ("Name contact@email"):
  mkdir -p data_raw/sec_suspensions/orders data_raw/sec_suspensions/prices
  curl -fL -A "Your Name you@example.org" -o data_raw/sec_suspensions/list_p0.html "https://www.sec.gov/enforcement-litigation/trading-suspensions?page=0"   # repeat for further pages
  curl -fL -A "Your Name you@example.org" -o data_raw/sec_suspensions/rss.xml "https://www.sec.gov/enforcement-litigation/trading-suspensions/rss"
  # every order PDF linked from the list goes to data_raw/sec_suspensions/orders/<same file name>.pdf
  # without pypdf: pdftotext orders/<name>.pdf orders/<name>.txt
or:  from maniparena.data.loaders.sec_suspensions import prepare
     prepare("data_raw/sec_suspensions", download=True, user_agent="Your Name you@example.org", pages=5)
Prices (needs `pip install yfinance`; Yahoo drops most delisted OTC tickers):
  python3 -c "import yfinance as yf; yf.Ticker('TICK').history(start='2019-01-01', end='2019-08-01', auto_adjust=False).to_csv('data_raw/sec_suspensions/prices/TICK.csv')"
  tickers Yahoo does not know: add one per line to data_raw/sec_suspensions/prices/_unavailable.txt"""

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_RELEASE = re.compile(r"\b(3[34]-\d{4,6})\b")
_PDF_HREF = re.compile(r"""href\s*=\s*["']([^"']+?\.pdf)(?:\?[^"']*)?["']""", re.I)
_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),                                            # 2019-06-28 (also <time datetime=...>)
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),\s*(\d{4})\b"),                           # June 28, 2019 / Jun. 28, 2019
    re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"),                            # 28 Jun 2019 (RSS pubDate)
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),                                         # 06/28/2019
]


def _parse_date(text: str) -> Optional[str]:
    """First recognisable date in text -> 'YYYY-MM-DD' (None if absent)."""
    best = None
    for k, pat in enumerate(_DATE_PATTERNS):
        m = pat.search(text)
        if not m:
            continue
        try:
            if k == 0:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif k == 1:
                mo = _MONTHS.get(m.group(1)[:3].lower()); d, y = int(m.group(2)), int(m.group(3))
            elif k == 2:
                mo = _MONTHS.get(m.group(2)[:3].lower()); d, y = int(m.group(1)), int(m.group(3))
            else:
                mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo is None or not (1 <= mo <= 12 and 1 <= d <= 31 and 1990 <= y <= 2100):
                continue
        except ValueError:
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), f"{y:04d}-{mo:02d}-{d:02d}")
    return best[1] if best else None


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_suspension_list(text: str) -> List[Dict[str, Any]]:
    """Parse one saved list page (HTML table rows) or the RSS feed into
        [{"release": "34-86252", "date": "2019-06-28", "title": respondents, "urls": [order pdf urls], "link": item page}]
    HTML: every <tr> that contains a release number 3x-nnnnn; date = first date in the row; pdf links = href ending in .pdf.
    RSS:  every <item>; release number / date looked up in title + description, falling back to pubDate.
    Rows without a release number are ignored; rows without a date are kept with date None (the loader drops them)."""
    out: List[Dict[str, Any]] = []
    head = text.lstrip()[:400].lower()
    if "<rss" in head or "<feed" in head or (head.startswith("<?xml") and "<html" not in head):
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
        for it in root.iter():
            if it.tag.split("}")[-1] not in ("item", "entry"):
                continue
            f: Dict[str, str] = {}
            for ch in it:
                tag = ch.tag.split("}")[-1].lower()
                f[tag] = (ch.text or "") or ch.attrib.get("href", "")
            blob = f.get("title", "") + " " + f.get("description", "") + " " + f.get("summary", "")
            rel = _RELEASE.search(blob + " " + f.get("link", ""))
            if not rel:
                continue
            urls = _PDF_HREF.findall(blob) + ([f["link"]] if f.get("link", "").lower().endswith(".pdf") else [])
            date = _parse_date(_strip_tags(f.get("description", ""))) or _parse_date(f.get("pubdate", "") or f.get("updated", "") or f.get("date", ""))
            out.append({"release": rel.group(1), "date": date, "title": _strip_tags(f.get("title", "")), "urls": _abs(urls), "link": f.get("link", "")})
    else:
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
            flat = _strip_tags(row)
            rel = _RELEASE.search(flat) or _RELEASE.search(row)
            if not rel:
                continue
            dt = re.search(r"""datetime\s*=\s*["'](\d{4}-\d{2}-\d{2})""", row)
            date = dt.group(1) if dt else _parse_date(flat)
            cells = [_strip_tags(c) for c in re.findall(r"<td\b[^>]*>(.*?)</td>", row, flags=re.S | re.I)]
            cand = [c for c in cells if c and not _RELEASE.fullmatch(c) and _parse_date(c) is None]
            title = max(cand, key=len) if cand else flat
            hrefs = re.findall(r"""href\s*=\s*["']([^"']+)["']""", row, flags=re.I)
            page = [h for h in hrefs if not h.lower().split("?")[0].endswith(".pdf")]
            out.append({"release": rel.group(1), "date": date, "title": title, "urls": _abs(_PDF_HREF.findall(row)), "link": _abs(page[:1])[0] if page else ""})
    seen, uniq = set(), []
    for it in out:
        k = (it["release"], it["date"])
        if k not in seen:
            seen.add(k); uniq.append(it)
    return uniq


def _abs(urls: List[str]) -> List[str]:
    return [u if u.startswith("http") else "https://www.sec.gov" + (u if u.startswith("/") else "/" + u) for u in urls]


_STOP = {"OTC", "OTCQB", "OTCQX", "OTCBB", "SEC", "CIK", "INC", "LLC", "LTD", "CORP", "THE", "AND", "FOR", "NYSE", "NASDAQ", "PINK", "LINK", "USA", "US",
         "EDT", "EST", "ATS", "CEO", "CFO", "IS", "OF", "ON", "IN", "TO", "AS", "BY", "OR", "ITS", "WAS", "ARE", "NO", "NOS"}
_T = r"([A-Z]{2,5})(?![A-Za-z])"
# R1  ... ticker symbol "ABCD" / under the symbol ABCD / ticker: ABCD / symbols "ABCD" and "EFGH" (the tail is scanned by R3)
_R1 = re.compile(r"(?:ticker|symbol)s?\b[\s:,\-–]*(?:(?:of|is|was|are|were)\s+)?[\"“”'‘’(\[]*" + _T)
# R2  (OTC: ABCD) / (OTC Pink: ABCD) / (NASDAQ: ABCD) / (NYSE American: ABCD) / OTC Link: ABCD
_R2 = re.compile(r"(?:OTC(?:QB|QX|BB)?(?:\s+(?:Pink|Link|Markets?))?|Pink\s+Sheets?|NASDAQ|Nasdaq|NYSE(?:\s+American|\s+Arca)?|AMEX)\s*:\s*" + _T)
# R3  further quoted symbols in the 80 characters after an R1 hit, up to the sentence end: , "EFGH" and "IJKL"
_R3 = re.compile(r"[\"“‘']" + _T + r"[\"”’']")


def extract_ticker(text: str) -> List[str]:
    """Ticker symbols named in the text of a suspension order, in order of first appearance (possibly empty).

    Three case-sensitive regexes on the whitespace-normalised text; a symbol is 2-5 capital letters not followed by a letter:
      R1  (ticker|symbol)[s] [:,-] [of|is|was|are|were] [quote/bracket] SYMBOL     e.g.  under the ticker symbol "ABCD"
      R2  (OTC|OTCQB|OTCQX|OTCBB [Pink|Link|Markets] | Pink Sheets | NASDAQ | NYSE [American|Arca] | AMEX) : SYMBOL
      R3  after an R1 hit, further QUOTED symbols within 80 characters and before the next full stop   e.g.  "ABCD" and "EFGH"
    Matches in a stop list of capitalised non-tickers (OTC, SEC, CIK, INC, ...) are discarded. Orders that give no symbol
    (many pre-2012 orders name only the issuer) return []."""
    t = re.sub(r"\s+", " ", text)
    hits = []
    for m in _R1.finditer(t):
        hits.append((m.start(1), m.group(1)))
        tail = t[m.end(1): m.end(1) + 80].split(". ")[0]
        for m3 in _R3.finditer(tail):
            hits.append((m.end(1) + m3.start(1), m3.group(1)))
    for m in _R2.finditer(t):
        hits.append((m.start(1), m.group(1)))
    out: List[str] = []
    for _, s in sorted(hits):
        if s not in _STOP and s not in out:
            out.append(s)
    return out


def _import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def order_text(path: str) -> str:
    """Text of an order: .txt as is; .pdf through pypdf if importable, else the pre-extracted .txt with the same stem."""
    if path.lower().endswith(".txt"):
        with open(path, "r", errors="replace") as f:
            return f.read()
    txt = os.path.splitext(path)[0] + ".txt"
    pypdf = _import("pypdf")
    if pypdf is None:
        if os.path.isfile(txt):
            return order_text(txt)
        raise FileNotFoundError(f"pypdf is not importable and {txt} does not exist; run `pdftotext {path} {txt}` or `pip install pypdf`")
    reader = pypdf.PdfReader(path)
    return "\n".join((pg.extract_text() or "") for pg in reader.pages)


def find_order(root: str, item: Dict[str, Any]) -> Optional[str]:
    """Local order file of a list item: same base name as one of its URLs, else any file whose name contains the release number."""
    d = os.path.join(root, "orders")
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(d))
    stems = [os.path.splitext(os.path.basename(u.split("?")[0]))[0] for u in item.get("urls", [])]
    for ext in (".pdf", ".txt"):       # prefer the pdf, order_text falls back to the txt when pypdf is absent
        for st in stems:
            if st + ext in files:
                return os.path.join(d, st + ext)
        for fn in files:
            if fn.lower().endswith(ext) and item["release"] in fn:
                return os.path.join(d, fn)
    return None


def read_list(root: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(root, "list*.htm*")) + glob.glob(os.path.join(root, "rss*.xml"))) if os.path.isdir(root) else []
    if not files:
        raise missing(os.path.join(root, "list_p0.html"), RECIPE)
    items: Dict[str, Dict[str, Any]] = {}
    for fn in files:
        with open(fn, "r", errors="replace") as f:
            for it in parse_suspension_list(f.read()):
                old = items.get(it["release"])
                if old is None:
                    items[it["release"]] = dict(it, list_file=os.path.basename(fn))
                else:                                    # HTML row and RSS item of the same release complement each other
                    old["urls"] = old["urls"] + [u for u in it["urls"] if u not in old["urls"]]
                    old["date"] = old["date"] or it["date"]
    return sorted(items.values(), key=lambda x: (x["date"] or "", x["release"]))


def _unavailable(root: str) -> set:
    p = os.path.join(root, "prices", "_unavailable.txt")
    if not os.path.isfile(p):
        return set()
    with open(p) as f:
        return {ln.strip().upper() for ln in f if ln.strip()}


def fetch_prices(root: str, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """yfinance daily history -> prices/<TICKER>.csv; an empty answer is appended to prices/_unavailable.txt (returns None)."""
    yf = _import("yfinance")
    if yf is None:
        raise missing(os.path.join(root, "prices", ticker + ".csv"), RECIPE)
    os.makedirs(os.path.join(root, "prices"), exist_ok=True)
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df is None or len(df) == 0:
        with open(os.path.join(root, "prices", "_unavailable.txt"), "a") as f:
            f.write(ticker + "\n")
        return None
    df.to_csv(os.path.join(root, "prices", ticker + ".csv"))
    return read_prices(root, ticker)


def read_prices(root: str, ticker: str) -> Optional[pd.DataFrame]:
    """Cached daily prices -> DataFrame [t (int64 ns, midnight UTC of the trading day), open, high, low, close, volume] or None."""
    p = os.path.join(root, "prices", ticker + ".csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    dcol = "date" if "date" in df.columns else df.columns[0]
    need = ["open", "high", "low", "close", "volume"]
    lack = [c for c in need if c not in df.columns]
    if lack:
        raise ValueError(f"{p}: missing columns {lack}")
    day = df[dcol].astype(str).str.slice(0, 10)                       # exchange-local trading date; drop the time / UTC offset
    out = pd.DataFrame({"t": pd.to_datetime(day, format="%Y-%m-%d").values.astype("datetime64[ns]").astype(np.int64)})
    for c in need:
        out[c] = pd.to_numeric(df[c], errors="coerce").values
    out = out.dropna(subset=["close"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return out[out["close"] > 0].reset_index(drop=True)


def daily_frame(px: pd.DataFrame, ticker: str, t_susp: int, pre_d: int, post_d: int, label_d: int, min_bars: int, meta: Dict[str, Any], min_pre_bars: int = 10):
    """BarFrame of one suspension, or a string with the reason it cannot be built."""
    from ...detect.features import BAR_FEATURES
    w = px[(px["t"] >= t_susp - pre_d * DAY_NS) & (px["t"] <= t_susp + post_d * DAY_NS)].reset_index(drop=True)
    t = w["t"].values.astype(np.int64)
    lab0 = t_susp - label_d * DAY_NS
    labels = ((t >= lab0) & (t <= t_susp)).astype(np.int8)
    if len(w) < min_bars:
        return f"only {len(w)} daily bars in the frame window (< {min_bars})"
    if labels.sum() == 0:
        return "no daily bar inside the label window"
    pre = w["volume"].values[(t < lab0) & (w["volume"].values > 0)]
    if len(pre) < min_pre_bars:
        return f"only {len(pre)} traded days before the label window (< {min_pre_bars}; volume normaliser undefined)"
    vref = float(np.median(pre))
    c = w["close"].values.astype(np.float64)
    hi = np.where(np.isfinite(w["high"].values) & (w["high"].values > 0), w["high"].values, c)
    lo = np.where(np.isfinite(w["low"].values) & (w["low"].values > 0), w["low"].values, c)
    X = np.zeros((len(w), len(BAR_FEATURES)), dtype=np.float64)
    X[1:, BAR_FEATURES.index("ret")] = np.diff(np.log(c))
    X[:, BAR_FEATURES.index("rv")] = np.log(np.maximum(hi, lo) / np.minimum(hi, lo))
    X[:, BAR_FEATURES.index("vol_trade")] = np.nan_to_num(w["volume"].values.astype(np.float64)) / vref
    mask = np.array([f in ("ret", "rv", "vol_trade") for f in BAR_FEATURES])
    ep = Episode("pump", "real", int(lab0), int(t_susp), 1, -1, {"weak": True, "release": meta.get("release", "")}, eid=0)
    meta = dict(meta); meta.update({"dt_ns": DAY_NS, "vref": vref, "trade_only": True, "daily": True, "irregular_times": True, "weak_label": True,
                                    "episodes": [ep.to_dict()], "n_bars_after_suspension": int((t > t_susp).sum())})
    fr = BarFrame(ticker, t, np.nan_to_num(X).astype(np.float32), list(BAR_FEATURES), labels, np.where(labels == 1, 0, -1).astype(np.int32), mask,
                  np.where(labels == 1, "pump", "").astype(str), meta)
    fr.validate()
    return fr


def prepare(root: str = DEFAULT_ROOT, download: bool = False, user_agent: Optional[str] = None, pages: int = 1) -> Dict[str, Any]:
    """download=True: fetch `pages` list pages, the RSS feed and the linked order PDFs (needs user_agent = "Name email",
    sec.gov fair-access rule; <= 10 requests / s). Prices are fetched lazily by the loader (yfinance) or by fetch_prices."""
    if download:
        if not user_agent:
            raise ValueError('sec.gov requires a declared User-Agent: prepare(root, download=True, user_agent="Your Name you@example.org")')
        import time
        import urllib.request

        def get(url, dst):
            if os.path.isfile(dst):
                return
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req) as r, open(dst + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dst + ".part", dst); time.sleep(0.25)

        os.makedirs(os.path.join(root, "orders"), exist_ok=True)
        get(RSS_URL, os.path.join(root, "rss.xml"))
        for p in range(int(pages)):
            get(f"{LIST_URL}?page={p}", os.path.join(root, f"list_p{p}.html"))
        for it in read_list(root):
            for u in it["urls"]:
                get(u, os.path.join(root, "orders", os.path.basename(u.split("?")[0])))
    items = read_list(root)
    with_order = [it for it in items if find_order(root, it)]
    if not with_order:
        raise missing(os.path.join(root, "orders", "<release>.pdf"), RECIPE)
    n_px = len(glob.glob(os.path.join(root, "prices", "*.csv")))
    return {"root": root, "n_items": len(items), "n_with_order": len(with_order), "n_price_files": n_px, "n_unavailable": len(_unavailable(root)),
            "pypdf": _import("pypdf") is not None, "yfinance": _import("yfinance") is not None}


@register_loader(KEY)
def load_sec_suspensions(cfg: Dict[str, Any]) -> List[BarFrame]:
    """cfg: root, pre_d (120), post_d (20), label_d (30), min_bars (20), min_pre_bars (10), symbols, dates (suspension dates), max_sessions."""
    cfg = cfg or {}
    root = cfg.get("root", DEFAULT_ROOT)
    pre_d, post_d, label_d, min_bars = int(cfg.get("pre_d", 120)), int(cfg.get("post_d", 20)), int(cfg.get("label_d", 30)), int(cfg.get("min_bars", 20))
    items = read_list(root)
    if not any(find_order(root, it) for it in items):
        raise missing(os.path.join(root, "orders", "<release>.pdf"), RECIPE)
    want_s = {str(s).upper() for s in cfg["symbols"]} if cfg.get("symbols") else None
    want_d = {str(d) for d in cfg["dates"]} if cfg.get("dates") else None
    del DROPPED[:]
    unavailable = _unavailable(root)
    frames: List[BarFrame] = []
    n_tickers = 0
    for it in items:
        if cfg.get("max_sessions") and len(frames) >= int(cfg["max_sessions"]):
            break
        if want_d is not None and it["date"] not in want_d:
            continue
        base = {"release": it["release"], "date": it["date"], "title": it["title"]}
        if it["date"] is None:
            DROPPED.append(dict(base, ticker=None, reason="no date in the list row")); continue
        path = find_order(root, it)
        if path is None:
            DROPPED.append(dict(base, ticker=None, reason="order file not downloaded")); continue
        try:
            tickers = extract_ticker(order_text(path))
        except FileNotFoundError as e:
            DROPPED.append(dict(base, ticker=None, reason=str(e))); continue
        if not tickers:
            DROPPED.append(dict(base, ticker=None, reason="no ticker symbol in the order text")); continue
        t_susp = int(np.datetime64(it["date"], "ns").astype(np.int64))
        for tk in tickers:
            if want_s is not None and tk not in want_s:
                continue
            n_tickers += 1
            px = read_prices(root, tk)
            if px is None and tk not in unavailable:
                start = str(np.datetime64(it["date"], "D") - pre_d - 5); end = str(np.datetime64(it["date"], "D") + post_d + 5)
                px = fetch_prices(root, tk, start, end)        # raises FileNotFoundError with the recipe when yfinance is absent
            if px is None or len(px) == 0:
                DROPPED.append(dict(base, ticker=tk, reason="no Yahoo price history (delisted / never covered)")); continue
            meta = {"source": KEY, "symbol": tk, "date": it["date"], "licence": LICENCE, "release": it["release"], "respondents": it["title"],
                    "provenance": {"list_url": LIST_URL, "order_urls": it["urls"], "order_file": os.path.basename(path), "prices": "Yahoo Finance via yfinance"},
                    "pre_d": pre_d, "post_d": post_d, "label_d": label_d,
                    "label_rule": "daily bars with suspension - label_d <= t <= suspension (weak label)"}
            fr = daily_frame(px, tk, t_susp, pre_d, post_d, label_d, min_bars, meta, int(cfg.get("min_pre_bars", 10)))
            if isinstance(fr, str):
                DROPPED.append(dict(base, ticker=tk, reason=fr)); continue
            frames.append(fr)
    summary = {"n_list_items": len(items), "n_tickers": n_tickers, "n_frames": len(frames), "n_dropped": len(DROPPED)}
    for fr in frames:
        fr.meta["survivorship_dropped"] = [dict(d) for d in DROPPED]
        fr.meta["survivorship_summary"] = dict(summary)
    if not frames:
        raise ValueError(f"no suspension produced a frame ({summary}); first drops: {DROPPED[:3]}")
    return frames
