"""sec_suspensions loader: synthetic list page / RSS / orders / price CSVs in a temp dir (no network, yfinance stubbed).
Run: python3 tests/maniparena_parts/test_sec_suspensions.py"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from maniparena.data.loaders import sec_suspensions as ss
from maniparena.data.loaders.common import LOADERS

DAY = 86400 * 1_000_000_000

LIST_HTML = """<html><body><table class="views-table">
<thead><tr><th>Date</th><th>Release No.</th><th>Respondents</th></tr></thead><tbody>
<tr><td><time datetime="2019-06-28T12:00:00Z">June 28, 2019</time></td><td>34-86252</td>
    <td><a href="/files/litigation/suspensions/2019/34-86252.pdf">Alpha Beta Holdings, Inc.</a><br/>See also: <a href="/files/litigation/suspensions/2019/34-86252-o.pdf">Order</a></td></tr>
<tr><td>Mar. 5, 2020</td><td>34-88321</td><td><a href="/files/litigation/suspensions/2020/34-88321.pdf">Gamma Corp. &amp; Delta Ltd.</a></td></tr>
<tr><td>04/10/2020</td><td>34-88610</td><td><a href="https://www.sec.gov/files/litigation/suspensions/2020/34-88610.pdf">Epsilon Mining Co.</a></td></tr>
<tr><td>May 1, 2020</td><td>34-88777</td><td><a href="/files/litigation/suspensions/2020/34-88777.pdf">Zeta Old Issuer</a></td></tr>
<tr><td>May 4, 2020</td><td>34-88800</td><td><a href="/files/litigation/suspensions/2020/34-88800.pdf">Eta Not Downloaded</a></td></tr>
<tr><td>no release here</td><td>n/a</td><td>ignored row</td></tr>
</tbody></table></body></html>"""

RSS_XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>Trading Suspensions</title>
<item><title>Theta Thinly Traded Inc.</title><link>https://www.sec.gov/files/litigation/suspensions/2020/34-88900.pdf</link>
<description>Release No. 34-88900; May 12, 2020</description><pubDate>Tue, 12 May 2020 16:00:00 -0400</pubDate></item>
<item><title>Alpha Beta Holdings, Inc.</title><link>https://www.sec.gov/enforcement-litigation/trading-suspensions/34-86252</link>
<description>34-86252</description><pubDate>Fri, 28 Jun 2019 12:00:00 -0400</pubDate></item>
</channel></rss>"""


def make_pdf(path, lines):
    """Minimal one-page PDF with a Helvetica text stream (enough for pypdf.extract_text)."""
    esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = "BT /F1 11 Tf 50 740 Td 14 TL " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = b"%PDF-1.4\n"; offs = []
    for i, o in enumerate(objs):
        offs.append(len(out)); out += f"{i + 1} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)


def write_prices(root, ticker, susp, start_off=-200, end_off=40, seed=0, pump=True, gap_after=10, yf_style=True):
    """Business-day OHLCV in the yfinance CSV layout; volume x8 and drift in the 30 days before the suspension; no bars in the
    `gap_after` calendar days after it (trading halted)."""
    rng = np.random.default_rng(seed)
    d0 = np.datetime64(susp, "D")
    days = np.array([d for d in np.arange(d0 + start_off, d0 + end_off + 1) if np.is_busday(d) and not (d0 < d <= d0 + gap_after)])
    inwin = (days >= d0 - 30) & (days <= d0)
    ret = rng.normal(0, 0.02, len(days)) + (0.03 if pump else 0.0) * inwin
    close = 0.5 * np.exp(np.cumsum(ret))
    vol = rng.integers(8000, 12000, len(days)) * np.where(inwin & pump, 8, 1)
    df = pd.DataFrame({"Date": [str(d) + (" 00:00:00-04:00" if yf_style else "") for d in days], "Open": close * 0.99, "High": close * 1.05, "Low": close * 0.95,
                       "Close": close, "Adj Close": close, "Volume": vol})
    os.makedirs(os.path.join(root, "prices"), exist_ok=True)
    df.to_csv(os.path.join(root, "prices", ticker + ".csv"), index=False)
    return days, close, vol


class FakeYF:
    """Stands in for yfinance: knows nothing (Yahoo dropped the ticker). Records the calls; never touches the network."""
    calls = []

    class Ticker:
        def __init__(self, t):
            self.t = t

        def history(self, start=None, end=None, auto_adjust=False):
            FakeYF.calls.append((self.t, start, end))
            return pd.DataFrame()


def run():
    assert "sec_suspensions" in LOADERS and isinstance(ss.RECIPE, str) and callable(ss.prepare)

    # --- list / RSS parsing
    items = ss.parse_suspension_list(LIST_HTML)
    assert [(i["release"], i["date"]) for i in items] == [("34-86252", "2019-06-28"), ("34-88321", "2020-03-05"), ("34-88610", "2020-04-10"),
                                                           ("34-88777", "2020-05-01"), ("34-88800", "2020-05-04")], items
    assert items[0]["urls"] == ["https://www.sec.gov/files/litigation/suspensions/2019/34-86252.pdf", "https://www.sec.gov/files/litigation/suspensions/2019/34-86252-o.pdf"]
    assert items[1]["title"] == "Gamma Corp. & Delta Ltd." and items[2]["urls"][0].startswith("https://www.sec.gov/files/")
    rss = ss.parse_suspension_list(RSS_XML)
    assert [(i["release"], i["date"]) for i in rss] == [("34-88900", "2020-05-12"), ("34-86252", "2019-06-28")], rss
    assert rss[0]["urls"] == ["https://www.sec.gov/files/litigation/suspensions/2020/34-88900.pdf"] and rss[1]["urls"] == []
    assert ss.parse_suspension_list("<html><body>nothing</body></html>") == []

    # --- ticker extraction
    cases = [('quoted on OTC Link operated by OTC Markets Group Inc. under the ticker symbol "ABHI". The Commission', ["ABHI"]),
             ("the securities of Gamma Corp. (ticker symbol “GMMA”) and Delta Ltd. (OTC Pink: DLTA), CIK No. 0001234567", ["GMMA", "DLTA"]),
             ('trading under the symbols "AAAA" and "BBBB". A later sentence quoting "CCCC" is not a symbol list', ["AAAA", "BBBB"]),
             ("Epsilon Mining Co. (NASDAQ: EPSM) because of questions regarding the accuracy", ["EPSM"]),
             ("its common stock (symbol: ZT) is quoted on OTC Link", ["ZT"]),
             ("It appears to the Securities and Exchange Commission that there is a lack of current and accurate information concerning the securities of Zeta Old Issuer Inc.", []),
             ("ticker symbol of the SEC registrant is unknown; quoted on the OTC: market", [])]
    for text, want in cases:
        assert ss.extract_ticker(text) == want, (text, ss.extract_ticker(text), want)

    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "orders"))
        with open(os.path.join(root, "list_p0.html"), "w") as f:
            f.write(LIST_HTML)
        with open(os.path.join(root, "rss.xml"), "w") as f:
            f.write(RSS_XML)
        have_pypdf = ss._import("pypdf") is not None
        order1 = ["UNITED STATES OF AMERICA before the SECURITIES AND EXCHANGE COMMISSION", "ORDER OF SUSPENSION OF TRADING",
                  "It appears that there is a lack of adequate information concerning Alpha Beta Holdings, Inc.",
                  'whose stock is quoted on OTC Link under the ticker symbol "ABHI". Trading is suspended.']
        if have_pypdf:
            make_pdf(os.path.join(root, "orders", "34-86252.pdf"), order1)
            assert "ABHI" in ss.order_text(os.path.join(root, "orders", "34-86252.pdf")), "pypdf must read the synthetic order"
        else:
            with open(os.path.join(root, "orders", "34-86252.txt"), "w") as f:
                f.write("\n".join(order1))
        with open(os.path.join(root, "orders", "34-88321.txt"), "w") as f:        # pre-extracted text, two issuers in one order
            f.write("the securities of Gamma Corp. (ticker symbol “GMMA”) and Delta Ltd. (OTC Pink: DLTA)")
        with open(os.path.join(root, "orders", "34-88610.txt"), "w") as f:
            f.write("Epsilon Mining Co. (NASDAQ: EPSM) because of questions")
        with open(os.path.join(root, "orders", "34-88777.txt"), "w") as f:        # no symbol in the text
            f.write("lack of current and accurate information concerning the securities of Zeta Old Issuer Inc.")
        with open(os.path.join(root, "orders", "release_34-88900_order.txt"), "w") as f:   # matched through the release number
            f.write("Theta Thinly Traded Inc., ticker symbol THTT.")

        days, close, vol = write_prices(root, "ABHI", "2019-06-28", seed=1)
        write_prices(root, "GMMA", "2020-03-05", seed=2, yf_style=False)
        write_prices(root, "THTT", "2020-05-12", start_off=-35, seed=3)            # history starts 35 d before: too short
        with open(os.path.join(root, "prices", "_unavailable.txt"), "w") as f:
            f.write("DLTA\n")                                                      # Yahoo has nothing: survivorship drop

        # pypdf absent -> the pre-extracted txt is used, or a FileNotFoundError tells how to make it
        real_import = ss._import
        try:
            ss._import = lambda name: None
            p = os.path.join(root, "orders", "fake.pdf")
            open(p, "wb").close()
            try:
                ss.order_text(p)
            except FileNotFoundError as e:
                assert "pdftotext" in str(e)
            else:
                raise AssertionError("expected FileNotFoundError without pypdf and without txt")
            with open(os.path.join(root, "orders", "fake.txt"), "w") as f:
                f.write("symbol FAKE")
            assert ss.extract_ticker(ss.order_text(p)) == ["FAKE"]
            os.remove(p); os.remove(os.path.join(root, "orders", "fake.txt"))

            # EPSM has neither a price file nor an unavailable mark, and yfinance is not importable -> recipe, no fabricated prices
            try:
                ss._import = lambda name: None if name == "yfinance" else real_import(name)
                LOADERS["sec_suspensions"]({"root": root})
            except FileNotFoundError as e:
                assert "EPSM.csv" in str(e) and "yfinance" in str(e) and "never fabricate" in str(e), str(e)
            else:
                raise AssertionError("expected FileNotFoundError for missing prices without yfinance")

            # yfinance importable but Yahoo returns nothing -> dropped, marked unavailable, no network
            ss._import = lambda name: FakeYF if name == "yfinance" else real_import(name)
            frames = LOADERS["sec_suspensions"]({"root": root})
        finally:
            ss._import = real_import
        assert FakeYF.calls == [("EPSM", "2019-12-07", "2020-05-05")], FakeYF.calls
        assert "EPSM" in ss._unavailable(root)

        assert [f.symbol for f in frames] == ["ABHI", "GMMA"], [f.symbol for f in frames]
        f = frames[0]
        f.validate()
        t_s = int(np.datetime64("2019-06-28", "ns").astype(np.int64))
        assert f.meta["dt_ns"] == DAY and f.meta["daily"] and f.meta["weak_label"] and f.meta["source"] == "sec_suspensions" and f.meta["release"] == "34-86252"
        assert f.times.min() >= t_s - 120 * DAY and f.times.max() <= t_s + 20 * DAY and np.all(np.diff(f.times) >= DAY)
        want = (f.times >= t_s - 30 * DAY) & (f.times <= t_s)
        assert np.array_equal(f.labels == 1, want) and 20 <= f.labels.sum() <= 23, f.labels.sum()
        ep = f.meta["episodes"][0]
        assert ep["family"] == "pump" and ep["difficulty"] == "real" and ep["t0"] == t_s - 30 * DAY and ep["t1"] == t_s and f.n_episodes() == 1
        assert not ((f.times > t_s) & (f.times <= t_s + 10 * DAY)).any() and f.meta["n_bars_after_suspension"] == int((f.times > t_s).sum()) > 0
        names = f.feat_names
        assert [n for n, m in zip(names, f.feat_mask) if m] == ["ret", "rv", "vol_trade"]
        sel = np.isin(days.astype("datetime64[ns]").astype(np.int64), f.times)
        assert np.allclose(f.feats[1:, names.index("ret")], np.diff(np.log(close[sel])), atol=1e-5)
        assert np.allclose(f.feats[:, names.index("rv")], np.log(1.05 / 0.95), atol=1e-5)
        pre = vol[sel][f.times < t_s - 30 * DAY]
        assert abs(f.meta["vref"] - float(np.median(pre))) < 1e-6, "volume normaliser must come from bars before the label window only"
        v = f.feats[:, names.index("vol_trade")]
        assert v[f.labels == 1].mean() > 4 * v[f.times < t_s - 30 * DAY].mean()
        # survivorship bookkeeping
        dropped = f.meta["survivorship_dropped"]
        got = {(d["release"], d["ticker"]): d["reason"] for d in dropped}
        assert set(got) == {("34-88321", "DLTA"), ("34-88610", "EPSM"), ("34-88777", None), ("34-88800", None), ("34-88900", "THTT")}, got
        assert "Yahoo" in got[("34-88321", "DLTA")] and "Yahoo" in got[("34-88610", "EPSM")] and "no ticker" in got[("34-88777", None)]
        assert "not downloaded" in got[("34-88800", None)] and "before the label window" in got[("34-88900", "THTT")]
        assert f.meta["survivorship_summary"] == {"n_list_items": 6, "n_tickers": 5, "n_frames": 2, "n_dropped": 5}, f.meta["survivorship_summary"]
        assert frames[1].meta["survivorship_dropped"] == dropped

        # filters
        assert [x.symbol for x in LOADERS["sec_suspensions"]({"root": root, "symbols": ["GMMA"]})] == ["GMMA"]
        assert [x.symbol for x in LOADERS["sec_suspensions"]({"root": root, "dates": ["2019-06-28"]})] == ["ABHI"]
        assert len(LOADERS["sec_suspensions"]({"root": root, "max_sessions": 1})) == 1
        short = LOADERS["sec_suspensions"]({"root": root, "symbols": ["ABHI"], "pre_d": 60, "post_d": 0, "label_d": 10})[0]
        assert short.times.max() <= t_s and short.times.min() >= t_s - 60 * DAY and 7 <= short.labels.sum() <= 9
        try:
            LOADERS["sec_suspensions"]({"root": root, "symbols": ["DLTA"]})
        except ValueError as e:
            assert "no suspension produced a frame" in str(e)
        else:
            raise AssertionError("expected ValueError when every candidate is dropped")
        info = ss.prepare(root)
        assert info["n_items"] == 6 and info["n_with_order"] == 5 and info["n_price_files"] == 3 and info["n_unavailable"] == 2

    # --- nothing on disk -> recipe
    with tempfile.TemporaryDirectory() as empty:
        for call in (lambda: LOADERS["sec_suspensions"]({"root": empty}), lambda: ss.prepare(empty), lambda: LOADERS["sec_suspensions"]({"root": os.path.join(empty, "x")})):
            try:
                call()
            except FileNotFoundError as e:
                assert "sec.gov/enforcement-litigation/trading-suspensions" in str(e) and "User-Agent" in str(e)
            else:
                raise AssertionError("expected FileNotFoundError")
        with open(os.path.join(empty, "list_p0.html"), "w") as f:
            f.write(LIST_HTML)
        try:
            LOADERS["sec_suspensions"]({"root": empty})
        except FileNotFoundError as e:
            assert "orders" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError without orders")
        try:
            ss.prepare(empty, download=True)
        except ValueError as e:
            assert "User-Agent" in str(e)
        else:
            raise AssertionError("download without a declared User-Agent must be refused")
    print(f"pypdf path exercised: {have_pypdf}")
    print("test_sec_suspensions OK")


if __name__ == "__main__":
    run()
