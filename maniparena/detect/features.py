"""Feature pipeline shared by every detector and every dataset.

bar_features(session, dt_ns)   -> BarFrame   fixed-time bars; the LOB-lite schema below (F = 26) is computable from any
                                              message + snapshot stream; datasets without a book (pump-and-dump trade
                                              data) set feat_mask=False for the book columns
event_features(session, L)     -> (T, 4L + 13) float32 per-message rows: raw L-level book columns + the 13 "manual"
                                              features used by the JFDS / cascaded-contrastive baselines
OnlineBars                     incremental version of bar_features for in-simulation detector feedback (RL reward)

LOB-lite bar schema (all book quantities are per-bar means unless stated; sizes normalised by the session's median
level-1 depth `dref`, prices by the tick; the normalisation constants are stored in BarFrame.meta so that a detector
trained on simulated bars can be applied to real bars with the same normalisation rule):
     0 ret            log mid return over the bar                      13 n_exec         executions
     1 rv             realised vol of mid inside the bar (ticks)       14 cancel_ratio   deletes / (adds + 1)
     2 spread         mean spread (ticks)                              15 vol_trade      traded shares / dref
     3 imb1           (b1 - a1) / (b1 + a1)                            16 vol_add_bid    added bid shares / dref
     4 imb3           top-3 imbalance                                  17 vol_add_ask    added ask shares / dref
     5 imb5           top-5 imbalance                                  18 vol_del_bid    deleted bid shares / dref
     6 depth_bid5     top-5 bid depth / dref                           19 vol_del_ask    deleted ask shares / dref
     7 depth_ask5     top-5 ask depth / dref                           20 max_add        largest add / dref
     8 depth_l1       (b1 + a1) / 2 / dref                             21 add_dist       mean distance of adds from best (ticks)
     9 ofi            order-flow imbalance (Cont et al.) / dref        22 far_add_frac   share of adds >= 3 ticks behind best
    10 msg_rate       messages per second / median rate               23 del_life       mean lifetime of deleted orders (s, log1p)
    11 n_add          adds                                            24 imb_change     imb3 at bar end - imb3 at bar start
    12 n_del          deletes                                         25 mid_disp       (mid_end - mid_start) sign-aligned with imb3 start
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..data.schema import LOBSession, BarFrame, MSG_DTYPE

BAR_FEATURES = ["ret", "rv", "spread", "imb1", "imb3", "imb5", "depth_bid5", "depth_ask5", "depth_l1", "ofi", "msg_rate", "n_add", "n_del", "n_exec",
                "cancel_ratio", "vol_trade", "vol_add_bid", "vol_add_ask", "vol_del_bid", "vol_del_ask", "max_add", "add_dist", "far_add_frac",
                "del_life", "imb_change", "mid_disp"]
BOOK_FEATURES = {"spread", "imb1", "imb3", "imb5", "depth_bid5", "depth_ask5", "depth_l1", "ofi", "add_dist", "far_add_frac", "imb_change", "mid_disp",
                 "n_add", "n_del", "cancel_ratio", "vol_add_bid", "vol_add_ask", "vol_del_bid", "vol_del_ask", "max_add", "del_life"}
NS = 1_000_000_000


def _imb(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    s = b + a
    out = np.zeros_like(s, dtype=np.float64)
    ok = s > 0
    out[ok] = (b[ok] - a[ok]) / s[ok]
    return out


def _ofi(book: np.ndarray) -> np.ndarray:
    """Cont, Kukanov, Stoikov (2014) level-1 order flow imbalance per message."""
    bp, bs, ap, as_ = book[:, 0, 0], book[:, 0, 1], book[:, 0, 2], book[:, 0, 3]
    e = np.zeros(len(book))
    if len(book) < 2:
        return e
    pb, pbs, pa, pas = bp[:-1], bs[:-1], ap[:-1], as_[:-1]
    cb, cbs, ca, cas = bp[1:], bs[1:], ap[1:], as_[1:]
    eb = np.where(cb > pb, cbs, np.where(cb == pb, cbs - pbs, -pbs))
    ea = np.where(ca < pa, cas, np.where(ca == pa, cas - pas, -pas))
    e[1:] = eb - ea
    e[~np.isfinite(e)] = 0.0
    return e


def bar_features(session: LOBSession, dt_ns: int = NS, dref: Optional[float] = None, rate_ref: Optional[float] = None,
                 t_start: Optional[int] = None, t_end: Optional[int] = None) -> BarFrame:
    msgs, book = session.msgs, session.book
    T, L = session.T, session.levels
    if T == 0:
        raise ValueError("empty session")
    t = msgs["t"]
    t0 = int(t[0] if t_start is None else t_start); t1 = int(t[-1] if t_end is None else t_end)
    n_bars = int((t1 - t0) // dt_ns) + 1
    bar = ((t - t0) // dt_ns).astype(np.int64)
    bar = np.clip(bar, 0, n_bars - 1)
    k5 = min(5, L); k3 = min(3, L)
    b1, a1 = book[:, 0, 1], book[:, 0, 3]
    bd5, ad5 = book[:, :k5, 1].sum(1), book[:, :k5, 3].sum(1)
    bd3, ad3 = book[:, :k3, 1].sum(1), book[:, :k3, 3].sum(1)
    mid = session.mid()
    valid = np.isfinite(mid)
    if dref is None:
        dref = float(np.median(((b1 + a1) / 2.0)[valid])) if valid.any() else 1.0
        dref = max(dref, 1.0)
    spread = (book[:, 0, 2] - book[:, 0, 0]).astype(np.float64)
    spread[~valid] = np.nan
    imb1, imb3, imb5 = _imb(b1, a1), _imb(bd3, ad3), _imb(bd5, ad5)
    ofi = _ofi(book)
    typ, side, size, price = msgs["type"], msgs["side"], msgs["size"].astype(np.float64), msgs["price"].astype(np.float64)
    is_add, is_del, is_exec = typ == 1, (typ == 2) | (typ == 3), typ == 4
    best_same = np.where(side == 1, book[:, 0, 0], book[:, 0, 2])
    prev_best = np.r_[best_same[:1], best_same[:-1]]
    dist = np.where(side == 1, prev_best - price, price - prev_best)   # ticks behind the pre-message best (adds)
    dist = np.where(is_add & np.isfinite(dist) & (prev_best > 0), np.clip(dist, -10, 50), np.nan)
    # order lifetime: match deletes to their add time by order id
    life = np.full(T, np.nan)
    if is_add.any() and is_del.any():
        oid = msgs["order_id"]
        add_t: Dict[int, int] = {}
        for i in np.flatnonzero(is_add | is_del):
            if is_add[i]:
                add_t[int(oid[i])] = int(t[i])
            else:
                ta = add_t.get(int(oid[i]))
                if ta is not None:
                    life[i] = (t[i] - ta) / NS

    def agg(x, mask=None, fn="sum"):
        w = np.ones(T) if mask is None else mask.astype(np.float64)
        v = np.nan_to_num(x, nan=0.0) * w
        s = np.bincount(bar, weights=v, minlength=n_bars)
        if fn == "sum":
            return s
        c = np.bincount(bar, weights=w * np.isfinite(x), minlength=n_bars)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    F = len(BAR_FEATURES)
    X = np.zeros((n_bars, F), dtype=np.float64)
    cnt = np.bincount(bar, minlength=n_bars).astype(np.float64)
    # first / last valid mid per bar
    first = np.full(n_bars, np.nan); last = np.full(n_bars, np.nan)
    vi = np.flatnonzero(valid)
    if vi.size:
        bv = bar[vi]
        first_idx = np.full(n_bars, -1); last_idx = np.full(n_bars, -1)
        np.minimum.at(first_idx, bv, vi) if False else None
        # vectorised first/last: sort by bar, take boundaries
        order = np.argsort(bv, kind="stable")
        bs, vs = bv[order], vi[order]
        starts = np.r_[0, np.flatnonzero(np.diff(bs)) + 1]
        ends = np.r_[starts[1:] - 1, len(bs) - 1]
        first[bs[starts]] = mid[vs[starts]]; last[bs[starts]] = mid[vs[ends]]
        imb3_first = np.full(n_bars, np.nan); imb3_last = np.full(n_bars, np.nan)
        imb3_first[bs[starts]] = imb3[vs[starts]]; imb3_last[bs[starts]] = imb3[vs[ends]]
    else:
        imb3_first = imb3_last = np.full(n_bars, np.nan)
    # forward-fill bar mids through empty bars
    def ffill(x):
        x = x.copy(); idx = np.where(np.isfinite(x), np.arange(len(x)), 0); np.maximum.accumulate(idx, out=idx); y = x[idx]
        return y
    last_f = ffill(last); first_f = np.where(np.isfinite(first), first, np.r_[last_f[:1], last_f[:-1]])
    prev_last = np.r_[last_f[:1], last_f[:-1]]
    X[:, 0] = np.nan_to_num(np.log(np.maximum(last_f, 1e-9)) - np.log(np.maximum(prev_last, 1e-9)))
    # realised vol inside bar: sum of squared mid changes
    dm = np.zeros(T); dm[1:] = np.diff(np.nan_to_num(mid, nan=0.0)); dm[~valid] = 0.0; dm[1:][~valid[:-1]] = 0.0
    X[:, 1] = np.sqrt(np.bincount(bar, weights=dm * dm, minlength=n_bars))
    X[:, 2] = agg(spread, fn="mean")
    X[:, 3] = agg(np.where(valid, imb1, np.nan), fn="mean")
    X[:, 4] = agg(np.where(valid, imb3, np.nan), fn="mean")
    X[:, 5] = agg(np.where(valid, imb5, np.nan), fn="mean")
    X[:, 6] = agg(np.where(valid, bd5, np.nan), fn="mean") / dref
    X[:, 7] = agg(np.where(valid, ad5, np.nan), fn="mean") / dref
    X[:, 8] = agg(np.where(valid, (b1 + a1) / 2.0, np.nan), fn="mean") / dref
    X[:, 9] = agg(ofi) / dref
    rate = cnt / (dt_ns / NS)
    if rate_ref is None:
        rate_ref = float(np.median(rate[rate > 0])) if (rate > 0).any() else 1.0
        rate_ref = max(rate_ref, 1e-9)
    X[:, 10] = rate / rate_ref
    X[:, 11] = agg(is_add.astype(float)); X[:, 12] = agg(is_del.astype(float)); X[:, 13] = agg(is_exec.astype(float))
    X[:, 14] = X[:, 12] / (X[:, 11] + 1.0)
    X[:, 15] = agg(size, is_exec) / dref
    X[:, 16] = agg(size, is_add & (side == 1)) / dref; X[:, 17] = agg(size, is_add & (side == -1)) / dref
    X[:, 18] = agg(size, is_del & (side == 1)) / dref; X[:, 19] = agg(size, is_del & (side == -1)) / dref
    mx = np.zeros(n_bars); np.maximum.at(mx, bar[is_add], size[is_add]); X[:, 20] = mx / dref
    X[:, 21] = np.nan_to_num(agg(dist, is_add, fn="mean"))
    far = np.where(is_add, (dist >= 3).astype(float), np.nan); X[:, 22] = np.nan_to_num(agg(far, is_add, fn="mean"))
    X[:, 23] = np.nan_to_num(np.log1p(agg(life, is_del, fn="mean")))
    X[:, 24] = np.nan_to_num(imb3_last - imb3_first)
    X[:, 25] = np.nan_to_num((last_f - first_f) * np.sign(np.nan_to_num(imb3_first)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # labels: a bar is positive if a manipulation episode overlaps it; episode id of the first overlapping episode
    labels = np.zeros(n_bars, dtype=np.int8); eid = np.full(n_bars, -1, dtype=np.int32); fam = np.array([""] * n_bars, dtype=object)
    for e in session.episodes:
        lo = int(max(0, (e.t0 - t0) // dt_ns)); hi = int(min(n_bars - 1, (e.t1 - t0) // dt_ns))
        if hi < 0 or lo >= n_bars:
            continue
        labels[lo:hi + 1] = 1
        sel = slice(lo, hi + 1)
        eid[sel] = np.where(eid[sel] < 0, e.eid, eid[sel]); fam[sel] = np.where(fam[sel] == "", e.family, fam[sel])
    if session.manip.any() and not session.episodes:            # real data with message-level labels only
        labels = (np.bincount(bar, weights=session.manip.astype(float), minlength=n_bars) > 0).astype(np.int8)
    times = t0 + np.arange(n_bars, dtype=np.int64) * dt_ns
    meta = {"dt_ns": int(dt_ns), "dref": float(dref), "rate_ref": float(rate_ref), "levels": int(L), "symbol": session.symbol, "source": session.meta.get("source", ""),
            "n_msgs": int(T), "empty_bars": int((cnt == 0).sum())}
    return BarFrame(session.symbol, times, X, list(BAR_FEATURES), labels, eid, np.ones(F, dtype=bool), fam.astype(str), meta)


def event_features(session: LOBSession, L: int = 5, dref: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """Per-message rows: [bid_px, bid_sz, ask_px, ask_sz] x L (prices relative to the mid in ticks, sizes / dref) + 13 manual
    features: spread, mid change, imb1, imb3, imb5, ofi, bid depth L, ask depth L, msg type one-hot(add/del/exec), size/dref, side."""
    book = session.book[:, :L]
    mid = session.mid()
    mid_f = np.where(np.isfinite(mid), mid, np.nan)
    if dref is None:
        dref = max(float(np.nanmedian((book[:, 0, 1] + book[:, 0, 3]) / 2.0)), 1.0)
    cols = []
    names = []
    for k in range(L):
        cols += [np.nan_to_num(book[:, k, 0] - mid_f), book[:, k, 1] / dref, np.nan_to_num(book[:, k, 2] - mid_f), book[:, k, 3] / dref]
        names += [f"bp{k+1}", f"bs{k+1}", f"ap{k+1}", f"as{k+1}"]
    b1, a1 = book[:, 0, 1], book[:, 0, 3]
    k3 = min(3, L)
    dmid = np.zeros(len(mid)); dmid[1:] = np.nan_to_num(np.diff(mid_f))
    typ = session.msgs["type"]
    manual = [np.nan_to_num(book[:, 0, 2] - book[:, 0, 0]), dmid, _imb(b1, a1), _imb(book[:, :k3, 1].sum(1), book[:, :k3, 3].sum(1)),
              _imb(book[:, :, 1].sum(1), book[:, :, 3].sum(1)), _ofi(session.book) / dref, book[:, :, 1].sum(1) / dref, book[:, :, 3].sum(1) / dref,
              (typ == 1).astype(float), ((typ == 2) | (typ == 3)).astype(float), (typ == 4).astype(float), session.msgs["size"] / dref, session.msgs["side"].astype(float)]
    names += ["spread", "dmid", "imb1", "imb3", "imbL", "ofi", "bdepth", "adepth", "is_add", "is_del", "is_exec", "size", "side"]
    X = np.stack(cols + manual, 1).astype(np.float32)
    return np.nan_to_num(X), names


class OnlineBars:
    """Incrementally turns a simulator's message log into LOB-lite bars (same schema as bar_features) for in-simulation
    detector feedback. Call `update(book, t)` at every manipulator decision step; `window(Lw)` returns the last Lw bars."""

    def __init__(self, dt_ns: int, dref: float, rate_ref: float, levels: int = 5):
        self.dt_ns, self.dref, self.rate_ref, self.levels = int(dt_ns), float(dref), float(rate_ref), int(levels)
        self._n_seen = 0
        self._rows: List[np.ndarray] = []
        self._t0: Optional[int] = None
        self._pending_msgs: List[tuple] = []
        self._pending_books: List[np.ndarray] = []

    def update(self, book, t: int, episodes=()) -> int:
        """Consume new messages from `book.log`; emit completed bars. Returns the number of bars emitted."""
        if self._t0 is None:
            self._t0 = int(t) - int(t) % self.dt_ns
        log = book.log
        new = log[self._n_seen:]
        self._n_seen = len(log)
        for m in new:
            self._pending_msgs.append(m)
            self._pending_books.append(book.snapshot(self.levels))
        cur_bar = (int(t) - self._t0) // self.dt_ns
        done_bar = len(self._rows)
        emitted = 0
        while done_bar < cur_bar:
            end_t = self._t0 + (done_bar + 1) * self.dt_ns
            k = 0
            while k < len(self._pending_msgs) and self._pending_msgs[k][0] < end_t:
                k += 1
            msgs = self._pending_msgs[:k]; books = self._pending_books[:k]
            del self._pending_msgs[:k]; del self._pending_books[:k]
            self._rows.append(self._bar(msgs, books, self._t0 + done_bar * self.dt_ns))
            done_bar += 1; emitted += 1
        return emitted

    def _bar(self, msgs, books, t_start) -> np.ndarray:
        if not msgs:
            row = self._rows[-1].copy() if self._rows else np.zeros(len(BAR_FEATURES), dtype=np.float32)
            row[[0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]] = 0.0
            return row
        arr = np.array(msgs, dtype=np.int64)
        m = np.zeros(len(arr), dtype=MSG_DTYPE)
        m["t"], m["type"], m["side"], m["price"], m["size"], m["order_id"], m["agent"], m["taker"] = arr.T
        bk = np.stack(books, 0)
        if self._rows and self._last_sess is not None:
            # prepend the last message of the previous bar so that returns / ofi are continuous
            m = np.concatenate([self._last_sess[0], m]); bk = np.concatenate([self._last_sess[1], bk], 0)
            sess = LOBSession("ONL", 0.01, m, bk, np.zeros(len(m), dtype=np.int8), [], {})
            bf = bar_features(sess, self.dt_ns, self.dref, self.rate_ref, t_start=t_start - self.dt_ns, t_end=t_start + self.dt_ns - 1)
            row = bf.feats[-1]
        else:
            sess = LOBSession("ONL", 0.01, m, bk, np.zeros(len(m), dtype=np.int8), [], {})
            bf = bar_features(sess, self.dt_ns, self.dref, self.rate_ref, t_start=t_start, t_end=t_start + self.dt_ns - 1)
            row = bf.feats[-1]
        self._last_sess = (m[-1:], bk[-1:])
        return row

    _last_sess = None

    def window(self, Lw: int) -> Optional[np.ndarray]:
        if len(self._rows) < Lw:
            return None
        return np.stack(self._rows[-Lw:], 0)

    @property
    def n_bars(self) -> int:
        return len(self._rows)
