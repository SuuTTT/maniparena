"""B1 - Tuccella, Nadler, Serban, "Protecting Retail Investors from Order Book Spoofing using a GRU-based Detection
Model", arXiv:2110.03687 (2021).

What the paper states (sections 3-4):
  * heuristic labelling rule on cancelled orders (eqs. 1-4):  V_cancelled > threshold_1 * V_cumul_25(side)  AND  distance
    of the cancelled order to the current bid-ask "under a certain threshold" (threshold_2; eq. 2 prints ">" but the text and
    section 1.1 say close to the touch, so we use "<")  AND  Delta_sigma > threshold_3 (increase of price volatility);
  * six L2 features: current bid-ask, delta volume, price of the current price-level update, price volatility, cumulative
    volume (bids), cumulative volume (asks); order book limited to 25 price levels; series of 200 book updates;
  * manipulated series = updates in [T0 - 12 s, T0 - 2 s] before the flagged cancel (2 s cropped for early detection);
    training set down-sampled to 1:1 (Table 1: 44/88, 62/124, ...);
  * model (Fig. 5): GRU -> final hidden state -> feed-forward network L1..L4 -> sigmoid; Adam, learning-rate decay, binary
    cross-entropy; number of GRU layers / learning rate / dropout / epochs tuned by random search (values not reported).

Two modes (cfg["mode"]):
  "gru"   supervised GRU on causal bar windows of the six features mapped to the LOB-lite bar schema (see MAPPING).
          cfg["labels"] = "benchmark" (default: bar labels of the benchmark) or "rule" (the paper's protocol: train on labels
          produced by the heuristic rule on the *training* frames; no ground truth is used).
  "rule"  the heuristic rule as an unsupervised score:  min(r1 / th1, (th2 + 1) / (dist + 1), dsigma / th3),
          which is > 1 exactly when conditions 1-3 hold. With a LOBSession attached (frame.meta["_session"]) the rule is
          evaluated per cancel message on the real book (all recorded levels, at most 25) and max-pooled to bars; without
          a session it falls back to bar proxies.
The code URL in the paper is "BlindedForReview" (no official code).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from ..detect.base import Detector, Normaliser, register_detector
from ..data.schema import BarFrame

# paper feature -> LOB-lite columns
MAPPING = {
    "current bid-ask": "spread (ticks; the price level itself is not stationary across instruments)",
    "delta volume": "vol_add_bid + vol_add_ask - vol_del_bid - vol_del_ask (net displayed-volume change in the bar / dref)",
    "price of the current price-level update": "add_dist (mean distance of the bar's updates from the best quote, ticks)",
    "price volatility": "rv (realised mid volatility inside the bar, ticks)",
    "cumulative volume (bids)": "depth_bid5 (top-5 levels instead of 25: the bar schema stores 5)",
    "cumulative volume (asks)": "depth_ask5",
}
_COLS = ["spread", "vol_add_bid", "vol_add_ask", "vol_del_bid", "vol_del_ask", "add_dist", "rv", "depth_bid5", "depth_ask5"]


def _six(W: np.ndarray) -> np.ndarray:
    """(..., 9) raw columns in _COLS order -> (..., 6) paper features."""
    sp, ab, aa, db, da, dist, rv, bd, ad = [W[..., i] for i in range(9)]
    return np.stack([sp, ab + aa - db - da, dist, rv, bd, ad], -1).astype(np.float32)


def rule_score_bars(W: np.ndarray, th1: float, th2: float, th3: float) -> np.ndarray:
    """Bar proxy of eqs. 1-4 on raw windows (N, L, 9): cancelled volume of the last bar relative to the same-side top-5 depth
    of the previous bar, distance proxy = add_dist of the last bar, dsigma = rv(last bar) / mean rv(earlier bars)."""
    db, da, dist, rv, bd, ad = W[..., 3], W[..., 4], W[..., 5], W[..., 6], W[..., 7], W[..., 8]
    prev = -2 if W.shape[1] > 1 else -1
    r1 = np.maximum(db[:, -1] / np.maximum(bd[:, prev], 1e-6), da[:, -1] / np.maximum(ad[:, prev], 1e-6))
    base = rv[:, :-1].mean(1) if W.shape[1] > 1 else rv[:, -1]
    dsig = rv[:, -1] / np.maximum(base, 1e-6)
    d = np.maximum(dist[:, -1], 0.0)
    return np.minimum(np.minimum(r1 / th1, (th2 + 1.0) / (d + 1.0)), dsig / th3)


def rule_score_events(sess, th1: float, th2: float, th3: float, short: int, long: int, levels: int = 25) -> np.ndarray:
    """Per-message rule score (0 for non-cancel messages). Book state and volatility are taken *before* the cancel."""
    msgs, book = sess.msgs, sess.book
    T = len(msgs)
    out = np.zeros(T)
    if T < 3:
        return out
    Lv = min(levels, book.shape[1])
    prev = np.r_[0, np.arange(T - 1)]
    cum_b, cum_a = book[prev, :Lv, 1].sum(1), book[prev, :Lv, 3].sum(1)
    best_b, best_a = book[prev, 0, 0], book[prev, 0, 2]
    side, size, price = msgs["side"], msgs["size"].astype(np.float64), msgs["price"].astype(np.float64)
    is_c = (msgs["type"] == 2) | (msgs["type"] == 3)
    cum = np.where(side == 1, cum_b, cum_a)
    best = np.where(side == 1, best_b, best_a)
    dist = np.where(best > 0, np.abs(price - best), np.inf)
    mid = sess.mid()
    idx = np.where(np.isfinite(mid), np.arange(T), 0); np.maximum.accumulate(idx, out=idx)
    dm = np.r_[0.0, np.diff(np.nan_to_num(mid[idx]))]
    c2 = np.r_[0.0, np.cumsum(dm * dm)]

    def trailing(n):                      # rms mid change over the last n updates ending at i (inclusive)
        i = np.arange(T) + 1
        lo = np.maximum(i - n, 0)
        return np.sqrt((c2[i] - c2[lo]) / np.maximum(i - lo, 1))
    dsig = trailing(short) / np.maximum(trailing(long), 1e-9)
    s = np.minimum(np.minimum(size / np.maximum(th1 * cum, 1e-9), (th2 + 1.0) / (dist + 1.0)), dsig / th3)
    out[is_c] = np.nan_to_num(s[is_c], nan=0.0, posinf=0.0)
    return out


class _SixNorm:
    """Normaliser over the six derived features (fitted on training windows only); identity scaling in rule mode."""

    def __init__(self, scale: bool):
        self.scale = scale
        self.n = Normaliser()

    def fit(self, W):
        if self.scale:
            self.n.fit(_six(W))
        return self

    def __call__(self, W):
        return self.n(_six(W)) if self.scale else np.asarray(W, dtype=np.float32)

    def state(self):
        return self.n.state() if self.scale else {}


class _GRUNet(nn.Module):
    def __init__(self, F: int, hidden: int, layers: int, ff: Sequence[int], dropout: float):
        super().__init__()
        self.gru = nn.GRU(F, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        mods: List[nn.Module] = []
        d = hidden
        for w in ff:                                    # L1..L3
            mods += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(dropout)]
            d = w
        mods += [nn.Linear(d, 1)]                       # L4 (the sigmoid is folded into the loss)
        self.ff = nn.Sequential(*mods)

    def forward(self, x):
        _, h = self.gru(x)
        return self.ff(h[-1]).squeeze(-1)


@register_detector("b1_tuccella_gru")
class TuccellaGRU(Detector):
    paper = "Tuccella, Nadler, Serban - Protecting Retail Investors from Order Book Spoofing using a GRU-based Detection Model, arXiv:2110.03687"
    year = 2021
    status = "reimplemented"
    supervised = True
    conventions = {
        "input_unit": "paper: series of 200 L2 book updates (25 levels, 6 signals); here: causal windows of bars with the 6 signals mapped to LOB-lite columns",
        "label_unit": "paper: series ending 2 s before a rule-flagged cancel; here: bar label at the window end (or rule labels, cfg labels='rule')",
        "normalisation": "not stated in the paper; here: per-feature median/IQR fitted on training windows only",
        "threshold_rule": "paper: sigmoid output, accuracy at an unstated cut (0.5 implied); here: continuous score, no threshold",
        "official_code": "none (URL blinded in the paper)",
    }
    unspecified = [
        "threshold_1, threshold_2, threshold_3 of the labelling rule are not reported -> th1=0.1, th2=5 ticks, th3=1.5 (th3 read off the dashed line of Fig. 2)",
        "definition of the volatility signal and of Delta_sigma -> rms mid change; Delta_sigma = short-window / long-window ratio (events: 200 / 2000 updates; bars: last bar / mean of earlier bars in the window)",
        "eq. 2 prints 'distance > threshold_2' while the text requires the order to be close to the touch -> we use distance < threshold_2",
        "GRU hidden size and number of layers (random search, not reported) -> hidden 64, 1 layer",
        "widths of the 4 feed-forward layers -> 64, 32, 16, 1; dropout 0.1",
        "learning rate, decay schedule, epochs, batch size -> Adam 1e-3, exponential decay 0.95 per epoch, 30 epochs, batch 128",
        "down-sampling scheme (paper only shows 1:1 class counts) -> random under-sampling of negatives to 1:1 each epoch",
        "feature normalisation -> median/IQR on training windows",
        "mapping of the 6 L2 signals to bars (see MAPPING); 5 levels of cumulative volume instead of 25 at bar level",
        "the 2-second early-detection crop is not applied: the benchmark scores bar t from the causal window ending at t",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        self.mode = str(cfg.get("mode", "gru"))
        if self.mode not in ("gru", "rule"):
            raise ValueError("b1_tuccella_gru: mode must be 'gru' or 'rule'")
        self.supervised = self.mode == "gru" and str(cfg.get("labels", "benchmark")) == "benchmark"
        self.th = (float(cfg.get("th1", 0.1)), float(cfg.get("th2", 5.0)), float(cfg.get("th3", 1.5)))
        self.norm = _SixNorm(scale=self.mode == "gru")

    def _select_features(self, frames: Sequence[BarFrame]):
        names = list(frames[0].feat_names)
        missing = [c for c in _COLS if c not in names]
        if missing:
            raise ValueError(f"b1_tuccella_gru needs LOB-lite columns {missing}")
        self.feat_idx = np.array([names.index(c) for c in _COLS])
        masked = sorted({c for f in frames if f.feat_mask is not None for c in _COLS if not f.feat_mask[names.index(c)]})
        if masked:
            self.diag["masked_inputs"] = masked       # book columns unavailable (trade-only frames): filled with 0

    # ------------------------------------------------------------------ fit
    def fit(self, frames, val=None):
        if self.mode == "gru" and str(self.cfg.get("labels", "benchmark")) == "rule":
            # the paper's protocol: labels come from the heuristic rule on the training frames; no ground truth is touched
            self._select_features(frames)
            W, _, e, _ = self.windows(frames)
            ry = (rule_score_bars(W, *self.th) > 1.0).astype(np.int8)
            self.diag["rule_label_rate"] = float(ry.mean()) if len(W) else 0.0
            self.norm.fit(W)
            self._fit(self.norm(W), ry, e, val)
            return self
        return super().fit(frames, val)

    def _fit(self, W, y, e, val):
        if self.mode == "rule":
            return
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        c = self.cfg
        self.net = _GRUNet(W.shape[-1], int(c.get("hidden", 64)), int(c.get("layers", 1)), tuple(c.get("ff", (64, 32, 16))),
                           float(c.get("dropout", 0.1))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-3)))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(c.get("lr_decay", 0.95)))
        lossf = nn.BCEWithLogitsLoss()
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            self.diag["degenerate_labels"] = True
        X = torch.tensor(W); Y = torch.tensor(y.astype(np.float32))
        bs = int(c.get("batch_size", 128))
        for ep in range(int(c.get("epochs", 30))):
            if len(pos) and len(neg) and c.get("downsample", True):
                idx = np.concatenate([pos, rng.choice(neg, min(len(neg), len(pos)), replace=False)])
            else:
                idx = np.arange(len(y))
            rng.shuffle(idx)
            self.net.train(); tot = 0.0
            for i in range(0, len(idx), bs):
                b = torch.as_tensor(idx[i:i + bs])
                opt.zero_grad()
                loss = lossf(self.net(X[b].to(self.device)), Y[b].to(self.device))
                loss.backward(); opt.step(); tot += loss.item() * len(b)
            sched.step()
            self.diag.setdefault("loss", []).append(tot / max(len(idx), 1))

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def _score(self, W):
        if self.mode == "rule":
            return rule_score_bars(W, *self.th)
        self.net.eval()
        out = [self.net(torch.tensor(W[i:i + 4096]).to(self.device)).cpu().numpy() for i in range(0, len(W), 4096)]
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, frame: BarFrame) -> np.ndarray:
        sess = frame.meta.get("_session") if self.mode == "rule" else None
        if sess is None or not self.cfg.get("use_session", True):
            return super().score(frame)
        se = rule_score_events(sess, *self.th, short=int(self.cfg.get("vol_short", 200)), long=int(self.cfg.get("vol_long", 2000)),
                               levels=int(self.cfg.get("levels", 25)))
        bar = np.clip(((sess.msgs["t"] - frame.times[0]) // frame.meta["dt_ns"]).astype(np.int64), 0, frame.n - 1)
        s = np.zeros(frame.n)
        np.maximum.at(s, bar, se)
        self.diag["rule_unit"] = "event"
        return s

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) if hasattr(self, "net") else 3
