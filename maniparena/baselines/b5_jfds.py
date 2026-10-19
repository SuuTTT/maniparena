"""B5 — Poutré, Chételat, Morales, "Deep unsupervised anomaly detection in high-frequency markets",
Journal of Finance and Data Science 10:100129 (2024), doi:10.1016/j.jfds.2024.100129.

ATTRIBUTION (CC-BY-4.0). `PositionalEncoding`, `TransformerCOCA`, `TransformerAutoencoder`, `jfds_features`, the Noam
schedule / early-stopping rule in `train_autoencoder` and the Nystroem + SGD one-class SVM in `fit_ocsvm` are a port of the
authors' official code, Zenodo record 11048480 ("JFDS Repo.zip", CC-BY-4.0):
    models/models.py            PositionalEncoding, TransformerCOCA, TransformerAutoencoder
    preprocessing/utilities.py  data_norm_and_features (the 14 level-1 features)
    train/utilities.py          model_train (Adam betas (0.9, 0.98), eps 1e-9, Noam warm-up 4000, sum-of-squares loss)
    scoring/utilities.py        generating_latent_vector (latent = model.encoding(model.embedding(x)))
    04_model_scoring.py         dissimilarity_score (Nystroem(gamma = 10 / (N * var)) + SGDOneClassSVM, score = -score_samples,
                                event score = mean over the windows that contain the event)
Changes with respect to the official code (all listed in conventions_b5_jfds.md):
    * device handling: the official `decoding` hard-codes `.cuda()`; the port follows the input's device (no numerical change);
    * parameter names are kept identical, so the official `best_TransformerAutoencoder.pt` loads with `strict=True`
      (`load_official_checkpoint`, a restricted unpickler: only torch / collections globals and the ported classes are allowed);
    * the official return features are log(p_t / p_{t+1}) (pandas `shift(-1)`, one event of look-ahead); the default here is the
      causal log(p_{t-1} / p_t) (`causal_returns=True`); set False to reproduce the official column exactly;
    * the official z-score is per instrument-day on fraud-free rows of the whole day (train + test rows); here it is fitted on
      normal *training* rows only;
    * the official event score averages the windows that contain the event (up to 24 future events); the default here assigns
      a window's score to its last event (`aggregate="causal"`); `aggregate="overlap_mean"` reproduces the official rule;
    * `restarts` (official: 10 hyperopt evaluations of a constant space, i.e. 10 restarts) defaults to 1.

Pipeline: 14 features per level-1 event -> windows of 25 events -> Transformer autoencoder trained on normal windows
(reconstruction, teacher forcing; validation in autoregressive "generate" mode) -> latent vector (d_model = 128) -> one-class
SVM fitted on train + validation latents -> dissimilarity score -> bars by max over the events of the bar.
"""
from __future__ import annotations
import math
import pickle
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import EventLevelBaseline, register_detector
from ..detect.base import Detector
from ..data.schema import BarFrame, LOBSession

JFDS_FEATURES = ["ReturnBid1", "ReturnAsk1", "DerivativeReturnBid1", "DerivativeReturnAsk1", "BidSize1", "AskSize1", "TradeBidSize", "TradeAskSize",
                 "CancelledBidSize", "CancelledAskSize", "TradeBidIndicator", "TradeAskIndicator", "CancelledBidIndicator", "CancelledAskIndicator"]
OFFICIAL = dict(d_model=128, dim_ff=512, num_layers=6, seq_length=25, batch_size=512, epochs=500, warmup_steps=4000, patience=100, restarts=10)


# ====================================================================================================================
# models/models.py (port; attribute names identical to the official module so that state dicts are interchangeable)
# ====================================================================================================================
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)


class TransformerCOCA(torch.nn.Module):
    """Official base class (marked deprecated upstream; kept because TransformerAutoencoder inherits its layers). Only the
    `pretraining=True` decoding branch — the one TransformerAutoencoder uses — is ported; `nhead` is 8 upstream."""

    def __init__(self, nb_features, seq_length, d_model, dim_ff, num_layers, latent_dim, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.nb_features = nb_features
        self.latent_dim = latent_dim

        self.pe = PositionalEncoding(d_model=d_model, max_len=seq_length, dropout=0.1)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_ff, dropout=0.1)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_ff, dropout=0.1)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

        self.linear_embedding = torch.nn.Linear(in_features=nb_features, out_features=d_model)
        self.linear_output = torch.nn.Linear(in_features=d_model, out_features=nb_features)
        self.decoder_projection = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.linear_encoding = torch.nn.Linear(in_features=d_model * seq_length, out_features=d_model)

        self.projection_head = torch.nn.Linear(in_features=d_model * seq_length, out_features=latent_dim, bias=False)
        self.projection_dropout = torch.nn.Dropout(p=0.1)

    def embedding(self, x):
        x = self.linear_embedding(x)
        x = self.pe(x)
        return x

    def encoding(self, x):
        x = self.encoder(x)
        x = self.linear_encoding(x.reshape((-1, self.seq_length * self.d_model)))
        return x

    def decoding(self, x, context, mode='train', pretraining=True):
        if not pretraining:
            raise NotImplementedError("only the pretraining (autoencoder) branch of the official decoder is ported")
        batch = x.shape[0]
        seq_length = x.shape[1]
        device = x.device
        memory = context.view((batch, 1, self.d_model))
        if mode == "train":
            mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).to(device)
            target = torch.cat((torch.zeros_like(x[:, 0:1, :]), x[:, :-1, :]), dim=1)
            target = self.embedding(target)
            output = self.decoder(tgt=target, memory=memory, tgt_mask=mask)
            output = self.linear_output(output)
        elif mode == "generate":
            decoder_output = torch.zeros((batch, seq_length + 1, self.nb_features), device=device)
            for t in range(1, seq_length + 1):
                target = self.embedding(decoder_output[:, :t, :])
                mask = torch.nn.Transformer.generate_square_subsequent_mask(target.shape[1]).to(device)
                decoder_output_t = self.decoder(tgt=target, memory=memory, tgt_mask=mask)
                decoder_output[:, t, :] = self.linear_output(decoder_output_t[:, -1, :])
            output = decoder_output[:, 1:, :]
        else:
            raise ValueError(mode)
        return output


class TransformerAutoencoder(TransformerCOCA):
    def __init__(self, nb_features, seq_length, d_model, dim_ff, num_layers, nhead: int = 8):
        super().__init__(nb_features, seq_length, d_model, dim_ff, num_layers, 32, nhead=nhead)

    def forward(self, x, mode):
        emb = self.embedding(x)
        context = self.encoding(emb)
        seq = self.decoding(x, context, mode=mode, pretraining=True)
        return seq

    def latent(self, x):
        """scoring/utilities.py generating_latent_vector: model.encoding(model.embedding(x))."""
        return self.encoding(self.embedding(x))


class _RestrictedUnpickler(pickle.Unpickler):
    """Whole-model checkpoints (`torch.save(model, ...)`) need unpickling; only torch / collections globals are resolved and
    the official `models.models` classes are mapped to the ported classes above. Everything else is refused."""
    _ALLOWED_ROOTS = ("torch", "collections")
    _ALLOWED_BUILTINS = ("set", "frozenset", "list", "dict", "tuple")      # data containers only (never getattr / eval / exec)

    def find_class(self, module, name):
        if module == "models.models":
            here = sys.modules[__name__]
            if name in ("PositionalEncoding", "TransformerCOCA", "TransformerAutoencoder"):
                return getattr(here, name)
            raise pickle.UnpicklingError(f"official class {name} is not ported")
        if module.split(".")[0] in self._ALLOWED_ROOTS:
            return super().find_class(module, name)
        if module in ("__builtin__", "builtins") and name in self._ALLOWED_BUILTINS:
            return {"set": set, "frozenset": frozenset, "list": list, "dict": dict, "tuple": tuple}[name]
        raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a JFDS checkpoint")


def load_official_checkpoint(path: str, nb_features: int = 14, seq_length: int = 25, d_model: int = 128, dim_ff: int = 512, num_layers: int = 6) -> TransformerAutoencoder:
    """Load `models/best_TransformerAutoencoder.pt` of the Zenodo archive into the ported class (strict state-dict match)."""
    shim = types.ModuleType("jfds_restricted_pickle")
    shim.Unpickler = _RestrictedUnpickler
    shim.load = pickle.load
    shim.__name__ = "pickle"
    obj = torch.load(path, map_location="cpu", pickle_module=shim, weights_only=False)
    state = obj if isinstance(obj, dict) else obj.state_dict()
    model = TransformerAutoencoder(nb_features, seq_length, d_model, dim_ff, num_layers)
    model.load_state_dict(state, strict=True)
    return model.eval()


# ====================================================================================================================
# preprocessing/utilities.py data_norm_and_features (port to the LOBSession schema)
# ====================================================================================================================
def _ffill_pos(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    ok = p > 0
    if not ok.any():
        return np.ones_like(p)
    idx = np.where(ok, np.arange(len(p)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = p[idx]
    out[~ok[idx]] = p[ok][0]            # leading empties take the first valid price (zero return)
    return out


def jfds_time_diff(t_ns: np.ndarray) -> np.ndarray:
    """Official: diff of TimeInMilliSecs, set to 1 where the diff is < 1 ms, NaN (first row) or 0."""
    ms = (np.asarray(t_ns, dtype=np.int64) // 1_000_000).astype(np.float64)
    d = np.ones(len(ms))
    if len(ms) > 1:
        d[1:] = np.diff(ms)
    d[d < 1] = 1.0
    return d


def jfds_features(sess: LOBSession, causal_returns: bool = True, roll: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(T, 14) un-normalised JFDS features in the official column order and the (T,) ms time deltas.

    Mapping from the LOBSTER schema (01_data_formatting.py): TradeIndicator = direction of the executed resting order for
    type-4 messages (+1 -> trade at the bid, -1 -> trade at the ask), TradeSize = TradeIndicator * size; cancellations are
    types 2 and 3 split by direction. Rolling means use window `roll` (10 upstream; min_periods=1 here instead of dropping
    the first 9 rows)."""
    bid_px, bid_sz = _ffill_pos(sess.book[:, 0, 0]), sess.book[:, 0, 1].astype(np.float64)
    ask_px, ask_sz = _ffill_pos(sess.book[:, 0, 2]), sess.book[:, 0, 3].astype(np.float64)
    typ, side, size = sess.msgs["type"], sess.msgs["side"], sess.msgs["size"].astype(np.float64)
    T = len(typ)
    td = jfds_time_diff(sess.msgs["t"])
    trade_bid = ((typ == 4) & (side == 1)).astype(np.float64)
    trade_ask = ((typ == 4) & (side == -1)).astype(np.float64)
    trade_size_abs = np.where(typ == 4, size, 0.0)
    canc = (typ == 2) | (typ == 3)
    canc_bid, canc_ask = (canc & (side == 1)).astype(np.float64), (canc & (side == -1)).astype(np.float64)

    def ret(p):
        r = np.zeros(T)
        if T > 1:
            lp = np.log(p)
            if causal_returns:
                r[1:] = lp[:-1] - lp[1:]            # log(p_{t-1} / p_t): the official column shifted by one event
            else:
                r[:-1] = lp[:-1] - lp[1:]           # official: log(p_t / p_{t+1}) (pandas shift(-1))
        return r

    def rmean(x):
        return pd.Series(x).rolling(window=roll, min_periods=1).mean().to_numpy()

    rb, ra = ret(bid_px), ret(ask_px)
    cols = [rb, ra, rb / td, ra / td, rmean(bid_sz), rmean(ask_sz),
            rmean(trade_size_abs) * trade_bid, rmean(trade_size_abs) * trade_ask,
            rmean(canc_bid * size), rmean(canc_ask * size),
            trade_bid / td, trade_ask / td, canc_bid / td, canc_ask / td]
    X = np.stack(cols, 1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), td


class ZScore:
    """Plain z-score (official normalisation rule), fitted on the rows it is given (normal training rows)."""

    def __init__(self, clip: float = 0.0):
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.clip = float(clip)

    def fit(self, X: np.ndarray) -> "ZScore":
        X2 = X.reshape(-1, X.shape[-1]).astype(np.float64)
        self.mu = X2.mean(0)
        sd = X2.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        if self.clip > 0:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)


# ====================================================================================================================
# shared plumbing (also used by b8_lin_yang)
# ====================================================================================================================
def event_bars(frame: BarFrame, t_ns: np.ndarray) -> np.ndarray:
    return np.clip(((t_ns - frame.times[0]) // frame.meta["dt_ns"]).astype(np.int64), 0, frame.n - 1)


def sliding_windows(X: np.ndarray, L: int) -> np.ndarray:
    """(N - L + 1, L, F) causal windows; window k ends at row k + L - 1."""
    if len(X) < L:
        return np.zeros((0, L, X.shape[1]), np.float32)
    idx = np.arange(L)[None, :] + np.arange(len(X) - L + 1)[:, None]
    return X[idx].astype(np.float32)


def window_all_normal(event_label: np.ndarray, L: int) -> np.ndarray:
    """True for windows whose L events are all normal."""
    if len(event_label) < L:
        return np.zeros(0, dtype=bool)
    c = np.r_[0, np.cumsum(event_label.astype(np.int64))]
    return (c[L:] - c[:-L]) == 0


def window_scores_to_events(s: np.ndarray, n_events: int, L: int, aggregate: str) -> np.ndarray:
    """Window scores -> event scores. "causal": the window's score goes to its last event (first L-1 events NaN).
    "overlap_mean": official JFDS rule, mean over all windows that contain the event (uses up to L-1 future events)."""
    out = np.full(n_events, np.nan)
    if len(s) == 0:
        return out
    if aggregate == "causal":
        out[L - 1:L - 1 + len(s)] = s
        return out
    if aggregate != "overlap_mean":
        raise ValueError(aggregate)
    c = np.r_[0.0, np.cumsum(s)]
    i = np.arange(n_events)
    lo = np.clip(i - L + 1, 0, len(s)); hi = np.clip(i + 1, 0, len(s))
    cnt = hi - lo
    ok = cnt > 0
    out[ok] = (c[hi[ok]] - c[lo[ok]]) / cnt[ok]
    return out


def events_to_bars(frame: BarFrame, t_ns: np.ndarray, s_event: np.ndarray) -> np.ndarray:
    """CONVENTIONS.md: bar score = max over the events inside the bar; bars without a scored event get the minimum."""
    s = np.full(frame.n, -np.inf)
    ok = np.isfinite(s_event)
    if ok.any():
        np.maximum.at(s, event_bars(frame, t_ns[ok]), s_event[ok])
    fin = np.isfinite(s)
    s[~fin] = s[fin].min() if fin.any() else 0.0
    return s


def fit_ocsvm(Z: np.ndarray, gamma_rule: str = "official", n_components: int = 100, seed: int = 42):
    """04_model_scoring.py dissimilarity_score: Nystroem(gamma = 10 / (nb_dim * var)) + SGDOneClassSVM(tol=1e-4). Upstream
    `nb_dim = train_valid_set.shape[0]`, i.e. the number of latent vectors (kept as "official"); "features" uses the latent
    dimension instead (sklearn's gamma="scale" times 10)."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.pipeline import make_pipeline
    var = float(Z.var())
    var = var if var > 1e-12 else 1.0
    nb_dim = Z.shape[0] if gamma_rule == "official" else Z.shape[1]
    gamma = 10.0 / (nb_dim * var)
    transform = Nystroem(gamma=gamma, random_state=seed, n_components=int(min(n_components, len(Z))))
    clf = SGDOneClassSVM(shuffle=True, fit_intercept=True, random_state=seed, tol=1e-4)
    return make_pipeline(transform, clf).fit(Z)


def noam_lr(step: int, dim_embed: int, warmup_steps: int) -> float:
    return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def train_autoencoder(model: TransformerAutoencoder, Wtr: np.ndarray, Wva: np.ndarray, epochs: int, batch_size: int, warmup_steps: int, patience: int,
                      device, seed: int, valid_mode: str = "generate", log: Optional[List[Dict[str, float]]] = None) -> float:
    """train/utilities.py model_train, task 'seq-reconstruction-transformer': Adam(betas=(0.9, 0.98), eps=1e-9) with the Noam
    schedule, loss = sum of squared errors over (time, features) averaged over the batch, validation in generate mode, stop
    when no 5 % improvement for `patience` epochs, keep the weights with the best validation loss."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    Xtr = torch.from_numpy(Wtr); Xva = torch.from_numpy(Wva)
    best_loss, best_state, best_epoch, step = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            xb = Xtr[perm[i:i + batch_size]].to(device)
            step += 1
            for grp in opt.param_groups:
                grp["lr"] = noam_lr(step, model.d_model, warmup_steps)
            loss = (model(xb, mode="train") - xb).pow(2).sum(1).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(Xva), batch_size):
                xb = Xva[i:i + batch_size].to(device)
                vl += float((model(xb, mode=valid_mode) - xb).pow(2).sum(1).sum(1).mean()); vb += 1
            valid_loss = vl / max(vb, 1) if vb else tot / max(nb, 1)
        if log is not None:
            log.append({"epoch": epoch, "train": tot / max(nb, 1), "valid": valid_loss})
        if best_loss / max(valid_loss, 1e-30) > 1.05:
            best_epoch = epoch
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


@torch.no_grad()
def encode_latents(model: TransformerAutoencoder, W: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(W), batch_size):
        out.append(model.latent(torch.from_numpy(W[i:i + batch_size]).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.d_model), np.float32)


def split_train_valid(W: np.ndarray, valid_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Chronological 1 - valid_frac / valid_frac split of the given windows (official: 70 / 30 of shuffled normal segments)."""
    n = len(W)
    k = int(np.ceil((1.0 - valid_frac) * n))
    k = min(max(k, 1), n)
    return W[:k], (W[k:] if k < n else W[max(0, n - max(1, n // 10)):])


# ====================================================================================================================
@register_detector("b5_jfds")
class JFDSDetector(EventLevelBaseline):
    paper = "Poutré, Chételat, Morales — Deep unsupervised anomaly detection in high-frequency markets, JFDS 10:100129 (2024), doi:10.1016/j.jfds.2024.100129"
    year = 2024
    status = "reimplemented"     # port of the official CC-BY code (Zenodo 11048480); "official" when the official checkpoint is loaded
    supervised = False
    event_window = 25
    event_levels = 1
    conventions = {
        "input_unit": "level-1 event (LOBSTER message + best bid / ask after it); window of 25 consecutive events, 14 features",
        "label_unit": "event (injected order); here bar label = episode overlap, event windows mapped to bars by max",
        "normalisation": "z-score per feature fitted on normal training rows (official: per instrument-day on fraud-free rows of the whole day)",
        "threshold_rule": "paper: threshold tau maximising F4 on the test set (reported next to AUROC / AUPRC); never used here: scores only",
        "official_code": "Zenodo 11048480 (CC-BY-4.0): models/models.py, preprocessing/utilities.py, train/utilities.py, scoring/utilities.py, 04_model_scoring.py",
    }
    unspecified = [
        "official code fixes everything; items below are adaptations to the ManipArena contract rather than open choices",
        "return features: official log(p_t / p_{t+1}) looks one event ahead -> causal log(p_{t-1} / p_t) by default (causal_returns)",
        "z-score sample: official per instrument-day incl. test rows -> normal training rows of all training sessions pooled",
        "event score aggregation: official mean over overlapping windows (24 future events) -> window score at its last event (aggregate)",
        "rolling means: official drops the first 9 rows (NaN) -> min_periods=1",
        "train / validation split: official 70 / 30 of shuffled fraud-free segments -> chronological 70 / 30 of normal windows unless val frames are given",
        "hidden executions (LOBSTER type 5) are dropped upstream; the shared loader maps them to type 4, so they count as trades here",
        "restarts: official 10 hyperopt evaluations of a constant space (10 restarts, 500 epochs each) -> restarts=1 by default",
        "OC-SVM gamma: upstream 10 / (N * var) with N = number of latent vectors (variable named nb_dim); kept as gamma_rule='official'",
        "bar-window fallback (no session attached): same autoencoder + OC-SVM on LOB-lite bar windows (not in the paper)",
    ]

    def __init__(self, cfg: Dict[str, Any], device: str = "cpu"):
        super().__init__(cfg, device)
        c = self.cfg
        self.event_window = int(c.get("seq_length", OFFICIAL["seq_length"]))
        self.d_model = int(c.get("d_model", OFFICIAL["d_model"]))
        self.dim_ff = int(c.get("dim_ff", OFFICIAL["dim_ff"]))
        self.num_layers = int(c.get("num_layers", OFFICIAL["num_layers"]))
        self.nhead = int(c.get("nhead", 8))
        self.epochs = int(c.get("epochs", OFFICIAL["epochs"]))
        self.batch_size = int(c.get("batch_size", OFFICIAL["batch_size"]))
        self.warmup_steps = int(c.get("warmup_steps", OFFICIAL["warmup_steps"]))
        self.patience = int(c.get("patience", OFFICIAL["patience"]))
        self.restarts = int(c.get("restarts", 1))
        self.valid_frac = float(c.get("valid_frac", 0.3))
        self.valid_mode = str(c.get("valid_mode", "generate"))
        self.aggregate = str(c.get("aggregate", "causal"))
        self.causal_returns = bool(c.get("causal_returns", True))
        self.gamma_rule = str(c.get("gamma_rule", "official"))
        self.max_train = int(c.get("max_train", 0))
        self.clip = float(c.get("clip", 0.0))
        self.model: Optional[TransformerAutoencoder] = None
        self.oc = None
        self.z: Optional[ZScore] = None
        self._event_mode = False

    # ------------------------------------------------------------------ features
    def _events(self, frame: BarFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sess: Optional[LOBSession] = frame.meta.get("_session")
        if sess is None:
            return None
        X, _ = jfds_features(sess, self.causal_returns)
        return X, sess.msgs["t"]

    def _normal_windows(self, frames: Sequence[BarFrame], ev) -> Tuple[np.ndarray, List[np.ndarray]]:
        L = self.event_window
        Ws = []
        for (X, t), f in zip(ev, frames):
            lab = f.labels[event_bars(f, t)]
            W = sliding_windows(self.z(X), L)
            Ws.append(W[window_all_normal(lab, L)])
        return (np.concatenate(Ws) if Ws else np.zeros((0, L, len(JFDS_FEATURES)), np.float32)), Ws

    # ------------------------------------------------------------------ core (windows in, model + OC-SVM out)
    def _fit_windows(self, Wtr: np.ndarray, Wva: np.ndarray) -> None:
        if len(Wtr) == 0:
            raise ValueError("b5_jfds: no normal training windows")
        rng = np.random.default_rng(self.seed)
        if self.max_train and len(Wtr) > self.max_train:
            Wtr = Wtr[np.sort(rng.choice(len(Wtr), self.max_train, replace=False))]
        if self.max_train and len(Wva) > self.max_train:
            Wva = Wva[np.sort(rng.choice(len(Wva), self.max_train, replace=False))]
        seq, nf = Wtr.shape[1], Wtr.shape[2]
        ckpt = self.cfg.get("official_checkpoint")
        if ckpt:
            self.model = load_official_checkpoint(ckpt, nf, seq, self.d_model, self.dim_ff, self.num_layers).to(self.device)
            self.status = "official"
            self.diag["checkpoint"] = str(ckpt)
        else:
            best = (float("inf"), None)
            for r in range(max(self.restarts, 1)):
                torch.manual_seed(self.seed + r)
                m = TransformerAutoencoder(nf, seq, self.d_model, self.dim_ff, self.num_layers, nhead=self.nhead).to(self.device)
                log: List[Dict[str, float]] = []
                loss = train_autoencoder(m, Wtr, Wva, self.epochs, self.batch_size, self.warmup_steps, self.patience, self.device, self.seed + r,
                                         self.valid_mode, log)
                self.diag.setdefault("restarts", []).append({"valid_loss": loss, "epochs_run": len(log)})
                if loss < best[0] or best[1] is None:
                    best = (loss, m)
            self.model = best[1]
            self.diag["valid_loss"] = best[0]
        Z = np.concatenate([encode_latents(self.model, Wtr, self.device), encode_latents(self.model, Wva, self.device)])
        self.oc = fit_ocsvm(Z.astype(np.float64), self.gamma_rule, seed=42)
        self.diag["n_train_windows"], self.diag["n_valid_windows"] = int(len(Wtr)), int(len(Wva))

    def _score_windows(self, W: np.ndarray) -> np.ndarray:
        if len(W) == 0:
            return np.zeros(0)
        Z = encode_latents(self.model, W, self.device)
        return -self.oc.score_samples(Z.astype(np.float64))

    # ------------------------------------------------------------------ contract
    def fit(self, frames: Sequence[BarFrame], val: Optional[Sequence[BarFrame]] = None) -> "Detector":
        ev = [self._events(f) for f in frames]
        if any(e is None for e in ev):
            self.diag["fallback"] = "no session attached; bar-window fallback"
            self.status = "reimplemented-fallback"
            self._event_mode = False
            return Detector.fit(self, frames, val)
        normal_rows = [X[f.labels[event_bars(f, t)] == 0] for (X, t), f in zip(ev, frames)]
        rows = np.concatenate(normal_rows)
        self.z = ZScore(self.clip).fit(rows if len(rows) else np.concatenate([X for X, _ in ev]))
        Wn, _ = self._normal_windows(frames, ev)
        vev = [self._events(f) for f in val] if val else []
        if val and all(e is not None for e in vev):
            Wva, _ = self._normal_windows(val, vev)
            Wtr = Wn
            if len(Wva) == 0:
                Wtr, Wva = split_train_valid(Wn, self.valid_frac)
        else:
            Wtr, Wva = split_train_valid(Wn, self.valid_frac)
        self._fit_windows(Wtr, Wva)
        self._event_mode = True
        return self

    def score(self, frame: BarFrame) -> np.ndarray:
        if not self._event_mode:
            return Detector.score(self, frame)
        ev = self._events(frame)
        if ev is None:
            raise ValueError("b5_jfds was fitted on message-level sessions; frame.meta['_session'] is required for scoring")
        X, t = ev
        W = sliding_windows(self.z(X), self.event_window)
        se = window_scores_to_events(self._score_windows(W), len(X), self.event_window, self.aggregate)
        return events_to_bars(frame, t, se)

    def score_window(self, W: np.ndarray) -> float:
        if self._event_mode:
            raise RuntimeError("b5_jfds scores event windows; score_window needs the bar-window fallback (fit without sessions)")
        return Detector.score_window(self, W)

    # bar-window fallback (Detector.fit -> _fit -> _fit_events)
    def _fit_events(self, W: np.ndarray, y: np.ndarray) -> None:
        Wn = W[y == 0] if (y == 0).any() else W
        Wtr, Wva = split_train_valid(Wn, self.valid_frac)
        self._fit_windows(Wtr, Wva)

    def _score_events(self, W: np.ndarray) -> np.ndarray:
        return self._score_windows(W)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters())) if self.model is not None else 0
