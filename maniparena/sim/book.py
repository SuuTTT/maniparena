"""Price-time-priority limit order book with agent attribution.

Prices are integer ticks. Every message (add / cancel / delete / execute) is appended to `log` as a tuple
    (t_ns, mtype, side, price_tick, size, order_id, agent_id, taker_id)
with mtype in {1: new limit, 2: partial cancel, 3: delete, 4: execution (visible)} — the LOBSTER event codes — so
real LOBSTER / ITCH streams and simulated streams share one message schema (maniparena.data.schema.LOBSession).
`agent_id` is the owner of the resting order the message refers to and `taker_id` the aggressor of an execution
(-1 otherwise); together they give the ground-truth attribution used to label manipulation (real data: -1 = unknown).
"""
from __future__ import annotations
from bisect import bisect_left, insort
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

BID, ASK = 1, -1
ADD, CANCEL, DELETE, EXEC = 1, 2, 3, 4


class Order:
    __slots__ = ("id", "agent", "side", "price", "qty", "t")

    def __init__(self, oid: int, agent: int, side: int, price: int, qty: int, t: int):
        self.id, self.agent, self.side, self.price, self.qty, self.t = oid, agent, side, price, qty, t


class Fill:
    __slots__ = ("t", "price", "qty", "maker_id", "maker_agent", "taker_agent", "taker_side")

    def __init__(self, t, price, qty, maker_id, maker_agent, taker_agent, taker_side):
        self.t, self.price, self.qty = t, price, qty
        self.maker_id, self.maker_agent, self.taker_agent, self.taker_side = maker_id, maker_agent, taker_agent, taker_side


class OrderBook:
    def __init__(self, tick: float = 0.01, log: bool = True):
        self.tick = float(tick)
        self.bids: Dict[int, Deque[Order]] = {}
        self.asks: Dict[int, Deque[Order]] = {}
        self.bid_px: List[int] = []     # ascending; best bid = bid_px[-1]
        self.ask_px: List[int] = []     # ascending; best ask = ask_px[0]
        self.orders: Dict[int, Order] = {}
        self.bid_depth: Dict[int, int] = {}
        self.ask_depth: Dict[int, int] = {}
        self._next_id = 1
        self.keep_log = log
        self.log: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.fills: List[Fill] = []
        self.last_trade_px: Optional[int] = None
        self.n_msgs = 0

    # ------------------------------------------------------------------ queries
    def best_bid(self) -> Optional[int]:
        return self.bid_px[-1] if self.bid_px else None

    def best_ask(self) -> Optional[int]:
        return self.ask_px[0] if self.ask_px else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return 0.5 * (a + b)

    def spread(self) -> Optional[int]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else a - b

    def depth(self, side: int, price: int) -> int:
        return (self.bid_depth if side == BID else self.ask_depth).get(price, 0)

    def depth_top(self, side: int, levels: int) -> int:
        px = self.bid_px[::-1] if side == BID else self.ask_px
        d = self.bid_depth if side == BID else self.ask_depth
        return sum(d[p] for p in px[:levels])

    def snapshot(self, levels: int) -> np.ndarray:
        """(levels, 4): [bid_px, bid_sz, ask_px, ask_sz] in ticks / shares; zeros where the book is empty."""
        out = np.zeros((levels, 4), dtype=np.float64)
        for k in range(min(levels, len(self.bid_px))):
            p = self.bid_px[-1 - k]
            out[k, 0] = p; out[k, 1] = self.bid_depth[p]
        for k in range(min(levels, len(self.ask_px))):
            p = self.ask_px[k]
            out[k, 2] = p; out[k, 3] = self.ask_depth[p]
        return out

    def agent_open_qty(self, agent: int) -> int:
        return sum(o.qty for o in self.orders.values() if o.agent == agent)

    def agent_orders(self, agent: int) -> List[Order]:
        return [o for o in self.orders.values() if o.agent == agent]

    # ------------------------------------------------------------------ mutations
    def _log(self, t, mtype, side, price, qty, oid, agent, taker=-1):
        self.n_msgs += 1
        if self.keep_log:
            self.log.append((t, mtype, side, price, qty, oid, agent, taker))

    def _add_resting(self, o: Order):
        book, px = (self.bids, self.bid_px) if o.side == BID else (self.asks, self.ask_px)
        q = book.get(o.price)
        if q is None:
            book[o.price] = deque([o]); insort(px, o.price)
        else:
            q.append(o)
        dep = self.bid_depth if o.side == BID else self.ask_depth
        dep[o.price] = dep.get(o.price, 0) + o.qty
        self.orders[o.id] = o

    def _remove_level_if_empty(self, side: int, price: int):
        book, px = (self.bids, self.bid_px) if side == BID else (self.asks, self.ask_px)
        q = book.get(price)
        if q is not None and not q:
            del book[price]
            (self.bid_depth if side == BID else self.ask_depth).pop(price, None)
            i = bisect_left(px, price)
            if i < len(px) and px[i] == price:
                px.pop(i)

    def _match(self, taker_agent: int, side: int, price: Optional[int], qty: int, t: int) -> Tuple[int, List[Fill]]:
        """Walk the opposite side while the limit (None = market) is crossed. Returns (remaining qty, fills)."""
        fills: List[Fill] = []
        book, px = (self.asks, self.ask_px) if side == BID else (self.bids, self.bid_px)
        while qty > 0 and px:
            best = px[0] if side == BID else px[-1]
            if price is not None and ((side == BID and best > price) or (side == ASK and best < price)):
                break
            q = book[best]
            while qty > 0 and q:
                o = q[0]
                take = min(qty, o.qty)
                o.qty -= take; qty -= take
                dep = self.bid_depth if o.side == BID else self.ask_depth
                dep[best] -= take
                self._log(t, EXEC, o.side, best, take, o.id, o.agent, taker_agent)
                fills.append(Fill(t, best, take, o.id, o.agent, taker_agent, side))
                self.last_trade_px = best
                if o.qty == 0:
                    q.popleft(); del self.orders[o.id]
            self._remove_level_if_empty(-side, best)
        if fills:
            self.fills.extend(fills)
        return qty, fills

    def add_limit(self, agent: int, side: int, price: int, qty: int, t: int) -> Tuple[Optional[int], List[Fill]]:
        """Submit a limit order; crosses are matched immediately, the remainder rests. Returns (order_id | None, fills)."""
        oid = self._next_id; self._next_id += 1
        qty, fills = self._match(agent, side, price, qty, t)
        if qty > 0:
            o = Order(oid, agent, side, price, qty, t)
            self._add_resting(o)
            self._log(t, ADD, side, price, qty, oid, agent)
            return oid, fills
        return None, fills

    def add_market(self, agent: int, side: int, qty: int, t: int) -> List[Fill]:
        _, fills = self._match(agent, side, None, qty, t)
        return fills

    def cancel(self, oid: int, t: int, qty: Optional[int] = None) -> bool:
        o = self.orders.get(oid)
        if o is None:
            return False
        if qty is None or qty >= o.qty:
            book = self.bids if o.side == BID else self.asks
            q = book[o.price]
            try:
                q.remove(o)
            except ValueError:
                return False
            del self.orders[oid]
            dep = self.bid_depth if o.side == BID else self.ask_depth
            dep[o.price] -= o.qty
            self._log(t, DELETE, o.side, o.price, o.qty, oid, o.agent)
            self._remove_level_if_empty(o.side, o.price)
        else:
            o.qty -= qty
            dep = self.bid_depth if o.side == BID else self.ask_depth
            dep[o.price] -= qty
            self._log(t, CANCEL, o.side, o.price, qty, oid, o.agent)
        return True

    def cancel_agent(self, agent: int, t: int) -> int:
        ids = [o.id for o in self.orders.values() if o.agent == agent]
        for oid in ids:
            self.cancel(oid, t)
        return len(ids)

    def check_integrity(self) -> None:
        b, a = self.best_bid(), self.best_ask()
        assert b is None or a is None or b < a, f"crossed book {b} >= {a}"
        for p, q in self.bids.items():
            assert q and all(o.qty > 0 and o.price == p and o.side == BID for o in q)
        for p, q in self.asks.items():
            assert q and all(o.qty > 0 and o.price == p and o.side == ASK for o in q)
        assert sorted(self.bids) == self.bid_px and sorted(self.asks) == self.ask_px
        for p, q in self.bids.items():
            assert self.bid_depth[p] == sum(o.qty for o in q), (p, self.bid_depth[p])
        for p, q in self.asks.items():
            assert self.ask_depth[p] == sum(o.qty for o in q), (p, self.ask_depth[p])

    def log_array(self) -> np.ndarray:
        from ..data.schema import MSG_DTYPE as dt
        if not self.log:
            return np.zeros(0, dtype=dt)
        arr = np.array(self.log, dtype=np.int64)
        out = np.zeros(len(arr), dtype=dt)
        out["t"], out["type"], out["side"], out["price"], out["size"], out["order_id"], out["agent"], out["taker"] = arr.T
        return out
