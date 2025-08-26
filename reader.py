# quant_dashboard.py
# Stock-style order book (Ask/Bid) + Kelly + snapshot/deltas (quotes) + REAL trades feed/counters.
import os, json, time, errno
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import altair as alt
import streamlit as st

# ---------------- Utils ----------------
def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

def to_iso(ts: Any) -> Optional[str]:
    if isinstance(ts, (int, float)):
        try: return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception: return None
    if isinstance(ts, str): return ts
    return None

def safe_int(x) -> Optional[int]:
    try: return int(x)
    except Exception: return None

def extract_msg_fields(obj: dict):
    """Return (type, payload, sid, seq). Payload can be under 'msg' or 'data'."""
    mtype = obj.get("type")
    payload = obj.get("msg") or obj.get("data") or {}
    return mtype, payload, obj.get("sid"), obj.get("seq")

# -------------- Order Book --------------
class OrderBook:
    """
    YES bids stored as-is; NO bids stored as-is.
    Display: ASKS are flipped from NO: ask_price = 100 - no_price (sizes preserved/aggregated).
    Tracks REAL trade flow (not deltas) and keeps a live trade tape.
    """
    def __init__(self):
        self.yes = defaultdict(int)   # YES ladder: price -> size (bid)
        self.no  = defaultdict(int)   # NO ladder: price -> size (bid in NO-space)
        self.market_id = None
        self.market_ticker = None
        self.last_update_iso = None
        self.latest_ticker = {}
        self.last_seq_by_sid: Dict[int, int] = {}

        # Executed volume counters (contracts)
        self.flow = {"yes_sold": 0, "yes_bought": 0, "no_sold": 0, "no_bought": 0}

        # Live trades buffer (most recent first)
        self.live_trades: deque = deque(maxlen=4000)

    # ----- Sequence guard for deltas/snapshots -----
    def _seq_ok(self, sid: Optional[int], seq: Optional[int]) -> bool:
        if sid is None or seq is None:
            return True
        prev = self.last_seq_by_sid.get(sid, -1)
        if seq <= prev: return False
        self.last_seq_by_sid[sid] = seq
        return True

    # ----- Snapshot / Delta (quotes only) -----
    def apply_snapshot(self, payload: dict, sid: Optional[int], seq: Optional[int]):
        self.yes.clear(); self.no.clear()
        for pair in payload.get("yes") or []:
            if isinstance(pair, list) and len(pair)==2:
                p, sz = safe_int(pair[0]), safe_int(pair[1])
                if p is not None and sz and sz>0: self.yes[p] += sz
        for pair in payload.get("no") or []:
            if isinstance(pair, list) and len(pair)==2:
                p, sz = safe_int(pair[0]), safe_int(pair[1])
                if p is not None and sz and sz>0: self.no[p] += sz

        self.market_id = payload.get("market_id", self.market_id)
        self.market_ticker = payload.get("market_ticker", self.market_ticker)
        iso = to_iso(payload.get("ts"))
        if iso: self.last_update_iso = iso
        if sid is not None and seq is not None: self.last_seq_by_sid[sid] = seq

    def apply_delta(self, payload: dict, sid: Optional[int], seq: Optional[int]):
        # Quotes being placed/changed/canceled (NOT executions)
        if not self._seq_ok(sid, seq): return
        side = payload.get("side")
        price = safe_int(payload.get("price"))
        delta = safe_int(payload.get("delta"))
        if side not in ("yes","no") or price is None or delta is None: return

        ladder = self.yes if side == "yes" else self.no
        ladder[price] += delta
        if ladder[price] <= 0: ladder.pop(price, None)

        self.market_id = payload.get("market_id", self.market_id)
        self.market_ticker = payload.get("market_ticker", self.market_ticker)
        iso = to_iso(payload.get("ts"))
        if iso: self.last_update_iso = iso

    # ----- Ticker -----
    def apply_ticker(self, payload: dict):
        self.market_id = payload.get("market_id", self.market_id)
        self.market_ticker = payload.get("market_ticker", self.market_ticker)
        self.latest_ticker = payload
        iso = to_iso(payload.get("ts"))
        if iso: self.last_update_iso = iso

    # ----- Trades (executions) -----
    def apply_trade(self, payload: dict, sid: Optional[int], seq: Optional[int]):
        """
        payload example:
        {
          'trade_id': '...', 'market_ticker': '...', 'yes_price': 36, 'no_price': 64,
          'count': 1, 'taker_side': 'yes', 'ts': 1755988579
        }
        Interpretation:
          taker_side == 'yes' -> BUY YES @ yes_price  (equivalently SELL NO @ no_price)
          taker_side == 'no'  -> BUY NO  @ no_price   (equivalently SELL YES @ yes_price)
        Counters updated on executed size only (count).
        """
        trade_id   = payload.get("trade_id")
        yes_price  = safe_int(payload.get("yes_price"))
        no_price   = safe_int(payload.get("no_price"))
        qty        = safe_int(payload.get("count")) or 1
        taker_side = payload.get("taker_side")
        iso_ts     = to_iso(payload.get("ts")) or now_iso_utc()

        if taker_side == "yes":
            # Executed: BUY YES; SELL NO
            self.flow["yes_bought"] += qty
            self.flow["no_sold"]    += qty
            price_yes = yes_price
        elif taker_side == "no":
            # Executed: BUY NO; SELL YES
            self.flow["no_bought"]  += qty
            self.flow["yes_sold"]   += qty
            price_yes = yes_price
        else:
            return  # unknown taker side

        self.live_trades.appendleft({
            "ts": iso_ts,
            "trade_id": trade_id,
            "taker_side": taker_side,
            "price_yes": price_yes,
            "no_price": no_price,
            "qty": qty
        })

        self.market_id = payload.get("market_id", self.market_id)
        self.market_ticker = payload.get("market_ticker", self.market_ticker)

    # --- Bests ---
    def best_yes_bid(self) -> Optional[int]:
        return max(self.yes) if self.yes else None
    def best_no_bid(self) -> Optional[int]:
        return max(self.no) if self.no else None

    # --- Kelly: x = p2 / (100 + p2 - p1) ---
    def kelly_estimate(self) -> Optional[float]:
        t = self.latest_ticker or {}
        t_bid = safe_int(t.get("yes_bid"))
        t_ask = safe_int(t.get("yes_ask"))
        if t_bid is not None and t_ask is not None:
            p2, p1 = t_bid, t_ask
        else:
            by = self.best_yes_bid()
            bn = self.best_no_bid()
            if by is None and bn is None: return None
            p2 = by if by is not None else 0
            p1 = (100 - bn) if bn is not None else None
            if p1 is None: return None
        denom = 100 + p2 - p1
        if denom <= 0: return None
        return p2 / denom

    # --- Tables ---
    def orderbook_tables(self, depth: int = 25):
        """
        Returns:
          df_yes (YES bids high->low),
          df_no  (NO bids high->low),
          df_stock (AskSize, AskPrice, BidPrice, BidSize) with asks from flipped NO.
        """
        yes_list = sorted(self.yes.items(), key=lambda x: x[0], reverse=True)[:depth]
        df_yes = pd.DataFrame(yes_list, columns=["price","size"]) if yes_list else pd.DataFrame(columns=["price","size"])

        no_list  = sorted(self.no.items(), key=lambda x: x[0], reverse=True)[:depth]
        df_no = pd.DataFrame(no_list, columns=["price","size"]) if no_list else pd.DataFrame(columns=["price","size"])

        # Asks from flipped NO
        if not df_no.empty:
            asks = df_no.copy()
            asks["AskPrice"] = 100 - asks["price"]
            asks = asks.groupby("AskPrice", as_index=False)["size"].sum()
            asks = asks.sort_values("AskPrice", ascending=True).reset_index(drop=True)
        else:
            asks = pd.DataFrame(columns=["AskPrice","size"])

        bids = df_yes.sort_values("price", ascending=False).reset_index(drop=True)
        n_rows = max(len(asks), len(bids))
        asks = asks.reindex(range(n_rows))
        bids = bids.reindex(range(n_rows))

        df_stock = pd.DataFrame({
            "AskSize": asks["size"],
            "AskPrice": asks["AskPrice"],
            "BidPrice": bids["price"],
            "BidSize": bids["size"]
        })
        return df_yes, df_no, df_stock

# ------------- History / Time Series -------------
class MarketHistory:
    def __init__(self, max_rows: int = 10000):
        self.max_rows = max_rows
        self.ticks = pd.DataFrame(columns=["ts_iso","last","yes_bid","yes_ask","spread","kelly"])
        self.order_flow = pd.DataFrame(columns=[
            "ts_iso","imbalance","top_yes_price","top_no_price",
            "top_yes_size","top_no_size","depth_yes","depth_no",
            "yes_sold","yes_bought","no_sold","no_bought"
        ])

    def add_ticker(self, payload: dict, ob: OrderBook):
        ts_iso = to_iso(payload.get("ts")) or now_iso_utc()
        last, yes_bid, yes_ask = payload.get("price"), payload.get("yes_bid"), payload.get("yes_ask")
        spread = None
        try:
            if yes_bid is not None and yes_ask is not None:
                spread = float(yes_ask) - float(yes_bid)
        except Exception: pass
        kelly = ob.kelly_estimate()

        self.ticks = pd.concat([self.ticks, pd.DataFrame([{
            "ts_iso": ts_iso, "last": last, "yes_bid": yes_bid, "yes_ask": yes_ask,
            "spread": spread, "kelly": kelly
        }])], ignore_index=True)
        if len(self.ticks) > self.max_rows:
            self.ticks = self.ticks.iloc[-self.max_rows:].reset_index(drop=True)

        # Order-flow snapshot from book (not executions)
        top_yes = ob.best_yes_bid(); top_no = ob.best_no_bid()
        yes_sz = ob.yes.get(top_yes, 0) if top_yes is not None else 0
        no_sz  = ob.no.get(top_no, 0)  if top_no is not None else 0
        depth_yes = sum(ob.yes.values()); depth_no = sum(ob.no.values())
        imb = ((yes_sz - no_sz) / (yes_sz + no_sz)) if (yes_sz + no_sz) else None

        of_row = {
            "ts_iso": ts_iso, "imbalance": imb, "top_yes_price": top_yes, "top_no_price": top_no,
            "top_yes_size": yes_sz, "top_no_size": no_sz, "depth_yes": depth_yes, "depth_no": depth_no,
            "yes_sold": ob.flow["yes_sold"], "yes_bought": ob.flow["yes_bought"],
            "no_sold": ob.flow["no_sold"], "no_bought": ob.flow["no_bought"]
        }
        self.order_flow = pd.concat([self.order_flow, pd.DataFrame([of_row])], ignore_index=True)
        if len(self.order_flow) > self.max_rows:
            self.order_flow = self.order_flow.iloc[-self.max_rows:].reset_index(drop=True)

# ------------- FIFO (non-blocking) -------------
def ensure_fifo(path: str):
    if not os.path.exists(path):
        try: os.mkfifo(path, 0o666)
        except FileExistsError: pass

def fifo_open_nonblock(path: str) -> Optional[int]:
    try: return os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except Exception: return None

def fifo_read_available(fd: int, max_bytes: int) -> bytes:
    try: return os.read(fd, max_bytes)
    except BlockingIOError: return b""
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK): return b""
        return b""

def buffer_extract_lines(buf: bytearray) -> List[bytes]:
    if not buf or b"\n" not in buf: return []
    data = bytes(buf); *lines, rest = data.split(b"\n")
    buf.clear(); buf.extend(rest); return lines

def parse_json_line(line: bytes) -> Optional[dict]:
    s = line.decode("utf-8", errors="ignore").strip()
    if not s: return None
    try: return json.loads(s)
    except json.JSONDecodeError:
        try: return json.loads(s.replace("'", '"'))
        except Exception: return None

# ------------- Streamlit App -------------
st.set_page_config(page_title="Quant Trading Dashboard (Trades, Stock-style L2)", layout="wide")
st.title("Quant Trading Dashboard — Stock-Style L2 + Kelly + REAL Trades")

with st.sidebar:
    pipe_path = st.text_input("FIFO path", value="/tmp/orderbook_pipe")
    read_bytes = st.number_input("Max bytes per refresh", min_value=1024, value=131072, step=8192)
    depth = st.slider("Depth (rows)", 5, 100, 25, step=5)
    auto = st.checkbox("Auto-refresh", value=True)
    refresh_ms = st.number_input("Refresh interval (ms)", min_value=200, value=1000, step=100)
    vol_window = st.number_input("Kelly rolling vol window (ticks)", min_value=5, value=60, step=5)
    clear_state = st.button("Reset state")

# Session state
if "fifo_fd" not in st.session_state: st.session_state.fifo_fd = None
if "fifo_buf" not in st.session_state: st.session_state.fifo_buf = bytearray()
if "book" not in st.session_state:     st.session_state.book = OrderBook()
if "hist" not in st.session_state:     st.session_state.hist = MarketHistory(max_rows=20000)
if "logs" not in st.session_state:     st.session_state.logs = deque(maxlen=500)

if clear_state:
    st.session_state.fifo_fd = None; st.session_state.fifo_buf = bytearray()
    st.session_state.book = OrderBook(); st.session_state.hist = MarketHistory(max_rows=20000)
    st.session_state.logs.clear(); st.success("State reset.")

# FIFO open + ingest
ensure_fifo(pipe_path)
if st.session_state.fifo_fd is None and os.path.exists(pipe_path):
    st.session_state.fifo_fd = fifo_open_nonblock(pipe_path)

ingested = 0
if st.session_state.fifo_fd is not None:
    chunk = fifo_read_available(st.session_state.fifo_fd, int(read_bytes))
    if chunk: st.session_state.fifo_buf.extend(chunk)
    for bline in buffer_extract_lines(st.session_state.fifo_buf):
        obj = parse_json_line(bline)
        if not obj: continue
        mtype, payload, sid, seq = extract_msg_fields(obj)
        if mtype == "orderbook_snapshot":
            st.session_state.book.apply_snapshot(payload, sid, seq); ingested += 1
        elif mtype == "orderbook_delta":
            st.session_state.book.apply_delta(payload, sid, seq); ingested += 1
        elif mtype == "ticker":
            st.session_state.book.apply_ticker(payload)
            st.session_state.hist.add_ticker(payload, st.session_state.book); ingested += 1
        elif mtype == "trade":
            st.session_state.book.apply_trade(payload, sid, seq); ingested += 1

# Derived (Kelly series)
ticks = st.session_state.hist.ticks.copy()
if not ticks.empty:
    for c in ["kelly","spread","yes_bid","yes_ask"]:
        if c in ticks.columns: ticks[c] = pd.to_numeric(ticks[c], errors="coerce")
    ticks["kelly_ret"] = ticks["kelly"].pct_change()
    ticks["kelly_roll_vol"] = ticks["kelly_ret"].rolling(int(vol_window), min_periods=max(3,int(vol_window)//3)).std()

kelly_now = st.session_state.book.kelly_estimate()
last_tick = ticks.iloc[-1] if not ticks.empty else None

# Tabs
tab_overview, tab_orderbook, tab_trades, tab_tape, tab_logs = st.tabs(
    ["Overview", "Order Book (Ask/Bid)", "Live Trades", "Tape & Flow", "Logs"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelly", f"{kelly_now:.3f}" if kelly_now is not None else "—")
    p2 = int(last_tick["yes_bid"]) if (last_tick is not None and pd.notna(last_tick.get("yes_bid"))) else st.session_state.book.best_yes_bid()
    bn = st.session_state.book.best_no_bid()
    p1 = int(last_tick["yes_ask"]) if (last_tick is not None and pd.notna(last_tick.get("yes_ask"))) else ((100 - bn) if bn is not None else None)
    c2.metric("YES Bid (p2)", p2 if p2 is not None else "—")
    c3.metric("YES Ask (p1)", p1 if p1 is not None else "—")
    c4.metric("Spread (ask-bid)", (p1 - p2) if (p1 is not None and p2 is not None) else "—")
    st.caption(f"Market: {st.session_state.book.market_ticker or '—'} | ID: {st.session_state.book.market_id or '—'} | Last update: {st.session_state.book.last_update_iso or '—'} | Ingested: {ingested}")

    if not ticks.empty and ticks["kelly"].notna().any():
        st.altair_chart(
            alt.Chart(ticks).mark_line().encode(
                x=alt.X("ts_iso:T", title="Time"),
                y=alt.Y("kelly:Q", title="Kelly Estimate")
            ).properties(height=260),
            use_container_width=True
        )
        if ticks["kelly_roll_vol"].notna().any():
            st.altair_chart(
                alt.Chart(ticks).mark_line().encode(
                    x="ts_iso:T", y=alt.Y("kelly_roll_vol:Q", title=f"Kelly Rolling Vol ({int(vol_window)})")
                ).properties(height=180),
                use_container_width=True
            )
    else:
        st.info("Waiting for ticker data…")

with tab_orderbook:
    df_yes, df_no, df_stock = st.session_state.book.orderbook_tables(depth=depth)
    st.subheader("Stock-Style L2 (Asks flipped from NO, Bids from YES)")
    st.dataframe(df_stock, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("YES Bids (native, high→low)")
        st.dataframe(df_yes, use_container_width=True, hide_index=True)
    with c2:
        st.caption("NO Bids (native, high→low)")
        st.dataframe(df_no, use_container_width=True, hide_index=True)

    # Executed flow KPIs (from trade messages)
    f1, f2 = st.columns(2)
    flow = st.session_state.book.flow
    f1.metric("YES Bought", flow["yes_bought"])
    f2.metric("NO Bought", flow["no_bought"])

with tab_trades:
    st.subheader("Live Trades (from 'trade' messages)")
    n_show = st.number_input("Rows to show", 10, 2000, 200, step=10)
    trades = list(st.session_state.book.live_trades)[:int(n_show)]
    if trades:
        df_tr = pd.DataFrame(trades).sort_values("ts", ascending=False)
        st.dataframe(df_tr, use_container_width=True, hide_index=True)
        agg = df_tr.groupby("taker_side", as_index=False)["qty"].sum()
        st.altair_chart(alt.Chart(agg).mark_bar().encode(x="taker_side:N", y="qty:Q"), use_container_width=True)
    else:
        st.info("No trades yet. Waiting for 'type':'trade' messages.")

with tab_tape:
    st.subheader("Ticker & Flow Time Series")
    oflow = st.session_state.hist.order_flow.copy()
    if not oflow.empty:
        if oflow["imbalance"].notna().any():
            st.altair_chart(
                alt.Chart(oflow).mark_line().encode(
                    x="ts_iso:T", y=alt.Y("imbalance:Q", title="TOB Imbalance (-1..1)")
                ).properties(height=200), use_container_width=True
            )
        melted = oflow.melt(id_vars=["ts_iso"], value_vars=["depth_yes","depth_no"], var_name="side", value_name="depth")
        st.altair_chart(
            alt.Chart(melted).mark_line().encode(x="ts_iso:T", y="depth:Q", color="side:N").properties(height=200),
            use_container_width=True
        )
        flow_cols = ["yes_sold","yes_bought","no_sold","no_bought"]
        melted2 = oflow.melt(id_vars=["ts_iso"], value_vars=flow_cols, var_name="metric", value_name="value")
        st.altair_chart(
            alt.Chart(melted2).mark_line().encode(x="ts_iso:T", y="value:Q", color="metric:N").properties(height=220),
            use_container_width=True
        )
    else:
        st.info("No tape yet.")

with tab_logs:
    st.write(f"Ingested this refresh: {ingested}")
    st.write(f"FIFO open: {st.session_state.fifo_fd is not None}")

# Auto-refresh
if auto:
    time.sleep(float(refresh_ms) / 1000.0)
    st.rerun()
