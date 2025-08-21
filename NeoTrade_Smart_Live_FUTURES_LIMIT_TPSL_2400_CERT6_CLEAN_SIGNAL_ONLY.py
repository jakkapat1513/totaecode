
import os, sys, math, json, time, asyncio, logging, random
import time as pytime
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

CONFIG: Dict = {
    "dry_run": False,                 # <<< เปลี่ยนเป็น False เพื่อ Live
    "exchange": "binance",
    "market_type": "futures",        # "spot" หรือ "futures"
    "symbol": "ETH/USDT",
    "leverage": 5,
    "initial_balance": 1000.0,
    "risk_per_trade": 0.02,
    "fee_rate": 0.0004,
    "slippage": 0.0005,
    "bar_interval_sec": 10,          # ใช้ร่วมกับ live (จะดึง 1 นาทีล่าสุดทุก ๆ นาที)
    "steps": 2000,                   # ใช้กับ paper
    "strategy_name": "smart",
    "strategy": {
        "atr_period": 14,
        "rsi_period": 14,
        "adx_period": 14,
        "supertrend_period": 10,
        "supertrend_mult": 3.0,
        "adx_floor": 20.0,
        "rsi_buy": 50.0,
        "rsi_sell": 45.0,
        "stop_atr_mult": 1.8,
        "take_atr_mult": 3.0,
        "trail_atr_mult": 1.2,
        "warmup": 80
    },
    # ⚠ HARD-CODED API KEYS for demo (คุณแก้ให้เป็นของคุณเองได้)
    "binance_api_key": "YPS8vLyG2c9ISjf2rmte1WC6J1eiet5BRNs3ODejGC6Yi5F7GXBeurXBoVdwJauN",
    "binance_api_secret": "ADVqsxb4Ju4pIkzVs7pB3HUMmYJzPp0doWHwERc8W87M67nauTnNnOi7snvfvqz0",
    "log_level": "INFO",

# ===== AUTOTRADE: BUY-ON-START + RSI SIGNAL =====
  # Binance futures: notional must be >= 20 USDT
# ===============================================

}
FORCE_TEST_ORDER_ON_START = True      # ยิง BUY ครั้งเดียวตอนเริ่ม
USE_PERCENT_BALANCE = 1.0            # ใช้ 5% ของ USDT ใน Futures wallet
RSI_PERIOD = 14
RSI_TF = "1m"
RSI_LOW = 30
RSI_HIGH = 70
TRADE_NOTIONAL_USDT = 5.0  # SPOT: ยิงไม้ละ 5 USDT (จะถูกยกระดับตาม cost.min ถ้าจำเป็น)
MIN_NOTIONAL_FLOOR = 20.0

MIN_COOLDOWN_SEC = 60  # ไม่ยิงซ้ำภายใน 60 วินาที

logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"], logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("neo_smart_live.log", encoding="utf-8")]
)
log = logging.getLogger("NeoSmartLive")

def clamp(v, a, b): return max(a, min(b, v))
def now_ts() -> datetime: return datetime.utcnow()

# ---------------- PAPER FEED -----------------
class RealTimeSimulator:
    def __init__(self, symbol: str, start_price: float = 3000.0, vol_bps: float = 35.0, trend_bps: float = 1.0):
        self.symbol = symbol
        self.price = start_price
        self.vol_bps = vol_bps
        self.trend_bps = trend_bps
    async def next_bar(self, sec: int) -> Tuple[datetime, float, float, float]:
        # สร้าง OHLCV 1 บาร์โดยคร่าว ๆ
        o = self.price
        for _ in range(sec // 2):
            drift = self.trend_bps / 10000.0
            shock = np.random.normal(0, self.vol_bps / 10000.0)
            self.price = float(self.price * math.exp(drift + shock))
        h = max(o, self.price) * (1 + 0.0008)
        l = min(o, self.price) * (1 - 0.0008)
        c = self.price
        await asyncio.sleep(0)
        return now_ts(), float(o), float(h), float(l), float(c)

# ---------------- LIVE FEED (CCXT) -----------------
class LiveDataFeed:
    def __init__(self, ex, symbol: str):
        self.ex = ex
        self.symbol = symbol
    async def next_bar(self, sec: int) -> Tuple[datetime, float, float, float, float]:
        # ดึง OHLCV 1 นาทีล่าสุด
        loop = asyncio.get_running_loop()
        def fetch():
            return self.ex.fetch_ohlcv(self.symbol, timeframe="1m", limit=2, params={"recvWindow": 60000})
        ohlcv = await loop.run_in_executor(None, fetch)
        if not ohlcv or len(ohlcv) == 0:
            raise RuntimeError("Cannot fetch OHLCV")
        t, o, h, l, c, v = ohlcv[-1]
        # รอจนกว่าจะครบ 1 นาทีถัดไป
        await asyncio.sleep(sec)
        return datetime.utcfromtimestamp(t/1000.0), float(o), float(h), float(l), float(c)

# ---------------- Indicators -----------------
def rsi(closes: List[float], period: int) -> Optional[float]:
    n = period
    if len(closes) < n + 1: return None
    diffs = np.diff(closes[-(n+1):])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period + 1: return None
    tr_list = []
    for i in range(-period, 0):
        h = highs[i]; l = lows[i]; pc = closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    return float(np.mean(tr_list))

def adx(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period + 1: return None
    plus_dm = []; minus_dm = []; tr_list = []
    for i in range(-period, 0):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        h = highs[i]; l = lows[i]; pc = closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    tr_n = np.sum(tr_list)
    if tr_n == 0: return 0.0
    plus_di = (np.sum(plus_dm) / tr_n) * 100.0
    minus_di = (np.sum(minus_dm) / tr_n) * 100.0
    dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9) * 100.0
    return float(dx)

def supertrend(highs: List[float], lows: List[float], closes: List[float], period: int, mult: float) -> Tuple[Optional[float], Optional[str]]:
    if len(closes) < period + 2: return None, None
    _atr = atr(highs, lows, closes, period)
    if _atr is None: return None, None
    mid = (highs[-1] + lows[-1]) / 2.0
    upper = mid + mult * _atr
    lower = mid - mult * _atr
    direction = "up" if closes[-1] > lower else "down"
    line = lower if direction == "up" else upper
    return float(line), direction

@dataclass
class SmartSignal:
    action: Optional[str]
    reason: str
    atr: Optional[float] = None

class SmartCompositeStrategy:
    def __init__(self, cfg: Dict):
        s = cfg["strategy"]
        self.atr_p = s["atr_period"]
        self.rsi_p = s["rsi_period"]
        self.adx_p = s["adx_period"]
        self.st_p = s["supertrend_period"]
        self.st_m = s["supertrend_mult"]
        self.adx_floor = s["adx_floor"]
        self.rsi_buy = s["rsi_buy"]
        self.rsi_sell = s["rsi_sell"]
        self.stop_k = s["stop_atr_mult"]
        self.take_k = s["take_atr_mult"]
        self.trail_k = s["trail_atr_mult"]
        self.warmup = s["warmup"]
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []

    def on_bar(self, o: float, h: float, l: float, c: float) -> SmartSignal:
        self.highs.append(h); self.lows.append(l); self.closes.append(c)
        if len(self.closes) < self.warmup:
            return SmartSignal("hold", "warmup", None)
        _atr = atr(self.highs, self.lows, self.closes, self.atr_p)
        _rsi = rsi(self.closes, self.rsi_p)
        _adx = adx(self.highs, self.lows, self.closes, self.adx_p)
        st_line, st_dir = supertrend(self.highs, self.lows, self.closes, self.st_p, self.st_m)
        if None in (_atr, _rsi, _adx) or st_dir is None:
            return SmartSignal("hold", "insufficient_indicators", _atr)
        if _adx >= self.adx_floor and st_dir == "up" and _rsi >= self.rsi_buy:
            return SmartSignal("buy", f"adx={_adx:.1f} st=up rsi={_rsi:.1f}", _atr)
        if st_dir == "down" or _rsi < self.rsi_sell:
            return SmartSignal("sell", f"st={st_dir} rsi={_rsi:.1f}", _atr)
        return SmartSignal("hold", "hold", _atr)

# ---------------- Risk Manager -----------------
class RiskManager:
    def __init__(self, cfg: Dict):
        self.risk = cfg["risk_per_trade"]
        self.fee = cfg["fee_rate"]
        self.slip = cfg["slippage"]
        self.stop_k = cfg["strategy"]["stop_atr_mult"]
    def size_by_risk(self, equity: float, price: float, _atr: Optional[float]) -> float:
        stop_dist = price * 0.01
        if _atr is not None and _atr > 0: stop_dist = self.stop_k * _atr
        risk_amt = equity * self.risk
        if stop_dist <= 0: return 0.0
        qty_risk = risk_amt / stop_dist
        eff_price = price * (1 + self.slip)
        qty_balance = equity / (eff_price * (1 + self.fee))
        qty = clamp(min(qty_risk, qty_balance), 0.0, qty_balance) * 0.98
        return max(0.0, round(qty, 4))

# ---------------- Paper Broker -----------------
@dataclass
class Position:
    qty: float = 0.0
    entry: float = 0.0
    stop: Optional[float] = None
    take: Optional[float] = None
    trail_anchor: Optional[float] = None
    def unrealized(self, price: float) -> float:
        return (price - self.entry) * self.qty if self.qty else 0.0

class PaperBroker:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cfg.setdefault('tp_pct_first',0.010)
        self.cfg.setdefault('sl_pct_first',0.007)
        self.balance = cfg["initial_balance"]
        self.pos = Position()
        self.trades: List[Dict] = []
    def _fee(self, notional): return notional * self.cfg["fee_rate"]
    def _buy(self, price: float, qty: float):
        px = price * (1 + self.cfg["slippage"])
        cost = px * qty
        fee = self._fee(cost)
        if self.balance < cost + fee: raise RuntimeError("Insufficient balance")
        if self.pos.qty != 0: raise RuntimeError("Position already open")
        self.pos = Position(qty=qty, entry=px)
        self.balance -= (cost + fee)
        self.trades.append({"time": now_ts().isoformat(), "action": "BUY", "price": price, "qty": qty, "fee": fee})
    def _sell_flat(self, price: float) -> float:
        if self.pos.qty == 0: return 0.0
        px = price * (1 - self.cfg["slippage"])
        notional = px * self.pos.qty
        fee = self._fee(notional)
        realized = (px - self.pos.entry) * self.pos.qty - fee
        self.balance += (notional - fee)
        self.trades.append({"time": now_ts().isoformat(), "action": "SELL", "price": price, "qty": self.pos.qty, "fee": fee, "pnl": realized})
        self.pos = Position()
        return realized
    def equity(self, price: float) -> float:
        return self.balance + self.pos.unrealized(price)

# ---------------- CCXT Broker (Live) -----------------
class CCXTBroker:
    def __init__(self, cfg: Dict):
        import importlib
        self.cfg = cfg
        self.ccxt = importlib.import_module("ccxt")
        self.ex = None
    async def connect(self):
        loop = asyncio.get_running_loop()
        def build():
            ex = self.ccxt.binance({
                "apiKey": self.cfg["binance_api_key"],
                "secret": self.cfg["binance_api_secret"],
                "enableRateLimit": True,
                "rateLimit": 2400,
                "options": {
                    "defaultType": "future" if self.cfg["market_type"] == "futures" else "spot",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                    "fetchCurrencies": False
                }
            })
            return ex
        self.ex = await loop.run_in_executor(None, build)
        # sync time difference to avoid InvalidNonce
        try:
            await loop.run_in_executor(None, self.ex.load_time_difference)
        except Exception as e:
            log.warning(f"โหลด time difference ไม่สำเร็จ: {e}")
        if self.cfg["market_type"] == "futures":
            try:
                self.ex.set_leverage(self.cfg.get("leverage", 1), self.cfg["symbol"].replace("/", ""))
                log.info(f"ตั้งค่า leverage = {self.cfg.get('leverage', 1)} ที่ {self.cfg['symbol']}")
            except Exception as e:
                log.warning(f"ตั้ง leverage ไม่สำเร็จ (อาจเป็นสิทธิ์/โหมดบัญชี): {e}")
        log.info("CCXT connected (FUTURES)")
    async def market_buy(self, symbol, qty):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.ex.create_order(symbol, "market", "buy", qty))
    async def market_sell(self, symbol, qty):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.ex.create_order(symbol, "market", "sell", qty))

# ---------------- Trader Core -----------------


class Trader:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        # defaults for first-order TP/SL if not provided
        self.cfg.setdefault('tp_pct_first', 0.010)  # +1.0%
        self.cfg.setdefault('sl_pct_first', 0.007)  # -0.7%
        self.strategy = SmartCompositeStrategy(cfg)
        self.risk = RiskManager(cfg)
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.price: float = 0.0
        self.symbol = self.cfg["symbol"]
        self._test_order_sent = False
        self._last_signal = None
        self._last_order_ts = 0
        if cfg["dry_run"]:
            self.broker = PaperBroker(cfg)
            self.feed = RealTimeSimulator(cfg["symbol"])
            self.ccxt_broker = None
        else:
            self.broker = None
            self.feed = None
            self.ccxt_broker = CCXTBroker(cfg)

    def _cooldown_ok(self):

        return (pytime.time() - self._last_order_ts) >= MIN_COOLDOWN_SEC

    async def _price_step(self) -> float:
        loop = asyncio.get_running_loop()
        def _mkt():
            return self.ccxt_broker.ex.market(self.symbol)
        try:
            mkt = await loop.run_in_executor(None, _mkt)
            # Try explicit tickSize from filters
            try:
                filters = (mkt.get('info', {}) or {}).get('filters', [])
                for f in filters:
                    if f.get('filterType') in ('PRICE_FILTER', 'PERCENT_PRICE_BY_SIDE'):
                        ts = f.get('tickSize')
                        if ts is not None:
                            ts = float(ts)
                            if ts > 0:
                                return ts
            except Exception:
                pass
            # Fallback to precision.price
            prec = (mkt.get('precision', {}) or {}).get('price', None)
            if isinstance(prec, int) and prec >= 0:
                return 10 ** (-prec) if prec > 0 else 1.0
            return 0.01
        except Exception:
            return 0.01

    async def _amount_step(self) -> float:
        """อ่าน stepSize (LOT_SIZE) ของสัญญา เพื่อปัดจำนวนให้ถูกต้อง"""
        loop = asyncio.get_running_loop()
        def _mkt():
            return self.ccxt_broker.ex.market(self.symbol)
        try:
            mkt = await loop.run_in_executor(None, _mkt)
            step = None
            try:
                filters = (mkt.get('info', {}) or {}).get('filters', [])
                for f in filters:
                    if f.get('filterType') in ('LOT_SIZE', 'MARKET_LOT_SIZE'):
                        ss = f.get('stepSize')
                        if ss is not None:
                            step = float(ss)
                            break
            except Exception:
                step = None
            if step and step > 0:
                return step
            prec = (mkt.get('precision', {}) or {}).get('amount', None)
            if isinstance(prec, int) and prec >= 0:
                return 10 ** (-prec) if prec > 0 else 1.0
            return 0.001
        except Exception:
            return 0.001

    async def _min_notional_usdt(self) -> float:
        loop = asyncio.get_running_loop()
        def _mkt():
            return self.ccxt_broker.ex.market(self.symbol)
        try:
            mkt = await loop.run_in_executor(None, _mkt)
            min_cost = (mkt.get('limits', {}).get('cost', {}) or {}).get('min', None)
            if min_cost is None:
                return MIN_NOTIONAL_FLOOR
            try:
                return max(float(min_cost), float(MIN_NOTIONAL_FLOOR))
            except Exception:
                return MIN_NOTIONAL_FLOOR
        except Exception:
            return MIN_NOTIONAL_FLOOR

    async def _free_usdt_futures(self) -> float:
        loop = asyncio.get_running_loop()
        def _fb():
            try:
                return self.ccxt_broker.ex.fetch_balance({'type': 'future'})
            except Exception:
                return self.ccxt_broker.ex.fetch_balance()
        bal = await loop.run_in_executor(None, _fb)
        free_usdt = None
        try:
            free_usdt = bal.get('free', {}).get('USDT')
        except Exception:
            pass
        if free_usdt is None:
            free_usdt = bal.get('total', {}).get('USDT', 0.0)
        return float(free_usdt or 0.0)

    async def _place_tp_sl(self, side: str, qty: float, tp_price: float, sl_price: float) -> bool:
        """Place reduceOnly TP/SL orders to close the current position (robust to mode by toggling position_side)."""
        try:
            step = await self._amount_step()
            def _floor_to_step(v, s):
                import math
                return math.floor(float(v)/s)*s if s>0 else float(v)
            qty = float(f"{_floor_to_step(qty, max(step,0.001)):.3f}")
            if qty <= 0:
                logging.warning("TP/SL: qty หลังปัด step <= 0, ข้าม")
                return False
            close_side = 'sell' if side.lower()=='buy' else 'buy'
            loop = asyncio.get_running_loop()
            def _send_tp_sl(with_pos_side: bool):
                base = {'reduceOnly': True}
                if with_pos_side:
                    base['positionSide'] = 'LONG' if side.lower()=='buy' else 'SHORT'
                self.ccxt_broker.ex.create_order(self.symbol, type='takeProfitMarket', side=close_side, amount=qty, params={**base, 'stopPrice': float(tp_price)})
                self.ccxt_broker.ex.create_order(self.symbol, type='stopMarket', side=close_side, amount=qty, params={**base, 'stopPrice': float(sl_price)})
            attempts = [(True,'with positionSide'), (False,'without positionSide')]
            try:
                if not getattr(self.ccxt_broker,'is_hedge',False):
                    attempts = [(False,'without positionSide'), (True,'with positionSide')]
            except Exception:
                pass
            last_err = None
            for flag,label in attempts:
                try:
                    logging.info(f"ตั้ง TP/SL ({label}) qty={qty}, tp={tp_price}, sl={sl_price}")
                    await loop.run_in_executor(None, lambda: _send_tp_sl(flag))
                    logging.info("ตั้ง TP/SL เรียบร้อย")
                    return True
                except Exception as e:
                    last_err = e
                    logging.warning(f"ตั้ง TP/SL ไม่สำเร็จ ({label}): {e}")
                    continue
            logging.error(f"ตั้ง TP/SL ล้มเหลว: {last_err}")
            return False
        except Exception as e:
            logging.error(f"ตั้ง TP/SL ไม่สำเร็จ: {e}")
            return False

    async def _place_limit_notional(self, side: str, notional_usdt: float, price: float) -> float:
        """Place LIMIT by notional with proper step/tick and robust position_side handling. Returns filled qty (or intended amount)."""
        loop = asyncio.get_running_loop()
        min_cost = await self._min_notional_usdt()
        target_min = float(min_cost) * 1.01
        if notional_usdt < target_min:
            notional_usdt = target_min
        def _ceil_to_step(v, s):
            import math
            return math.ceil(float(v)/s)*s if s>0 else float(v)
        def _floor_to_step(v, s):
            import math
            return math.floor(float(v)/s)*s if s>0 else float(v)
        price_step = await self._price_step()
        q_price = _ceil_to_step(price, price_step) if side.lower()=='buy' else _floor_to_step(price, price_step)
        if q_price <= 0:
            q_price = float(price)
        step = await self._amount_step()
        step_eff = max(step, 0.001)
        raw_amount = notional_usdt / q_price
        amount = _floor_to_step(raw_amount, step_eff)
        # enforce min amount
        try:
            mkt = await loop.run_in_executor(None, lambda: self.ccxt_broker.ex.market(self.symbol))
            min_amt = float((mkt.get('limits', {}).get('amount', {}) or {}).get('min', 0.0) or 0.0)
        except Exception:
            min_amt = 0.0
        if min_amt and amount < min_amt:
            import math
            n = math.ceil((min_amt - amount)/step_eff)
            amount = amount + n*step_eff
        # ensure notional >= target_min
        import math
        curr_notional = amount*q_price
        if curr_notional < target_min - 1e-8*q_price:
            need = (target_min - curr_notional)/(q_price*step_eff)
            amount = amount + math.ceil(need)*step_eff
        amount = float(f"{amount:.3f}")
        logging.info(f"LIMIT debug -> min_cost={min_cost}, target_min={target_min:.6f}, price_step={price_step}, q_price={q_price}, step_eff={step_eff}, amount={amount:.3f}, notional={amount*q_price:.6f}")
        # try with/without positionSide
        def _params(with_pos):
            p = {}
            
            if with_pos:
                p['positionSide'] = 'LONG' if side.lower()=='buy' else 'SHORT'
            return p
        attempts = [(True,'with positionSide'), (False,'without positionSide')]
        try:
            if not getattr(self.ccxt_broker, 'is_hedge', False):
                attempts = [(False,'without positionSide'), (True,'with positionSide')]
        except Exception:
            pass
        last_err = None
        order = None
        for flag,label in attempts:
            try:
                params = _params(flag)
                logging.info(f"ส่ง LIMIT {label}: side={side}, amount={amount}, price={q_price}, params={params}")
                order = await loop.run_in_executor(None, lambda: self.ccxt_broker.ex.create_order(self.symbol, type='limit', side=side, amount=amount, price=float(q_price), params=params))
                break
            except Exception as e:
                last_err = e
                logging.warning(f"LIMIT สร้างไม่สำเร็จ ({label}): {e}")
                continue
        if order is None:
            logging.error(f"ORDER LIMIT FAIL: {last_err}")
            return 0.0
        logging.info(f"ORDER LIMIT RESULT: {order}")
        return float(order.get('filled') or amount)

    async def _place_market_notional(self, side: str, notional_usdt: float) -> bool:
        """สั่ง market order ตาม notional (USDT) พร้อมปรับจำนวนตาม stepSize เพื่อผ่านเกณฑ์ notional ขั้นต่ำ"""
        min_cost = await self._min_notional_usdt()
        if notional_usdt < min_cost:
            notional_usdt = min_cost
        loop = asyncio.get_running_loop()
        def _ticker():
            return self.ccxt_broker.ex.fetch_ticker(self.symbol)
        tk = await loop.run_in_executor(None, _ticker)
        last = float(tk['last'])
        raw_amount = notional_usdt / last
        step = await self._amount_step()
        def _floor_to_step(v,s):
            import math
            return math.floor(float(v)/s)*s if s>0 else float(v)
        amount = _floor_to_step(raw_amount, step)
        while amount * last < min_cost:
            amount = float(f"{(amount + max(step,0.001)):.8f}")
        logging.info(f"ส่ง {side.upper()} {amount} {self.symbol} (~{amount*last:.2f} USDT) ที่ราคา {last} (step={step})")
        try:
            order = await loop.run_in_executor(None, lambda: self.ccxt_broker.ex.create_order(self.symbol, type='market', side=side, amount=amount))
            logging.info(f"ORDER RESULT: {order}")
            return True
        except Exception as e:
            logging.error(f"ORDER FAIL: {e}")
            return False

    def _compute_rsi(self, closes, period: int = 14):
        import pandas as _pd
        s = _pd.Series(closes)
        if len(s) < period + 1:
            return None
        delta = s.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        avg_gain = up.rolling(window=period, min_periods=period).mean()
        avg_loss = down.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    async def control_loop(self):
        if not self.cfg["dry_run"]:
            await self.ccxt_broker.connect()
            self.feed = LiveDataFeed(self.ccxt_broker.ex, self.cfg["symbol"])
        pos_qty = 0.0
        last_stop = None
        last_take = None
        trail_anchor = None
        steps = self.cfg["steps"] if self.cfg["dry_run"] else 10**9
        for _ in range(steps):
            ts, o, h, l, c = await self.feed.next_bar(self.cfg["bar_interval_sec"])
            self.price = c
            # BUY once on start
            if False and not self._test_order_sent:  # disabled: initial test buy
                free_usdt = await self._free_usdt_futures() if not self.cfg['dry_run'] else self.cfg['initial_balance']
                use_usdt = min(free_usdt, free_usdt * USE_PERCENT_BALANCE)
                min_cost = await self._min_notional_usdt()
                if use_usdt < min_cost and free_usdt >= min_cost:
                    use_usdt = min_cost
                if use_usdt < min_cost:
                    logging.warning(f"USDT ไม่พอสำหรับยิงทดสอบ (free≈{free_usdt:.2f} < {min_cost:.2f})")
                    self._test_order_sent = True
                else:
                    if self._cooldown_ok():
                        ok_qty = await self._place_limit_notional("buy", use_usdt, c) if not self.cfg['dry_run'] else 0.0
                        ok = ok_qty > 0 or self.cfg['dry_run']
                        if ok:

                            self._test_order_sent = True
                    self._last_order_ts = pytime.time()
                    # set TP/SL for the first test order immediately
                    if not self.cfg['dry_run'] and ok_qty > 0:
                        tp = c * (1.0 + float(self.cfg.get('tp_pct_first', 0.010)))
                        sl = c * (1.0 - float(self.cfg.get('sl_pct_first', 0.007)))
                        try:
                            placed = await self._place_tp_sl('buy', ok_qty, tp, sl)
                            if placed:
                                logging.info(f"ตั้ง TP/SL สำหรับไม้แรกเรียบร้อย (tp={tp:.2f}, sl={sl:.2f}, qty={ok_qty})")
                            else:
                                logging.warning("ตั้ง TP/SL สำหรับไม้แรกไม่สำเร็จ")
                        except Exception as e:
                            logging.warning(f"ข้อผิดพลาดในการตั้ง TP/SL ไม้แรก: {e}")
                    else:
                        logging.info("ข้ามเพราะยังไม่พ้นคูลดาวน์")
            # RSI signal
            try:
                loop = asyncio.get_running_loop()
                def _ohlcv():
                    return self.ccxt_broker.ex.fetch_ohlcv(self.symbol, timeframe=RSI_TF, limit=100)
                ohlcv = await loop.run_in_executor(None, _ohlcv)
                closes = [row[4] for row in ohlcv]
                rsi_val = self._compute_rsi(closes, RSI_PERIOD)
                if rsi_val is not None:
                    logging.info(f"RSI ล่าสุด: {rsi_val:.2f}")
                    if not self.cfg['dry_run']:
                        free_usdt = await self._free_usdt_futures()
                        use_usdt = min(free_usdt, free_usdt * USE_PERCENT_BALANCE)
                        min_cost = await self._min_notional_usdt()
                        if use_usdt < min_cost and free_usdt >= min_cost:
                            use_usdt = min_cost
                        if use_usdt >= min_cost:
                            if rsi_val < RSI_LOW and self._last_signal != "buy":
                                if self._cooldown_ok():
                                    ok_qty = await self._place_limit_notional("buy", use_usdt, c)
                                    if ok_qty > 0:

                                        self._last_signal = "buy"
                                        self._last_order_ts = pytime.time()
                                else:
                                    logging.info("ข้ามเพราะยังไม่พ้นคูลดาวน์")
                            elif rsi_val > RSI_HIGH and self._last_signal != "sell":
                                if self._cooldown_ok():
                                    ok_qty = await self._place_limit_notional("sell", use_usdt, c)
                                    if ok_qty > 0:

                                        self._last_signal = "sell"
                                        self._last_order_ts = pytime.time()
                                else:
                                    logging.info("ข้ามเพราะยังไม่พ้นคูลดาวน์")
                        else:
                            logging.info("USDT ไม่พอขั้นต่ำ futures notional")
                else:
                    logging.info("RSI: ยังคำนวณไม่พอแท่ง")
            except Exception as e:
                logging.warning(f"RSI/Signal error: {e}")

            sig = self.strategy.on_bar(o, h, l, c)

            if pos_qty > 0 and trail_anchor is not None:
                trail_anchor = max(trail_anchor, c)
                if sig.atr:
                    last_stop = max(last_stop or 0, trail_anchor - self.cfg["strategy"]["trail_atr_mult"] * sig.atr)

            if pos_qty > 0:
                if last_stop and c <= last_stop:
                    # market close (paper/live simplified)
                    pos_qty, last_stop, last_take, trail_anchor = 0.0, None, None, None
                    self._record(ts); continue
                if last_take and c >= last_take:
                    pos_qty, last_stop, last_take, trail_anchor = 0.0, None, None, None
                    self._record(ts); continue

            if sig.action == "buy" and pos_qty == 0:
                equity = self.broker.equity(c) if self.cfg["dry_run"] else 1000.0
                qty = self.risk.size_by_risk(equity, c, sig.atr)
                if qty > 0:
                    sl = c - self.cfg["strategy"]["stop_atr_mult"] * (sig.atr or c*0.01)
                    tp = c + self.cfg["strategy"]["take_atr_mult"] * (sig.atr or c*0.02)
                    ok_qty = await self._place_limit_notional("buy", qty * c, c) if not self.cfg['dry_run'] else qty
                    if ok_qty > 0:
                        pos_qty = qty
                        await self._place_tp_sl("buy", qty, tp, sl)
                        last_stop, last_take = sl, tp
                        trail_anchor = c if sig.atr else None
            elif sig.action == "sell" and pos_qty > 0:
                # simplified close
                pos_qty, last_stop, last_take, trail_anchor = 0.0, None, None, None

            self._record(ts)
            await asyncio.sleep(0)

    def _record(self, ts):
        if self.cfg["dry_run"]:
            self.equity_curve.append((ts, self.broker.equity(self.price)))

    async def main(self):
        await self.control_loop()
        out = Path("outputs"); out.mkdir(exist_ok=True)
        if self.cfg["dry_run"]:
            pd.DataFrame(self.broker.trades).to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(self.equity_curve, columns=["time", "equity"]).to_csv(out / "equity.csv", index=False, encoding="utf-8-sig")
            summary = {
                "final_balance": self.broker.balance,
                "final_equity": self.broker.equity(self.price),
                "num_trades": int(sum(1 for t in self.broker.trades if t["action"] == "BUY")),
                "symbol": self.cfg["symbol"],
                "dry_run": self.cfg["dry_run"],
            }
            with open(out / "summary.json", "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info(f"เสร็จสิ้น (โหมด {'paper' if self.cfg['dry_run'] else 'live'})  — logs: neo_smart_live.log")


import pandas as pd
from datetime import datetime, timedelta
def backtest_last_year(symbol: str, exchange):
    now = datetime.utcnow()
    since = int((now - timedelta(days=365)).timestamp() * 1000)
    logging.info(f"ดึงข้อมูลย้อนหลัง 1 ปี ของ {symbol} ...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.to_csv("backtest_1y.csv", index=False)
    logging.info("✅ บันทึกผลเป็น backtest_1y.csv แล้ว")
    print("\n=== 20 แถวล่าสุดจาก backtest ===")
    print(df.tail(20).to_string(index=False))
# ===== END BACKTEST =====



if __name__ == "__main__":
    import asyncio, ccxt, logging

    # --- Backtest 1Y then export + print tail(20) ---
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        backtest_last_year(CONFIG["symbol"], ex)
    except Exception as e:
        logging.error(f"Backtest error: {e}")

    # --- Start live trading loop ---
    trader = Trader(CONFIG)
    asyncio.run(trader.main())
