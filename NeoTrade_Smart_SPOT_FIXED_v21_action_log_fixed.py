import os, sys, math, asyncio, logging, time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import ccxt
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG: Dict = {
    "dry_run": False,
    "exchange": "binance",
    "market_type": "spot",
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "bar_interval_sec": 60,
    "initial_balance": 10.0,
    "risk_per_trade": 0.02,
    "fee_rate": 0.0004,
    "slippage": 0.0005,
    "rsi_period": 21,
    "log_level": "INFO",
    "binance_api_key": "MCGYRJfCGO17sekSHa4jERZGDpX3rcJkapTJUJhABdIqgVfOrC8R1ZBgSagNxMJV",
    "binance_api_secret": "HTN2iPBCbzovxIE5phCh7PAtkBuQZW1e540F1KulZj1u22U8CaQxbkyCs27wZcs4",
    "steps": 2000,
    "tp_pct": 0.01,
    "sl_pct": 0.005,
}

logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"], logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("neo_smart_live.log", encoding="utf-8")]
)
log = logging.getLogger("NeoSmartLive")

# ===== Indicators / Models (เหมือนเดิม) =====
def smma(values: np.ndarray, period: int) -> np.ndarray:
    smma_vals = np.full_like(values, np.nan, dtype=float)
    smma_vals[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        smma_vals[i] = (smma_vals[i - 1] * (period - 1) + values[i]) / period
    return smma_vals

def rsi(closes: List[float], period: int = 21) -> Optional[float]:
    closes = np.array(closes)
    if len(closes) < period + 1:
        return None
    diffs = np.diff(closes[-period - 1:])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = smma(gains, period)[-1]
    avg_loss = smma(losses, period)[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    tr_list = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr_list = np.maximum(tr_list, np.abs(lows[1:] - closes[:-1]))
    atr_vals = smma(tr_list[-period:], period)[-1]
    return float(atr_vals)

def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
    minus_dm = np.maximum(lows[:-1] - lows[1:], 0)
    tr_list = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr_list = np.maximum(tr_list, np.abs(lows[1:] - closes[:-1]))
    plus_dm_sm = smma(plus_dm[-period:], period)[-1]
    minus_dm_sm = smma(minus_dm[-period:], period)[-1]
    tr_sm = smma(tr_list[-period:], period)[-1]
    if tr_sm == 0:
        return None
    plus_di = (plus_dm_sm / tr_sm) * 100
    minus_di = (minus_dm_sm / tr_sm) * 100
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
    return float(dx)

def supertrend(highs: List[float], lows: List[float], closes: List[float], period: int, mult: float):
    if len(closes) < period + 2:
        return None, None
    atr_val = atr(highs[-period - 1:], lows[-period - 1:], closes[-period - 1:], period)
    if atr_val is None:
        return None, None
    mid = (highs[-1] + lows[-1]) / 2.0
    upper = mid + mult * atr_val
    lower = mid - mult * atr_val
    direction = "up" if closes[-1] > upper else "down"
    line = lower if direction == "up" else upper
    return float(line), direction

def bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0):
    if len(closes) < period + 1:
        return None, None, None
    mean = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper_band = mean + (std_dev * std)
    lower_band = mean - (std_dev * std)
    return float(mean), float(upper_band), float(lower_band)

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ===== Dataset + Training (เหมือนเดิม) =====
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["close"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    rsi_vals = [rsi(closes[:i+1]) for i in range(len(closes))]
    atr_vals = [atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
    adx_vals = [adx(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
    st_line, st_dir = [], []
    for i in range(len(closes)):
        line, dirc = supertrend(highs[:i+1], lows[:i+1], closes[:i+1], 10, 3.0)
        st_line.append(line); st_dir.append(1 if dirc=="up" else -1 if dirc=="down" else 0)
    bb_mid, bb_up, bb_low = [], [], []
    for i in range(len(closes)):
        m,u,l = bollinger_bands(closes[:i+1])
        bb_mid.append(m); bb_up.append(u); bb_low.append(l)
    df["rsi"] = rsi_vals
    df["atr"] = atr_vals
    df["adx"] = adx_vals
    df["st_line"] = st_line
    df["st_dir"] = st_dir
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_up
    df["bb_low"] = bb_low
    return df

def create_dataset(df: pd.DataFrame, lookback=30):
    features = df[["rsi","atr","adx","st_line","bb_mid","bb_up"]].values
    labels = np.sign(np.diff(df["close"].ffill()))
    labels = np.append(labels, 0)
    labels = np.where(labels == -1, 0, labels)   # -1 → 0
    labels = np.where(labels == 0, 1, labels)    # 0 → 1
    labels = np.where(labels == 1, 2, labels)    # 1 → 2
    X, y = [], []
    for i in range(len(features)-lookback):
        if np.any(np.isnan(features[i:i+lookback])):
            continue
        X.append(features[i:i+lookback])
        y.append(int(labels[i+lookback]))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

def train_lstm(model, X, y, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# ===== Risk/Order/Position/Equity/Broker (เหมือนเดิม) =====
class RiskManager:
    def __init__(self, cfg: Dict):
        self.risk = cfg["risk_per_trade"]
        self.fee = cfg["fee_rate"]
        self.slip = cfg["slippage"]
    def size_by_risk(self, equity: float, price: float, atr_val: Optional[float]) -> float:
        stop_dist = price * 0.01 if not atr_val else atr_val
        risk_amt = equity * self.risk
        if stop_dist <= 0: return 0.0
        qty_risk = risk_amt / stop_dist
        eff_price = price * (1 + self.slip)
        qty_balance = equity / (eff_price * (1 + self.fee))
        qty = min(qty_risk, qty_balance) * 0.98
        return max(0.0, round(qty, 6))

class OrderManager:
    def __init__(self, broker):
        self.broker=broker
    def market(self,symbol,side,qty):
        return self.broker.market_order(symbol,side,qty)
    def oco(self,symbol,side,qty,tp,sl):
        return self.broker.oco_order(symbol,side,qty,tp,sl)
    def cancel(self,order_id):
        return None

class PositionManager:
    def __init__(self):
        self.positions={}
    def update(self,symbol,side,qty,price):
        if side=="buy":
            pos=self.positions.get(symbol,{"qty":0,"entry":0})
            new_qty=pos["qty"]+qty
            pos["entry"]=(pos["entry"]*pos["qty"]+price*qty)/max(new_qty,1e-8)
            pos["qty"]=new_qty
            self.positions[symbol]=pos
        elif side=="sell":
            if symbol in self.positions and self.positions[symbol]["qty"]>=qty:
                self.positions[symbol]["qty"]-=qty
                if self.positions[symbol]["qty"]<=0:
                    del self.positions[symbol]
    def get_position(self,symbol):
        return self.positions.get(symbol,{"qty":0,"entry":0})

class EquityManager:
    def __init__(self,initial:float):
        self.initial=initial
        self.curve=[initial]
    def update(self,balance:float):
        self.curve.append(balance)
    def metrics(self):
        arr=np.array(self.curve)
        ret=np.diff(arr)/arr[:-1]
        sharpe=np.mean(ret)/np.std(ret)*np.sqrt(252) if np.std(ret)!=0 else 0
        drawdown=np.max(arr)-np.min(arr)
        return {"final":arr[-1],"sharpe":sharpe,"drawdown":drawdown}

class PaperBroker:
    def __init__(self, cfg: Dict):
        self.balance = cfg["initial_balance"]
        self.positions = {}
    def market_order(self, symbol: str, side: str, qty: float):
        price=100.0
        cost=qty*price
        if side=="buy":
            if self.balance>=cost:
                self.balance-=cost
                self.positions[symbol]=self.positions.get(symbol,0)+qty
                return True
        elif side=="sell":
            if self.positions.get(symbol,0)>=qty:
                self.positions[symbol]-=qty
                self.balance+=cost
                return True
        return False
    def oco_order(self, symbol:str, side:str, qty:float, tp:float, sl:float):
        return True

class CCXTBroker:
    def __init__(self, cfg: Dict):
        self.ex = ccxt.binance({
            "apiKey": cfg["binance_api_key"],
            "secret": cfg["binance_api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
    def market_order(self, symbol: str, side: str, qty: float):
        try:
            return self.ex.create_order(symbol,"market",side,qty)
        except Exception as e:
            log.error(f"Market order error {e}")
            return None
    def oco_order(self, symbol:str, side:str, qty:float, tp:float, sl:float):
        try:
            sym=symbol.replace("/","")
            sl_limit_price=sl*0.999 if side.lower()=="sell" else sl*1.001
            return self.ex.private_post_order_oco({
                "symbol": sym,
                "side": side.upper(),
                "quantity": qty,
                "price": f"{tp:.2f}",
                "stopPrice": f"{sl:.2f}",
                "stopLimitPrice": f"{sl_limit_price:.2f}",
                "stopLimitTimeInForce": "GTC"
            })
        except Exception as e:
            log.error(f"OCO error {e}")
            return None

# ===== Trader Loop (แก้ log) =====
class Trader:
    def __init__(self,cfg:Dict,symbol:str,model:LSTMModel):
        self.cfg=cfg
        self.symbol=symbol
        self.model=model
        self.lookback=30
        self.risk=RiskManager(cfg)
        self.broker=PaperBroker(cfg) if cfg["dry_run"] else CCXTBroker(cfg)
        self.orders=OrderManager(self.broker)
        self.positions=PositionManager()
        self.equity=EquityManager(cfg["initial_balance"])
        self.last_ts = None

    async def loop(self):
        while True:
            try:
                for attempt in range(3):
                    try:
                        ohlcv=self.broker.ex.fetch_ohlcv(self.symbol,timeframe=self.cfg["timeframe"],limit=self.lookback+1) if not self.cfg["dry_run"] else ccxt.binance().fetch_ohlcv(self.symbol,timeframe=self.cfg["timeframe"],limit=self.lookback+1)
                        break
                    except Exception as e:
                        log.warning(f"Fetch OHLCV retry {attempt+1}/3 failed: {e}")
                        await asyncio.sleep(2)
                else:
                    log.error(f"Fetch OHLCV failed after 3 retries for {self.symbol}")
                    await asyncio.sleep(self.cfg["bar_interval_sec"])
                    continue

                df=pd.DataFrame(ohlcv,columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df=prepare_features(df)
                latest_ts = df["timestamp"].iloc[-2]
                latest_rsi = df["rsi"].iloc[-2]

                if latest_ts != self.last_ts:
                    self.last_ts = latest_ts
                    log.info(f"{self.symbol} | New Candle {latest_ts} | RSI: {latest_rsi:.2f}")

                X,y=create_dataset(df,lookback=self.lookback)
                if len(X)==0: continue
                with torch.no_grad():
                    out=self.model(X[-1:].unsqueeze(0))
                    pred=torch.argmax(out,dim=1).item()

                action={0:"HOLD",1:"BUY",2:"SELL"}.get(pred,"HOLD")

                # === แสดง Prediction ทุกแท่ง ===
                log.info(f"{self.symbol} | Prediction={action} | RSI={latest_rsi:.2f}")

                # === ตัดสินใจว่าจะส่ง order จริงไหม ===
                if action == "BUY" and latest_rsi < 30:
                    do_order = True
                elif action == "SELL" and latest_rsi > 70:
                    do_order = True
                else:
                    do_order = False

                if do_order:
                    price = float(df["close"].iloc[-1])
                    equity=self.cfg["initial_balance"]
                    size=self.risk.size_by_risk(equity,price,df["atr"].iloc[-1])
                    if size>0:
                        if self.cfg["dry_run"]:
                            order = self.broker.market_order(self.symbol,action.lower(),size)
                            log.info(f"TRADE EXECUTED: {self.symbol} | Side={action} | Size={size} | Price={price} | RSI={latest_rsi:.2f}")

                        # === เพิ่ม TP/SL ด้วย OCO ===
                        if action == "BUY":
                            tp_price = price * (1 + self.cfg["tp_pct"])
                            sl_price = price * (1 - self.cfg["sl_pct"])
                        elif action == "SELL":
                            tp_price = price * (1 - self.cfg["tp_pct"])
                            sl_price = price * (1 + self.cfg["sl_pct"])
                        else:
                            tp_price = sl_price = None

                        if tp_price and sl_price:
                            oco = self.orders.oco(self.symbol, action.lower(), size, tp_price, sl_price)
                            log.info(f"OCO SET for {self.symbol} | Side={action} | Size={size} | TP={tp_price:.2f} | SL={sl_price:.2f}")
                            self.positions.update(self.symbol,action.lower(),size,price)
                            self.equity.update(self.broker.balance if self.cfg["dry_run"] else equity)
                else:
                    log.info(f"{self.symbol} | HOLD (No trade executed)")

                self.model=train_lstm(self.model,X[-10:],y[-10:],epochs=1)

            except Exception as e:
                log.error(f"Trader error {self.symbol}: {e}")
            await asyncio.sleep(self.cfg["bar_interval_sec"])

async def main():
    models={}
    for sym in CONFIG["symbols"]:
        models[sym]=backtest_last_5y(sym,timeframe=CONFIG["timeframe"])
    traders=[Trader(CONFIG,sym,models[sym]).loop() for sym in CONFIG["symbols"] if models[sym]]
    await asyncio.gather(*traders)

def backtest_last_5y(symbol: str, timeframe="1d"):
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    since = int((datetime.utcnow() - timedelta(days=365*5)).timestamp() * 1000)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = prepare_features(df)
    print(df.tail(20))
    df.to_csv(f"backtest_5y_{symbol.replace('/', '')}.csv", index=False)
    X, y = create_dataset(df)
    if len(X)==0: return None
    model = LSTMModel()
    train_lstm(model, X, y, epochs=5)
    return model

if __name__=="__main__":
    asyncio.run(main())
