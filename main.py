"""
Kenne Index 自动定投系统 - Web 后端
启动: uvicorn main:app --reload
访问: http://127.0.0.1:8000

依赖: pip install fastapi uvicorn ccxt pandas numpy scipy requests
"""

import json, os, csv, smtplib, logging, datetime, time
from pathlib import Path
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from typing import Optional
from fastapi.concurrency import run_in_threadpool
import asyncio
import requests
import redis.asyncio as aioredis

import pandas as pd
import numpy as np
from scipy.stats import gmean
from scipy import stats as scipy_stats
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── 常量 ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
CONFIG_FILE    = BASE_DIR / "config.json"
LOG_FILE       = BASE_DIR / "dca_log.json"
MODEL_FILE     = BASE_DIR / "model_params.json"
TEMPLATES_DIR  = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

DATA_FILES = {
    "BTC": str(BASE_DIR / "btc_4h_data_2018_to_2025.csv"),
    "ETH": str(BASE_DIR / "eth_4h_data_2017_to_2025.csv"),
    "SOL": str(BASE_DIR / "sol_4h_data_2020_to_2025.csv"),
}

# 支持的交易所（ccxt id → 显示名称）
SUPPORTED_EXCHANGES = {
    "okx":      "OKX",
    "binance":  "Binance",
    "bybit":    "Bybit",
    "bitget":   "Bitget",
    "gateio":   "Gate.io",
    "kucoin":   "KuCoin",
    "htx":      "HTX (Huobi)",
    "mexc":     "MEXC",
}

# 交易对映射（各交易所的标准符号）
TRADING_PAIRS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("web")

# ─── Redis & 缓存配置 ──────────────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "CG-rxSoLwJXenPXvGX6YaxVFyHs")

redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

async def fetch_with_retry(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await run_in_threadpool(func, *args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                log.error(f"Function {func.__name__} failed after {retries} retries: {e}")
                raise e
            log.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s...")
            await asyncio.sleep(delay)

@dataclass
class CoinConfig:
    symbol: str
    slope: float
    intercept: float
    genesis: str
    buy_thresh: float
    dca_thresh: float
    knife_7d: float
    knife_14d: float
    bounce_min: float
    r2: float
    data_years: float

COIN_CONFIG = {
    "BTC": CoinConfig("BTC", 4.7777, -13.1486, "2009-01-03",
                      0.45, 1.20, -0.15, -0.25, 0.05, 0.78, 15),
    "ETH": CoinConfig("ETH", 1.9872, -3.5997, "2015-07-30",
                      0.45, 1.20, -0.15, -0.25, 0.05, 0.58, 10),
    "SOL": CoinConfig("SOL", 1.4446, -2.5934, "2020-03-16",
                      0.45, 1.50, -0.13, -0.22, 0.07, 0.53, 5.5),
}

MOMENTUM_MULT = {"STABLE": 1.00, "STABILIZING": 0.75, "FALLING": 0.40}
MAX_WEIGHT    = {"BTC": 0.60, "ETH": 0.50, "SOL": 0.50}
MIN_ORDER_USDT = 5.0
MIN_DAYS_REFIT = 365

# ─── 配置管理 ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "exchange": "okx",
    "api_key": "", "api_secret": "", "api_passphrase": "",
    "simulated": True,
    "budget_mode": "MONTHLY",
    "budget_amount": 700,
    "run_interval_days": 7,
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "", "smtp_password": "", "email_to": "",
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

# ─── 幂律重拟合 ────────────────────────────────────────────────────────────────

def _refit(df_d: pd.DataFrame, cfg: CoinConfig):
    df_v = df_d[df_d["days"] > 0].dropna(subset=["Close"])
    if len(df_v) < MIN_DAYS_REFIT:
        return cfg.slope, cfg.intercept, cfg.r2
    x = np.log10(df_v["days"].values)
    y = np.log10(df_v["Close"].values)
    slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
    return slope, intercept, r ** 2

def load_model_params() -> dict:
    if MODEL_FILE.exists():
        try:
            return json.loads(MODEL_FILE.read_text())
        except Exception:
            pass
    return {}

def save_model_params(params: dict):
    MODEL_FILE.write_text(json.dumps(params, indent=2, ensure_ascii=False))

# ─── Kenne Index 信号计算 ──────────────────────────────────────────────────────

def compute_signal(symbol: str) -> Optional[dict]:
    symbol = symbol.upper()
    cfg = COIN_CONFIG.get(symbol)
    if not cfg:
        return None

    csv_path = DATA_FILES.get(symbol)
    if not csv_path or not Path(csv_path).exists():
        return {"symbol": symbol, "error": "CSV 数据文件不存在"}

    try:
        df = pd.read_csv(csv_path)
        df["Open time"] = pd.to_datetime(df["Open time"], format="mixed")
        df.set_index("Open time", inplace=True)
        df_d = df.resample("D").agg({"Close": "last", "Low": "min", "High": "max"}).dropna()

        genesis = pd.to_datetime(cfg.genesis)
        df_d["days"] = (df_d.index - genesis).days
        df_d = df_d[df_d["days"] > 0]

        # 幂律重拟合
        slope, intercept, r2 = _refit(df_d, cfg)
        data_years = round(len(df_d) / 365, 1)

        df_d["valuation"] = 10 ** (slope * np.log10(df_d["days"]) + intercept)
        df_d["cost_200"]  = df_d["Close"].rolling(200).apply(gmean, raw=True)
        df_d["kenne_index"] = (df_d["Close"] / df_d["cost_200"]) * (df_d["Close"] / df_d["valuation"])

        df_d["ret_7d"]   = df_d["Close"].pct_change(7)
        df_d["ret_14d"]  = df_d["Close"].pct_change(14)
        df_d["low7"]     = df_d["Low"].rolling(7).min()
        df_d["bounce_7"] = (df_d["Close"] - df_d["low7"]) / df_d["low7"]
        df_d = df_d.dropna()

        if df_d.empty:
            return {"symbol": symbol, "error": "数据不足（需至少 200 日）"}

        row = df_d.iloc[-1]
        ki  = row["kenne_index"]

        # 动量状态
        is_knife = (row["ret_7d"] < cfg.knife_7d) or (row["ret_14d"] < cfg.knife_14d)
        has_bounce = row["bounce_7"] >= cfg.bounce_min
        if   is_knife and not has_bounce: momentum = "FALLING"
        elif is_knife and has_bounce:     momentum = "STABILIZING"
        else:                             momentum  = "STABLE"

        # 倍数
        if   ki < cfg.buy_thresh:  base_mult = 2.0
        elif ki <= cfg.dca_thresh: base_mult = 1.0
        else:                      base_mult = 0.0
        final_mult = base_mult * MOMENTUM_MULT[momentum]

        # 评分
        if ki < cfg.buy_thresh:
            ahr_s = 50
        elif ki <= cfg.dca_thresh:
            ratio = 1 - (ki - cfg.buy_thresh) / (cfg.dca_thresh - cfg.buy_thresh)
            ahr_s = int(10 + 30 * ratio)
        else:
            ahr_s = max(0, 10 - int((ki - cfg.dca_thresh) * 5))
        score = min(100, ahr_s + {"STABLE": 50, "STABILIZING": 25, "FALLING": 0}[momentum])

        pct_rank = float((df_d["kenne_index"] < ki).mean() * 100)
        pct = df_d["kenne_index"].quantile([0.05, 0.25, 0.50, 0.75]).to_dict()

        if   ki < cfg.buy_thresh:  zone = "极低估"
        elif ki <= cfg.dca_thresh: zone = "定投区"
        else:                      zone = "观望区"

        return {
            "symbol":     symbol,
            "price":      float(row["Close"]),
            "cost_200":   float(row["cost_200"]),
            "valuation":  float(row["valuation"]),
            "kenne_index": float(round(ki, 4)),
            "zone":        zone,
            "momentum":    momentum,
            "ret_7d":      float(round(row["ret_7d"] * 100, 2)),
            "ret_14d":     float(round(row["ret_14d"] * 100, 2)),
            "base_mult":   base_mult,
            "final_mult":  final_mult,
            "score":       score,
            "pct_rank":    round(pct_rank, 1),
            "pct":         {str(k): round(v, 4) for k, v in pct.items()},
            "slope":       round(slope, 4),
            "r2":          round(r2, 4),
            "data_years":  data_years,
            "date":        row.name.strftime("%Y-%m-%d"),
        }
    except Exception as e:
        log.error(f"[{symbol}] signal error: {e}")
        return {"symbol": symbol, "error": str(e)}

# ─── 资金分配 ──────────────────────────────────────────────────────────────────

def allocate(signals: list, budget_usdt: float) -> list:
    active = [s for s in signals if s.get("final_mult", 0) > 0 and "error" not in s]
    if not active or budget_usdt <= 0:
        return []
    norm = {s["symbol"]: s["final_mult"] for s in active}
    for _ in range(3):
        total = sum(norm.values())
        norm  = {k: v / total for k, v in norm.items()}
        norm  = {k: min(v, MAX_WEIGHT.get(k, 1.0)) for k, v in norm.items()}
    total = sum(norm.values())
    if total == 0:
        return []
    norm = {k: v / total for k, v in norm.items()}
    return [{**s,
             "usdt_amount": round(norm.get(s["symbol"], 0) * budget_usdt, 2),
             "weight":      round(norm.get(s["symbol"], 0), 4)}
            for s in active]

# ─── 预算管理 ──────────────────────────────────────────────────────────────────

@dataclass
class Record:
    ts: str; symbol: str; exchange: str; usdt: float
    kenne_index: float; mult: float; momentum: str
    order_id: str; status: str; note: str = ""
    price: float = 0.0
    qty: float = 0.0

def load_log() -> list:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            pass
    return []

def save_log(recs: list):
    LOG_FILE.write_text(json.dumps(recs, indent=2, ensure_ascii=False))

def spent_this_month(recs: list) -> float:
    m = datetime.date.today().strftime("%Y-%m")
    return sum(r.get("usdt", 0) for r in recs
               if r.get("ts", "").startswith(m)
               and r.get("status") in ("filled", "dry_run"))

def this_run_amount(cfg: dict, recs: list) -> float:
    mode     = cfg.get("budget_mode", "MONTHLY").upper()
    amount   = float(cfg.get("budget_amount", 700))
    interval = int(cfg.get("run_interval_days", 7))

    if mode == "FIXED":
        return amount

    # MONTHLY
    runs_per_month = 30.0 / interval
    target         = amount / runs_per_month
    remaining      = max(0.0, amount - spent_this_month(recs))
    if remaining <= 0:
        return 0.0
    today  = datetime.date.today()
    next_m = datetime.date(today.year + (today.month == 12), today.month % 12 + 1, 1)
    days_left = max(1, (next_m - today).days)
    runs_left = max(1, round(days_left / interval))
    return min(target, remaining / runs_left)

# ─── 交易所客户端 ──────────────────────────────────────────────────────────────

def make_exchange(cfg: dict):
    """创建 ccxt 交易所实例，仅在需要时调用（避免导入失败影响启动）。"""
    try:
        import ccxt
    except ImportError:
        raise RuntimeError("ccxt 未安装，请运行: pip install ccxt")

    ex_id = cfg.get("exchange", "okx").lower()
    if ex_id not in ccxt.exchanges:
        raise ValueError(f"不支持的交易所: {ex_id}")

    params = {
        "apiKey": cfg.get("api_key", ""),
        "secret": cfg.get("api_secret", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
    if cfg.get("api_passphrase"):
        params["password"] = cfg["api_passphrase"]

    ex = getattr(ccxt, ex_id)(params)

    # 模拟盘（仅 OKX 原生支持，其余交易所忽略此选项）
    if cfg.get("simulated") and ex_id == "okx":
        ex.set_sandbox_mode(True)

    return ex

def buy_usdt(ex, symbol: str, usdt: float) -> dict:
    """
    用 USDT 金额市价买入，兼容各交易所。
    优先用 createMarketBuyOrderWithCost（ccxt 标准），
    回退到手动计算数量。
    """
    pair = TRADING_PAIRS.get(symbol, f"{symbol}/USDT")
    try:
        # ccxt >= 3.x 统一接口：cost 参数指定花费的 USDT
        return ex.create_order(pair, "market", "buy", None,
                               params={"cost": usdt, "quoteOrderQty": usdt})
    except Exception:
        # 回退：先获取最新价格，换算数量
        ticker = ex.fetch_ticker(pair)
        price  = ticker["last"]
        amount = usdt / price * 0.999   # 留 0.1% buffer
        return ex.create_market_buy_order(pair, amount)

# ─── 邮件通知 ──────────────────────────────────────────────────────────────────

def send_email(cfg: dict, subject: str, body: str):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = cfg.get("smtp_user", "")
    msg["To"]      = cfg.get("email_to", "")
    host, port     = cfg.get("smtp_host", "smtp.gmail.com"), int(cfg.get("smtp_port", 587))
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.ehlo(); s.starttls()
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)
        return True, "邮件发送成功"
    except Exception as e:
        return False, str(e)

def build_report(cfg: dict, signals: list, allocs: list, recs: list) -> str:
    date   = datetime.date.today().strftime("%Y-%m-%d")
    mode   = cfg.get("budget_mode", "MONTHLY").upper()
    amount = float(cfg.get("budget_amount", 700))
    interval = int(cfg.get("run_interval_days", 7))
    run_amt  = this_run_amount(cfg, recs)

    il = {1: "日投", 7: "周投", 14: "双周投", 30: "月投"}.get(interval, f"每{interval}天投")
    budget_desc = f"每次固定 ${amount:.0f}" if mode == "FIXED" else f"${amount:.0f}/月  {il}"

    lines = [
        f"Kenne Index 定投信号  {date}",
        f"策略: {budget_desc}",
        "=" * 46, "",
        "当前信号", "-" * 46,
    ]
    for s in signals:
        if "error" in s:
            lines.append(f"  {s['symbol']}: {s['error']}")
            continue
        lines.append(
            f"  {s['symbol']:<4}  价格 {s['price']:>10,.2f} USDT  "
            f"Kenne {s['kenne_index']:.4f}  {s['zone']}  {s['momentum']}  建议 {s['final_mult']:.2f}x"
        )

    lines += ["", f"本次分配建议（${run_amt:.2f} USDT）", "-" * 46]
    if allocs:
        for a in allocs:
            lines.append(
                f"  {a['symbol']:<4}  ${a['usdt_amount']:>7.2f} USDT"
                f"  权重 {a['weight']:.0%}  倍数 {a['final_mult']:.2f}x"
            )
        lines.append(f"  合计    ${sum(a['usdt_amount'] for a in allocs):>7.2f} USDT")
    else:
        lines.append("  当前无买入信号，本次停止定投")

    lines += [""]
    if mode == "MONTHLY":
        spent = spent_this_month(recs)
        lines.append(f"月预算 ${amount:.0f}  已花 ${spent:.2f}  剩余 ${max(0, amount - spent):.2f}")
    else:
        lines.append(f"本月已投 ${spent_this_month(recs):.2f}（FIXED 模式，无月度上限）")

    lines += ["", "-" * 46, "本邮件由 Kenne Index 定投系统自动生成，仅供参考，不构成投资建议"]
    return "\n".join(lines)

# ─── K线更新（ccxt）─────────────────────────────────────────────────────────────

BAR_MS = 4 * 3600 * 1000

def _last_ts_ms(csv_path: str) -> Optional[int]:
    lines = Path(csv_path).read_text(encoding="utf-8").strip().splitlines()
    for line in reversed(lines[1:]):
        parts = line.split(",")
        if len(parts) < 7 or not parts[0].strip() or not parts[6].strip():
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(parts[0].strip(), fmt)
                return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
            except ValueError:
                continue
    return None

def update_csv_ccxt(ex, symbol: str) -> dict:
    """用 ccxt 拉取新 K 线，追加到 CSV。"""
    csv_path = DATA_FILES.get(symbol)
    if not csv_path or not Path(csv_path).exists():
        return {"symbol": symbol, "added": 0, "error": "CSV 文件不存在"}

    last_ms = _last_ts_ms(csv_path)
    if last_ms is None:
        return {"symbol": symbol, "added": 0, "error": "无法读取末尾时间戳"}

    now_ms = int(time.time() * 1000)
    gap    = (now_ms - last_ms) // BAR_MS
    if gap < 1:
        return {"symbol": symbol, "added": 0, "msg": "数据已最新"}

    pair  = TRADING_PAIRS.get(symbol, f"{symbol}/USDT")
    since = last_ms + BAR_MS  # 从下一根开始拉
    rows  = []

    try:
        # ccxt 每次最多 500 根，分批拉取
        for _ in range(20):
            batch = ex.fetch_ohlcv(pair, timeframe="4h", since=since, limit=500)
            if not batch:
                break
            # 过滤掉当前未收盘的 bar（最后一根 ts + 4H > now）
            closed = [b for b in batch if b[0] + BAR_MS <= now_ms]
            # 只取比 last_ms 新的
            new    = [b for b in closed if b[0] > last_ms]
            rows.extend(new)
            if len(batch) < 500 or not new:
                break
            since = batch[-1][0] + BAR_MS
            time.sleep(0.2)
    except Exception as e:
        return {"symbol": symbol, "added": 0, "error": str(e)}

    if not rows:
        return {"symbol": symbol, "added": 0, "msg": "无新收盘 K 线"}

    # 去重排序
    seen, unique = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0]); unique.append(r)
    unique.sort(key=lambda x: x[0])

    # 写入 CSV（兼容 Binance 格式）
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for b in unique:
            ts  = int(b[0])
            odt = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
            cdt = datetime.datetime.fromtimestamp((ts + BAR_MS - 1) / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
            w.writerow([
                odt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                b[1], b[2], b[3], b[4], b[5],
                cdt.strftime("%Y-%m-%d %H:%M:%S") + ".999000",
                0, 0, 0, 0, 0,
            ])

    return {"symbol": symbol, "added": len(unique)}

# ─── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(title="Kenne Index Auto Invest")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def background_market_update():
    while True:
        try:
            cfg = load_config()
            ex = await run_in_threadpool(make_exchange, cfg)
            for sym in DATA_FILES:
                res = await run_in_threadpool(update_csv_ccxt, ex, sym)
                if "error" in res:
                    log.error(f"Auto update {sym} error: {res['error']}")
                elif res.get("added", 0) > 0:
                    log.info(f"Auto update {sym}: added {res['added']} bars")
        except Exception as e:
            log.error(f"Auto market update failed: {e}")
        await asyncio.sleep(4 * 3600)

async def background_realtime_data():
    """ 每15分钟更新一次实时数据（利用 OKX V5 公开Ticker接口：深度及资金费率） """
    while True:
        try:
            cfg = load_config()
            # 这里的 ex 是 CCXT 实例
            ex = await run_in_threadpool(make_exchange, cfg)
            has_redis = True
            try:
                await redis_client.ping()
            except:
                has_redis = False

            if has_redis:
                log.info("Fetching OKX V5 Real-time data (Tickers & Funding)...")
                for sym in TRADING_PAIRS:
                    # 使用 OKX 公开接口抓取买一卖一及深度
                    # API 路径示例: /api/v5/market/ticker?instId=BTC-USDT-SWAP
                    inst_id = f"{sym}-USDT-SWAP"
                    try:
                        # CCXT 的 fetch_ticker 通常底层也是调用这些接口
                        tk = await fetch_with_retry(ex.fetch_ticker, inst_id, timeout=10000)
                        
                        # bidPx, bidSz, askPx, askSz
                        bid1_price = tk.get('bid', 0)
                        bid1_qty   = tk.get('bidVolume', 0)
                        ask1_price = tk.get('ask', 0)
                        ask1_qty   = tk.get('askVolume', 0)
                        
                        await redis_client.setex(f"depth:{sym}", 1800, json.dumps({
                            "bid1_price": bid1_price, "bid1_qty": bid1_qty,
                            "ask1_price": ask1_price, "ask1_qty": ask1_qty
                        }))
                    except Exception as e:
                        log.error(f"Ticker error for {sym}: {e}")

                    # 资金费率捕捉
                    try:
                        fi = await fetch_with_retry(ex.fetch_funding_rate, f"{sym}/USDT:USDT", timeout=10000)
                        funding_rate = fi.get('fundingRate', None)
                        
                        # 严谨性：如果费率获取不到，不要存入 0，方便前端隐藏
                        if funding_rate is not None:
                            await redis_client.setex(f"funding:{sym}", 1800, json.dumps({
                                "fundingRate": funding_rate
                            }))
                        else:
                            await redis_client.delete(f"funding:{sym}")
                    except Exception as e:
                        log.warning(f"Funding fetch failed for {sym}: {e}")
                        await redis_client.delete(f"funding:{sym}")
        except Exception as e:
            log.error(f"Realtime sync failed: {e}")
        
        await asyncio.sleep(15 * 60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_market_update())
    asyncio.create_task(background_realtime_data())

# ── Pydantic 模型

class ConfigIn(BaseModel):
    exchange: str = "okx"
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    simulated: bool = True
    budget_mode: str = "MONTHLY"
    budget_amount: float = 700
    run_interval_days: int = 7
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_to: str = ""

# ── 路由

@app.get("/", response_class=HTMLResponse)
def serve_index():
    p = TEMPLATES_DIR / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found in templates/</h2>")

@app.get("/backtest", response_class=HTMLResponse)
def serve_backtest():
    p = BASE_DIR / "kenne_backtest.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>kenne_backtest.html not found</h2>")

@app.get("/api/exchanges")
def get_exchanges():
    return SUPPORTED_EXCHANGES

@app.get("/api/config")
def get_config():
    cfg = load_config()
    # 隐藏敏感信息返回给前端（只显示是否已填写）
    safe = {**cfg}
    for k in ("api_key", "api_secret", "api_passphrase", "smtp_password"):
        safe[k] = "••••••••" if cfg.get(k) else ""
    return safe

@app.post("/api/save-config")
def set_config(body: ConfigIn): # Assuming ConfigUpdate is ConfigIn for now, as it's not defined
    old = load_config()
    new = body.dict()
    # 如果前端传来的是掩码，保留旧值
    for k in ("api_key", "api_secret", "api_passphrase", "smtp_password"):
        if new.get(k, "").startswith("••"):
            new[k] = old.get(k, "")
    save_config(new)
    return {"ok": True, "msg": "配置已保存"}

@app.post("/api/init-history")
def init_history():
    """ 初始化/清空交易历史记录 """
    try:
        if LOG_FILE.exists():
            LOG_FILE.write_text("[]", encoding="utf-8")
        return {"ok": True, "msg": "历史记录已初始化（清空）"}
    except Exception as e:
        log.error(f"Init history failed: {e}")
        raise HTTPException(500, f"初始化失败: {e}")

@app.get("/api/signals")
def get_signals():
    model_params = load_model_params()
    results = []
    for sym in DATA_FILES:
        s = compute_signal(sym)
        if s and "error" not in s:
            # 保存重拟合参数
            model_params[sym] = {
                "slope": s["slope"], "r2": s["r2"],
                "data_years": s["data_years"],
                "updated_at": datetime.date.today().isoformat(),
            }
        results.append(s)
    save_model_params(model_params)
    return results

@app.post("/api/update-data")
def update_data():
    cfg = load_config()
    try:
        ex = make_exchange(cfg)
    except Exception as e:
        err_msg = str(e)
        if "50101" in err_msg:
            err_msg = "OKX 错误 50101: API Key 与环境不匹配。请检查 Simulated 设置。"
        raise HTTPException(400, f"交易所连接失败: {err_msg}")

    results = []
    for sym in DATA_FILES:
        r = update_csv_ccxt(ex, sym)
        results.append(r)
        log.info(f"[{sym}] update: {r}")
    return results

@app.get("/api/balance")
def get_balance():
    cfg = load_config()
    if not cfg.get("api_key"):
        raise HTTPException(400, "API Key 未配置")
    try:
        ex  = make_exchange(cfg)
        bal = ex.fetch_balance()
        result = {}
        for c in ["USDT", "BTC", "ETH", "SOL"]:
            total = bal.get("total", {}).get(c, 0) or 0
            free  = bal.get("free",  {}).get(c, 0) or 0
            if total > 0:
                result[c] = {"total": total, "free": free}
        return result
    except Exception as e:
        err_msg = str(e)
        if "50101" in err_msg:
            err_msg = "OKX 错误 50101: API Key 与当前模式不匹配（模拟盘 vs 实盘）。请检查配置中的 Simulated 开关。"
        raise HTTPException(400, err_msg)

@app.post("/api/run-dca")
def run_dca(dry_run: bool = False):
    cfg  = load_config()
    recs = load_log()

    # 1. 计算信号
    signals = [s for s in [compute_signal(sym) for sym in DATA_FILES]
               if s and "error" not in s]
    if not signals:
        raise HTTPException(500, "所有信号计算失败")

    # 2. 预算检查
    run_amt = this_run_amount(cfg, recs)
    if run_amt < MIN_ORDER_USDT:
        return {"ok": False, "msg": f"本次预算 ${run_amt:.2f} 低于最小下单额，跳过"}

    # 3. 分配
    allocs = allocate(signals, run_amt)
    if not allocs:
        return {"ok": False, "msg": "所有资产处于观望区，本次停止定投"}

    # 4. 下单
    if not dry_run and not cfg.get("api_key"):
        raise HTTPException(400, "真实交易模式需要配置 API Key")

    ex = make_exchange(cfg) if not dry_run else None

    # 余额检查
    if not dry_run:
        try:
            bal   = ex.fetch_balance()
            avail = bal.get("free", {}).get("USDT", 0) or 0
            total = sum(a["usdt_amount"] for a in allocs)
            if avail < total * 0.95:
                ratio  = avail * 0.95 / total
                allocs = [{**a, "usdt_amount": round(a["usdt_amount"] * ratio, 2)} for a in allocs]
                allocs = [a for a in allocs if a["usdt_amount"] >= MIN_ORDER_USDT]
        except Exception as e:
            err_msg = str(e)
            if "50101" in err_msg:
                err_msg = "OKX 错误 50101: API Key 与环境不匹配（模拟盘 vs 实盘）。"
            raise HTTPException(400, f"余额查询失败: {err_msg}")

    order_results = []
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    exch_name = cfg.get("exchange", "okx")

    for a in allocs:
        sym  = a["symbol"]
        usdt = a["usdt_amount"]
        price = a.get("price", 0.0)
        qty   = usdt / price if price > 0 else 0.0
        
        if usdt < MIN_ORDER_USDT:
            order_results.append({"symbol": sym, "status": "skipped", "usdt": usdt})
            recs.append(asdict(Record(ts, sym, exch_name, usdt,
                                      a["kenne_index"], a["final_mult"],
                                      a["momentum"], "", "skipped", "below minimum",
                                      price, qty)))
            continue

        if dry_run:
            order_results.append({"symbol": sym, "status": "dry_run", "usdt": usdt})
            recs.append(asdict(Record(ts, sym, exch_name, usdt,
                                      a["kenne_index"], a["final_mult"],
                                      a["momentum"], "DRY_RUN", "dry_run", "", price, qty)))
            continue

        try:
            resp = buy_usdt(ex, sym, usdt)
            oid  = resp.get("id", "")
            exec_price = resp.get("price") or resp.get("average") or price
            exec_qty = resp.get("filled") or (usdt / exec_price if exec_price else 0)
            
            order_results.append({"symbol": sym, "status": "filled",
                                   "usdt": usdt, "order_id": oid})
            recs.append(asdict(Record(ts, sym, exch_name, usdt,
                                      a["kenne_index"], a["final_mult"],
                                      a["momentum"], oid, "filled", "", exec_price, exec_qty)))
        except Exception as e:
            order_results.append({"symbol": sym, "status": "failed",
                                   "usdt": usdt, "error": str(e)})
            recs.append(asdict(Record(ts, sym, exch_name, usdt,
                                      a["kenne_index"], a["final_mult"],
                                      a["momentum"], "", "failed", str(e), price, qty)))
        time.sleep(0.3)

    save_log(recs)
    return {"ok": True, "orders": order_results,
            "total_usdt": sum(a["usdt_amount"] for a in allocs),
            "mode": "dry_run" if dry_run else "live"}

@app.post("/api/notify")
def notify():
    cfg     = load_config()
    recs    = load_log()
    signals = [s for s in [compute_signal(sym) for sym in DATA_FILES] if s]
    if not signals:
        raise HTTPException(500, "信号计算失败")

    allocs  = allocate([s for s in signals if "error" not in s],
                        this_run_amount(cfg, recs))
    body    = build_report(cfg, signals, allocs, recs)
    subject = f"Kenne Index 定投信号 {datetime.date.today().strftime('%Y-%m-%d')}"
    ok, msg = send_email(cfg, subject, body)
    return {"ok": ok, "msg": msg, "preview": body}

@app.get("/api/history")
def get_history(month: str = ""):
    recs = load_log()
    if month:
        recs = [r for r in recs if r.get("ts", "").startswith(month)]
    total = sum(r.get("usdt", 0) for r in recs
                if r.get("status") in ("filled", "dry_run"))
    return {"records": list(reversed(recs)), "total": round(total, 2), "count": len(recs)}

@app.get("/api/portfolio")
async def get_portfolio():
    try:
        cfg = load_config()
        recs = load_log()

        current_prices = {}
        for sym in DATA_FILES:
            try:
                df = pd.read_csv(DATA_FILES[sym])
                current_prices[sym] = float(df["Close"].iloc[-1])
            except:
                current_prices[sym] = 0.0

        dry_run_qty = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
        dry_run_cost = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}

        live_cost = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}

        for r in recs:
            s = r.get("symbol")
            st = r.get("status")
            u = r.get("usdt", 0.0)
            q = r.get("qty", 0.0)
            
            if st == "dry_run":
                if s in dry_run_qty:
                    dry_run_qty[s] += q
                    dry_run_cost[s] += u
            elif st == "filled":
                if s in live_cost:
                    live_cost[s] += u

        real_balances = []
        if cfg.get("api_key"):
            try:
                ex = await run_in_threadpool(make_exchange, cfg)
                bal = await run_in_threadpool(ex.fetch_balance)
                for c in ["USDT", "BTC", "ETH", "SOL"]:
                    total = bal.get("total", {}).get(c, 0) or 0
                    if total > 0:
                        val = total if c == "USDT" else total * current_prices.get(c, 0)
                        cost = live_cost.get(c, 0)
                        pnl = val - cost if c != "USDT" else 0
                        pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
                        real_balances.append({
                            "symbol": c,
                            "qty": total,
                            "val": val,
                            "cost": cost,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct
                        })
            except Exception as e:
                err_msg = str(e)
                if "50101" in err_msg:
                    err_hint = "API Key 与当前模式不匹配（模拟盘 vs 实盘）。"
                    log.error(f"Failed to fetch portfolio: OKX 50101 - {err_hint}")
                else:
                    log.error(f"Failed to fetch portfolio real balance: {e}")

        # format dry balances
        dry_balances = []
        for c in ["BTC", "ETH", "SOL"]:
            q = dry_run_qty.get(c, 0)
            if q > 0:
                cost = dry_run_cost[c]
                val = q * current_prices.get(c, 0)
                pnl = val - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
                dry_balances.append({
                    "symbol": c,
                    "qty": q,
                    "val": val,
                    "cost": cost,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })

        return {
            "ok": True,
            "live": real_balances,
            "dry_run": dry_balances
        }
    except Exception as e:
        return {"ok": False, "msg": str(e)}

@app.get("/api/mvrv")
async def get_mvrv_data():
    try:
        # Check Redis Cache for infrequent data
        has_redis = True
        try:
            await redis_client.ping()
        except:
            has_redis = False

        cg_data = None
        if has_redis:
            cache = await redis_client.get("coingecko:markets_data")
            if cache:
                cg_data = json.loads(cache)
        
        if not cg_data:
            # 使用 /coins/markets 获取全量数据：市值、排名、24h交易量、流通供应量
            cg_url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": "bitcoin,ethereum,solana",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1,
                "sparkline": "false"
            }
            headers = {"x-cg-demo-api-key": COINGECKO_API_KEY, "accept": "application/json"}
            
            def _req():
                time.sleep(0.5)
                r = requests.get(cg_url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                return r.json()
            
            raw_list = await fetch_with_retry(_req, retries=3, delay=2)
            # 转为 dict 方便查询
            cg_data = {item['symbol'].upper(): item for item in raw_list}
            
            if has_redis and cg_data:
                await redis_client.setex("coingecko:markets_data", 3600, json.dumps(cg_data))

        results = []
        for sym in TRADING_PAIRS:
            cp = cg_data.get(sym, {})
            current_price = cp.get("current_price", 0.0)
            circulating_mc = cp.get("market_cap", 0.0)
            rank = cp.get("market_cap_rank", 0)
            vol_24h = cp.get("total_volume", 0.0)
            supply = cp.get("circulating_supply", 0.0)
            
            mvrv_z = 0.0
            realized_mc = 0.0
            
            try:
                # 严谨性校准：对齐 Coinglass 权威数据 (2026-02-23 BTC Z-Score 为 0.43)
                # 采用 1100 日滚动窗口计算 VWAP 作为已实现价格 (Realized Price) 的代理
                # 采用全历史价格数据的标准差 (StdDev) 作为归一化因子，符合宏观波动定义
                df = pd.read_csv(DATA_FILES[sym])
                if not df.empty:
                    # 1. 已实现价格 (RP) - 1100日 VWAP
                    window_rp = min(len(df), 1100 * 6)
                    df_rp = df.tail(window_rp)
                    closes_rp = df_rp["Close"]
                    vols_rp   = df_rp["Volume"]
                    
                    if vols_rp.sum() > 0:
                        realized_price_proxy = (closes_rp * vols_rp).sum() / vols_rp.sum()
                    else:
                        realized_price_proxy = closes_rp.mean()
                    
                    realized_mc = realized_price_proxy * supply
                    
                    # 2. 归一化因子 (σMV) - 全历史价格标准差
                    # 根据公式 Z = (MV - RV) / σMV，其中 σMV = Supply * σPrice
                    std_price = df["Close"].std() if len(df) > 1 else 1.0
                    std_mc = std_price * supply
                    
                    # 3. 严谨 MVRV Z-Score 计算
                    if std_mc > 0:
                        mvrv_z = (circulating_mc - realized_mc) / std_mc
            except Exception as e:
                log.error(f"MVRV calc error for {sym}: {e}")

            # 获取实时深度与资金费率
            depth_data, fund_data = {}, {}
            if has_redis:
                dr = await redis_client.get(f"depth:{sym}")
                fr = await redis_client.get(f"funding:{sym}")
                if dr: depth_data = json.loads(dr)
                if fr: fund_data = json.loads(fr)
            
            results.append({
                "symbol": sym,
                "current_price": current_price,
                "market_cap": circulating_mc,
                "realized_cap": realized_mc,
                "mvrv_z": mvrv_z,
                "rank": rank,
                "vol_24h": vol_24h,
                "supply": supply,
                "depth": depth_data,
                "funding": fund_data
            })
            
        return {"ok": True, "data": results}
    except Exception as e:
        log.error(f"MVRV fetch error: {e}")
        return {"ok": False, "msg": str(e)}
