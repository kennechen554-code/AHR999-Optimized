"""
AHR999 x OKX 自动定投 (Open Source Edition)
用法:
  python3 ahr999_dca.py --update       # 仅补全 K 线数据
  python3 ahr999_dca.py --dry-run      # 模拟完整流程（不下单）
  python3 ahr999_dca.py                # 补数据 + 真实下单
  python3 ahr999_dca.py --daemon       # 守护进程，按设定间隔执行
  python3 ahr999_dca.py --history      # 查看历史
  python3 ahr999_dca.py --history 2026-03

环境变量:
  OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE
  BUDGET_MODE       # MONTHLY (默认) 或 FIXED
  BUDGET_AMOUNT     # 月预算总额 或 单次固定金额
  RUN_INTERVAL_DAYS # 运行间隔天数 (默认 7)

crontab:
  0 9 * * 1  cd /path && python3 ahr999_dca.py >> dca.log 2>&1
"""

import os, sys, json, hmac, base64, hashlib, time, logging, argparse
import datetime, csv, smtplib, textwrap, calendar
from email.mime.text import MIMEText
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))
from ahr999 import analyze as ahr999_analyze


# ─── 配置 ─────────────────────────────────────────────────────────────────────

CFG = {
    'API_KEY':        os.environ.get('OKX_API_KEY',        'YOUR_API_KEY'),
    'API_SECRET':     os.environ.get('OKX_API_SECRET',     'YOUR_API_SECRET'),
    'API_PASSPHRASE': os.environ.get('OKX_API_PASSPHRASE', 'YOUR_PASSPHRASE'),
    'SIMULATED':      True,       # True=模拟盘  False=真实交易
    
    # 定投策略配置
    # MONTHLY: 按月预算自动分配 (BUDGET_AMOUNT 代表月总额)
    # FIXED:   每次运行固定金额 (BUDGET_AMOUNT 代表单次金额)
    'BUDGET_MODE':    os.environ.get('BUDGET_MODE', 'MONTHLY'),
    'BUDGET_AMOUNT':  float(os.environ.get('BUDGET_AMOUNT', '700.0')),
    'RUN_INTERVAL_DAYS': float(os.environ.get('RUN_INTERVAL_DAYS', '7')),

    'DATA_FILES': {
        'BTC': 'btc_4h_data_2018_to_2025.csv',
        'ETH': 'eth_4h_data_2017_to_2025.csv',
        'SOL': 'sol_4h_data_2020_to_2025.csv',
    },
    'INST_ID': {
        'BTC': 'BTC-USDT',
        'ETH': 'ETH-USDT',
        'SOL': 'SOL-USDT',
    },
    'MAX_WEIGHT':     {'BTC': 0.60, 'ETH': 0.50, 'SOL': 0.50},
    'MIN_ORDER_USDT': 5.0,
    'LOG_FILE':       'dca_log.json',

    # 邮件通知
    'SMTP_HOST':     os.environ.get('SMTP_HOST',     'smtp.gmail.com'),
    'SMTP_PORT':     int(os.environ.get('SMTP_PORT', '587')),
    'SMTP_USER':     os.environ.get('SMTP_USER',     'your@gmail.com'),
    'SMTP_PASSWORD': os.environ.get('SMTP_PASSWORD', 'your_app_password'),
    'EMAIL_TO':      os.environ.get('EMAIL_TO',      'your@gmail.com'),
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('dca')


# ─── OKX 客户端 ───────────────────────────────────────────────────────────────

class OKXClient:
    BASE = 'https://www.okx.com'

    def __init__(self, key, secret, passphrase, simulated):
        self.key, self.secret, self.passphrase = key, secret, passphrase
        self.simulated = simulated
        self.sess = requests.Session()
        self.sess.headers['Content-Type'] = 'application/json'

    @staticmethod
    def _ts():
        n = datetime.datetime.utcnow()
        return n.strftime('%Y-%m-%dT%H:%M:%S.') + f'{n.microsecond//1000:03d}Z'

    def _sign(self, ts, method, path, body=''):
        return base64.b64encode(
            hmac.new(self.secret.encode(), (ts + method + path + body).encode(),
                     hashlib.sha256).digest()
        ).decode()

    def _auth(self, method, path, body=''):
        ts = self._ts()
        h  = {'OK-ACCESS-KEY': self.key, 'OK-ACCESS-SIGN': self._sign(ts, method, path, body),
               'OK-ACCESS-TIMESTAMP': ts, 'OK-ACCESS-PASSPHRASE': self.passphrase}
        if self.simulated:
            h['x-simulated-trading'] = '1'
        return h

    def _get(self, path, params=None, auth=False):
        qs   = ('?' + '&'.join(f'{k}={v}' for k, v in params.items())) if params else ''
        full = path + qs
        kw   = {'headers': self._auth('GET', full)} if auth else {}
        r    = self.sess.get(self.BASE + full, timeout=15, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body):
        s = json.dumps(body)
        r = self.sess.post(self.BASE + path, headers=self._auth('POST', path, s),
                           data=s, timeout=15)
        r.raise_for_status()
        return r.json()

    def candles(self, inst_id, bar='4H', before=None, after=None, limit=100):
        p = {'instId': inst_id, 'bar': bar, 'limit': str(limit)}
        if before is not None: p['before'] = str(before)
        if after  is not None: p['after']  = str(after)
        try:
            resp = self._get('/api/v5/market/history-candles', p)
        except Exception:
            resp = self._get('/api/v5/market/candles', p)
        if resp.get('code') != '0':
            raise RuntimeError(f"candles error: {resp.get('msg', resp)}")
        return resp.get('data', [])

    def balance(self, ccy='USDT'):
        resp = self._get('/api/v5/account/balance', {'ccy': ccy}, auth=True)
        if resp.get('code') != '0':
            raise RuntimeError(f"balance error: {resp}")
        for d in resp['data'][0]['details']:
            if d['ccy'] == ccy:
                return float(d['availBal'])
        return 0.0

    def buy_market_usdt(self, inst_id, usdt):
        return self._post('/api/v5/trade/order', {
            'instId': inst_id, 'tdMode': 'cash', 'side': 'buy',
            'ordType': 'market', 'sz': f'{usdt:.4f}', 'tgtCcy': 'quote_ccy',
        })


# ─── 数据更新 ─────────────────────────────────────────────────────────────────

BAR_MS = 4 * 3600 * 1000  # 4H in ms


def _candle_to_row(c):
    ts   = int(c[0])
    odt  = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
    cdt  = datetime.datetime.fromtimestamp((ts + BAR_MS - 1) / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
    return [
        odt.strftime('%Y-%m-%d %H:%M:%S.%f'),
        c[1], c[2], c[3], c[4], c[5],
        cdt.strftime('%Y-%m-%d %H:%M:%S') + '.999000',
        c[7], '0', '0', '0', '0',
    ]


class DataUpdater:
    def __init__(self, client):
        self.client = client

    def _last_ts_ms(self, path):
        lines = path.read_text(encoding='utf-8').strip().splitlines()
        if len(lines) < 2: return None
        for line in reversed(lines):
            parts = line.split(',')
            if len(parts) < 7: continue
            ot_str = parts[0].strip()
            ct_str = parts[6].strip()
            if ot_str and ct_str:
                try:
                    dt = datetime.datetime.strptime(ot_str, '%Y-%m-%d %H:%M:%S.%f')
                    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
                except ValueError:
                    try:
                        dt = datetime.datetime.strptime(ot_str, '%Y-%m-%d %H:%M:%S')
                        return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
                    except ValueError:
                        continue
        return None

    def _clean_tail(self, path):
        lines   = path.read_text(encoding='utf-8').splitlines()
        cleaned = [lines[0]] + [
            l for l in lines[1:]
            if len(l.split(',')) > 6
            and l.split(',')[0].strip()
            and l.split(',')[6].strip()
        ]
        removed = len(lines) - len(cleaned)
        if removed:
            path.write_text('\n'.join(cleaned) + '\n', encoding='utf-8')
        return removed

    def _fetch_after(self, inst_id, since_ms):
        collected, cursor = [], since_ms
        for _ in range(20):
            try:
                batch = self.client.candles(inst_id, bar='4H', before=cursor, limit=100)
            except Exception as e:
                log.error(f'candles fetch error: {e}')
                break
            if not batch: break
            closed = [c for c in batch if c[8] == '1']
            if closed: collected.extend(closed)
            newest = int(batch[0][0])
            if newest <= cursor: break
            cursor = newest
            if len(batch) < 100: break
            time.sleep(0.15)

        seen, unique = set(), []
        for c in collected:
            if c[0] not in seen:
                seen.add(c[0])
                unique.append(c)
        unique.sort(key=lambda c: int(c[0]))
        return unique

    def update(self, symbol, csv_file):
        path = Path(csv_file)
        if not path.exists():
            log.warning(f'[{symbol}] CSV not found: {csv_file}')
            return 0

        removed = self._clean_tail(path)
        if removed: log.info(f'[{symbol}] removed {removed} incomplete tail rows')

        last_ms = self._last_ts_ms(path)
        if last_ms is None:
            log.warning(f'[{symbol}] no valid data in CSV')
            return 0

        gap = (int(time.time() * 1000) - last_ms) // BAR_MS
        last_dt = datetime.datetime.fromtimestamp(last_ms / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
        log.info(f'[{symbol}] local last: {last_dt.strftime("%Y-%m-%d %H:%M")} UTC  gap: ~{gap} bars')

        if gap < 1:
            log.info(f'[{symbol}] up to date')
            return 0

        new = [c for c in self._fetch_after(CFG['INST_ID'][symbol], since_ms=last_ms)
               if int(c[0]) > last_ms]

        if not new:
            log.info(f'[{symbol}] current bar not yet closed, no new data')
            return 0

        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(_candle_to_row(c) for c in new)

        t0 = datetime.datetime.fromtimestamp(int(new[0][0])  / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
        t1 = datetime.datetime.fromtimestamp(int(new[-1][0]) / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
        log.info(f'[{symbol}] +{len(new)} bars  '
                 f'{t0.strftime("%Y-%m-%d %H:%M")} -> {t1.strftime("%Y-%m-%d %H:%M")} UTC')
        return len(new)


# ─── 预算追踪 ─────────────────────────────────────────────────────────────────

@dataclass
class Record:
    ts:       str
    symbol:   str
    inst_id:  str
    usdt:     float
    ahr999:   float
    mult:     float
    momentum: str
    order_id: str
    status:   str    # filled | dry_run | skipped | failed
    note:     str = ''


class Budget:
    def __init__(self):
        self.path   = Path(CFG['LOG_FILE'])
        self.amount = CFG['BUDGET_AMOUNT']
        self.mode   = CFG['BUDGET_MODE']
        self.recs   = json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self):
        self.path.write_text(json.dumps(self.recs, indent=2, ensure_ascii=False))

    def spent_this_month(self):
        m = datetime.date.today().strftime('%Y-%m')
        return sum(r['usdt'] for r in self.recs
                   if r.get('ts', '').startswith(m)
                   and r.get('status') in ('filled', 'dry_run'))

    def remaining_monthly(self):
        if self.mode != 'MONTHLY':
            return self.amount
        return max(0.0, self.amount - self.spent_this_month())

    def get_run_budget(self):
        """计算本次运行的可用基础金额"""
        if self.mode == 'FIXED':
            return self.amount

        # MONTHLY Logic
        today = datetime.date.today()
        _, days_in_month = calendar.monthrange(today.year, today.month)
        days_remaining   = days_in_month - today.day
        
        interval  = CFG['RUN_INTERVAL_DAYS']
        runs_left = max(1, int(days_remaining / interval) + 1)
        
        rem = self.remaining_monthly()
        return rem / runs_left if rem > 0 else 0.0

    def add(self, r):
        self.recs.append(asdict(r))
        self._save()


# ─── 资金分配 ─────────────────────────────────────────────────────────────────

def allocate(signals, budget):
    """按 final_mult 权重分配，应用各币种权重上限后归一化。"""
    active = [s for s in signals if s['final_mult'] > 0]
    if not active or budget <= 0:
        return []
    norm = {s['symbol']: s['final_mult'] for s in active}
    for _ in range(3):
        total = sum(norm.values())
        norm  = {k: v / total for k, v in norm.items()}
        cap   = {k: min(v, CFG['MAX_WEIGHT'].get(k, 1.0)) for k, v in norm.items()}
        if sum(cap.values()) == 0:
            break
        norm = cap
    total = sum(norm.values())
    norm  = {k: v / total for k, v in norm.items()}
    return [{**s,
             'usdt_amount': round(norm.get(s['symbol'], 0) * budget, 2),
             'weight':      round(norm.get(s['symbol'], 0), 4)}
            for s in active]


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def _make_client():
    return OKXClient(CFG['API_KEY'], CFG['API_SECRET'],
                     CFG['API_PASSPHRASE'], CFG['SIMULATED'])


def run_update():
    updater = DataUpdater(_make_client())
    total   = 0
    for sym, f in CFG['DATA_FILES'].items():
        try:
            total += updater.update(sym, f)
        except Exception as e:
            log.error(f'[{sym}] update failed: {e}')
    log.info(f'update done, {total} new bars total')


def run_dca(dry_run=False):
    log.info(f'--- AHR999 DCA {"[dry-run]" if dry_run else "[live]"} ---')
    log.info(f'Mode: {CFG["BUDGET_MODE"]}  Amount: ${CFG["BUDGET_AMOUNT"]:.2f}  Interval: {CFG["RUN_INTERVAL_DAYS"]}d')

    client  = _make_client()
    updater = DataUpdater(client)
    budget  = Budget()

    # 1. 补全 K 线
    log.info('[1/4] updating market data')
    total_new = 0
    for sym, f in CFG['DATA_FILES'].items():
        try:
            total_new += updater.update(sym, f)
        except Exception as e:
            log.error(f'[{sym}] update failed, using local data: {e}')
    log.info(f'{total_new} new bars added' if total_new else 'data up to date')

    # 2. 预算检查
    log.info('[2/4] budget check')
    run_amt = budget.get_run_budget()
    
    if CFG['BUDGET_MODE'] == 'MONTHLY':
        log.info(f'monthly=${CFG["BUDGET_AMOUNT"]:.0f}  spent=${budget.spent_this_month():.2f}'
                 f'  remaining=${budget.remaining_monthly():.2f}')
    
    log.info(f'this_run_base=${run_amt:.2f}')
    if run_amt < CFG['MIN_ORDER_USDT']:
        log.info('budget below minimum, skipping')
        return

    # 3. 信号计算
    log.info('[3/4] calculating signals')
    signals = []
    for sym, f in CFG['DATA_FILES'].items():
        try:
            r = ahr999_analyze(f, sym)
            if r:
                signals.append(r)
                log.info(f'  {sym}: ahr999={r["ahr999"]:.4f}  '
                         f'momentum={r["momentum"]}  mult={r["final_mult"]:.2f}x')
        except Exception as e:
            log.error(f'[{sym}] analysis failed: {e}')
    if not signals:
        log.error('all analysis failed, abort')
        return

    # 4. 分配并下单
    log.info('[4/4] allocating and ordering')
    allocs = allocate(signals, run_amt)
    if not allocs:
        log.info('all assets in hold zone, no orders this run')
        return

    for a in allocs:
        log.info(f'  plan: {a["symbol"]} ${a["usdt_amount"]:.2f}'
                 f' weight={a["weight"]:.1%} mult={a["final_mult"]:.2f}x')
    log.info(f'  total: ${sum(a["usdt_amount"] for a in allocs):.2f}')

    if not dry_run:
        try:
            avail = client.balance('USDT')
            log.info(f'okx USDT balance: ${avail:.2f}')
            total = sum(a['usdt_amount'] for a in allocs)
            if avail < total * 0.95:
                ratio  = avail * 0.95 / total
                allocs = [{**a, 'usdt_amount': round(a['usdt_amount'] * ratio, 2)}
                          for a in allocs]
                allocs = [a for a in allocs if a['usdt_amount'] >= CFG['MIN_ORDER_USDT']]
                log.warning(f'insufficient balance, scaled to ${sum(a["usdt_amount"] for a in allocs):.2f}')
        except Exception as e:
            log.error(f'balance check failed: {e}')
            if not CFG['SIMULATED']:
                return

    for a in allocs:
        sym     = a['symbol']
        inst_id = CFG['INST_ID'][sym]
        usdt    = a['usdt_amount']
        ts      = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        if usdt < CFG['MIN_ORDER_USDT']:
            log.info(f'  {sym}: ${usdt:.2f} below minimum, skip')
            budget.add(Record(ts, sym, inst_id, usdt, a['ahr999'],
                              a['final_mult'], a['momentum'], '', 'skipped', 'below minimum'))
            continue

        if dry_run:
            log.info(f'  {sym}: [dry-run] would buy ${usdt:.2f} USDT')
            budget.add(Record(ts, sym, inst_id, usdt, a['ahr999'],
                              a['final_mult'], a['momentum'], 'DRY_RUN', 'dry_run'))
            continue

        try:
            resp = client.buy_market_usdt(inst_id, usdt)
            if resp.get('code') == '0':
                oid = resp['data'][0]['ordId']
                log.info(f'  {sym}: filled  order={oid}  ${usdt:.2f} USDT')
                budget.add(Record(ts, sym, inst_id, usdt, a['ahr999'],
                                  a['final_mult'], a['momentum'], oid, 'filled'))
            else:
                err = resp.get('data', [{}])[0].get('sMsg', str(resp))
                log.error(f'  {sym}: order failed: {err}')
                budget.add(Record(ts, sym, inst_id, usdt, a['ahr999'],
                                  a['final_mult'], a['momentum'], '', 'failed', err))
        except Exception as e:
            log.error(f'  {sym}: exception: {e}')
            budget.add(Record(ts, sym, inst_id, usdt, a['ahr999'],
                              a['final_mult'], a['momentum'], '', 'failed', str(e)))
        time.sleep(0.3)

    log.info('--- done ---')


# ─── 守护进程 & 历史 ──────────────────────────────────────────────────────────

def _get_next_run_time():
    """根据配置的间隔天数计算下次运行时间"""
    now = datetime.datetime.now()
    interval = CFG['RUN_INTERVAL_DAYS']
    # 简单增加间隔天数
    next_run = now + datetime.timedelta(days=interval)
    # 重置秒数，保持整洁
    return next_run.replace(second=0, microsecond=0)


def run_daemon(dry_run=False):
    interval = CFG['RUN_INTERVAL_DAYS']
    log.info(f'daemon started, runs every {interval} days (Ctrl+C to stop)')
    
    while True:
        t    = _get_next_run_time()
        secs = (t - datetime.datetime.now()).total_seconds()
        log.info(f'next run: {t.strftime("%Y-%m-%d %H:%M")} ({secs/3600:.1f}h from now)')
        time.sleep(max(1, secs))
        try:
            run_dca(dry_run=dry_run)
        except Exception as e:
            log.error(f'run_dca exception: {e}')
        time.sleep(60)


def show_history(month=None):
    recs = json.loads(Path(CFG['LOG_FILE']).read_text()) \
           if Path(CFG['LOG_FILE']).exists() else []
    if month:
        recs = [r for r in recs if r.get('ts', '').startswith(month)]
    if not recs:
        print('no records')
        return

    status_map = {'filled': 'OK', 'dry_run': 'DRY', 'failed': 'ERR', 'skipped': 'SKP'}
    total = 0.0
    print(f'\n{"date":<20} {"sym":<4} {"usdt":>8} {"ahr999":>8} {"mult":>6} status')
    print('-' * 58)
    for r in recs:
        u = r.get('usdt', 0)
        if r['status'] in ('filled', 'dry_run'):
            total += u
        st = status_map.get(r['status'], '???')
        print(f'  {r.get("ts",""):<18} {r["symbol"]:<4} '
              f'{u:>8.2f} {r.get("ahr999",0):>8.4f} '
              f'{r.get("mult",0):>5.2f}x  {st}')
    print(f'{"-"*58}\n  total: ${total:.2f}')


# ─── 邮件通知 ─────────────────────────────────────────────────────────────────

def _send_email(subject, body):
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From']    = CFG['SMTP_USER']
    msg['To']      = CFG['EMAIL_TO']

    port = CFG['SMTP_PORT']
    host = CFG['SMTP_HOST']
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                s.login(CFG['SMTP_USER'], CFG['SMTP_PASSWORD'])
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.ehlo()
                s.starttls()
                s.login(CFG['SMTP_USER'], CFG['SMTP_PASSWORD'])
                s.send_message(msg)
        log.info(f'email sent -> {CFG["EMAIL_TO"]}')
    except Exception as e:
        log.error(f'email failed: {e}')


def _build_report(signals, allocs, budget):
    date  = datetime.date.today().strftime('%Y-%m-%d')
    lines = [
        f'AHR999 定投信号  {date}',
        '=' * 44,
        '',
        '当前信号',
        '-' * 44,
    ]

    for s in signals:
        ahr  = s['ahr999']
        mult = s['final_mult']
        mom  = s['momentum']
        prc  = s.get('price', 0)

        if   ahr < 0.45:       zone = '极低估'
        elif s['base_mult'] > 0: zone = '定投区'
        else:                   zone = '观望区'

        lines.append(
            f"  {s['symbol']:<4}  价格 {prc:>10,.2f} USDT  "
            f"AHR999 {ahr:.4f}  {zone}  {mom}  建议 {mult:.2f}x"
        )
    
    label = '本期分配' if CFG['BUDGET_MODE'] == 'FIXED' else f'本周分配（{CFG["BUDGET_AMOUNT"]:.0f}/月）'
    lines += ['', label, '-' * 44]

    if allocs:
        for a in allocs:
            lines.append(
                f"  {a['symbol']:<4}  ${a['usdt_amount']:>7.2f} USDT"
                f"  权重 {a['weight']:.0%}  倍数 {a['final_mult']:.2f}x"
            )
        lines.append(f"  {'合计':<4}  ${sum(a['usdt_amount'] for a in allocs):>7.2f} USDT")
    else:
        lines.append('  当前无买入信号，本期停止定投')

    if CFG['BUDGET_MODE'] == 'MONTHLY':
        lines += [
            '',
            f'月预算 ${budget.amount:.0f}  已花 ${budget.spent_this_month():.2f}'
            f'  剩余 ${budget.remaining_monthly():.2f}',
        ]

    lines += [
        '',
        '-' * 44,
        '本邮件由 AHR999 定投系统自动生成，仅供参考',
    ]
    return '\n'.join(lines)


def run_notify():
    log.info('--- AHR999 notify ---')

    client  = _make_client()
    updater = DataUpdater(client)
    budget  = Budget()

    # 补全数据
    log.info('[1/3] updating market data')
    for sym, f in CFG['DATA_FILES'].items():
        try:
            updater.update(sym, f)
        except Exception as e:
            log.error(f'[{sym}] update failed, using local data: {e}')

    # 计算信号
    log.info('[2/3] calculating signals')
    signals = []
    for sym, f in CFG['DATA_FILES'].items():
        try:
            r = ahr999_analyze(f, sym)
            if r:
                signals.append(r)
        except Exception as e:
            log.error(f'[{sym}] analysis failed: {e}')

    if not signals:
        log.error('all analysis failed')
        return

    # 计算分配
    run_amt = budget.get_run_budget()
    allocs  = allocate(signals, run_amt)

    # 打印到终端
    for s in signals:
        log.info(f"  {s['symbol']}: ahr999={s['ahr999']:.4f}  "
                 f"momentum={s['momentum']}  mult={s['final_mult']:.2f}x")

    # 发送邮件
    log.info('[3/3] sending email')
    body    = _build_report(signals, allocs, budget)
    subject = f"AHR999 定投信号 {datetime.date.today().strftime('%Y-%m-%d')}"
    _send_email(subject, body)
    log.info('--- done ---')


def run_notify_daemon():
    log.info('notify daemon started (Ctrl+C to stop)')
    while True:
        t    = _get_next_run_time()
        secs = (t - datetime.datetime.now()).total_seconds()
        log.info(f'next notify: {t.strftime("%Y-%m-%d %H:%M")} ({secs/3600:.1f}h from now)')
        time.sleep(max(1, secs))
        try:
            run_notify()
        except Exception as e:
            log.error(f'run_notify exception: {e}')
        time.sleep(60)


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='AHR999 x OKX auto DCA')
    p.add_argument('--update',  action='store_true', help='update CSV data only')
    p.add_argument('--notify',  action='store_true', help='send signal email (no trade)')
    p.add_argument('--dry-run', action='store_true', help='simulate without ordering')
    p.add_argument('--daemon',  action='store_true', help='run as daemon')
    p.add_argument('--notify-daemon', action='store_true', help='run notify as daemon')
    p.add_argument('--history', nargs='?', const='', metavar='YYYY-MM', help='show history')
    a = p.parse_args()

    if   a.history is not None:    show_history(month=a.history or None)
    elif a.update:                 run_update()
    elif a.notify:                 run_notify()
    elif a.notify_daemon:          run_notify_daemon()
    elif a.daemon:                 run_daemon(dry_run=a.dry_run)
    else:                          run_dca(dry_run=a.dry_run)


if __name__ == '__main__':
    main()