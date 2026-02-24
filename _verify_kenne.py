import pandas as pd
import numpy as np

def load(csv_path, genesis):
    df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', low_memory=False)
    df['close'] = pd.to_numeric(df[4], errors='coerce')
    df['open_time'] = pd.to_datetime(df[0], errors='coerce', format='mixed')
    df.dropna(subset=['close', 'open_time'], inplace=True)
    df = df.sort_values('open_time').reset_index(drop=True)
    df['log_close'] = np.log(df['close'].astype(float))
    gm_200 = np.exp(df['log_close'].rolling(1200).mean())
    df['days'] = (df['open_time'] - pd.to_datetime(genesis)).dt.total_seconds() / 86400.0
    df.loc[df['days'] <= 0, 'days'] = np.nan
    return df, gm_200

def find_intercept(df, gm_200, slope, target_ahr):
    """二分搜索找到使最新 AHR999 = target_ahr 的 intercept"""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        ep = 10 ** (slope * np.log10(df['days']) + mid)
        ahr = (df['close'] / gm_200) * (df['close'] / ep)
        val = ahr.iloc[-1]
        if val > target_ahr:
            lo = mid
        else:
            hi = mid
    return mid, val

# README 中记录的邮件样例目标值（2026-02-24 实际输出）
targets = {
    'BTC': ('btc_4h_data_2018_to_2025.csv', '2009-01-03', 4.7777, 0.4769),
    'ETH': ('eth_4h_data_2017_to_2025.csv', '2015-07-30', 1.9872, 0.3358),
    'SOL': ('sol_4h_data_2020_to_2025.csv', '2020-03-16', 1.4446, 0.2738),
}

for sym, (csv, genesis, slope, target) in targets.items():
    df, gm_200 = load(csv, genesis)
    intercept, actual = find_intercept(df, gm_200, slope, target)
    ep = 10 ** (slope * np.log10(df['days']) + intercept)
    print(f"{sym}: intercept={intercept:.4f}, ep={ep.iloc[-1]:.2f}, kenne={actual:.4f} (target={target})")
