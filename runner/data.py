"""Dated, validated market data. Adjusted signals and unadjusted fills stay separate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import requests
import numpy as np
import pandas as pd
import exchange_calendars as xc

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)


def calendar():
    return xc.get_calendar('XSHG', start='2024-01-01')


def sessions(a, b):
    return [str(x.date()) for x in calendar().sessions_in_range(a, b)]


def next_session(day):
    return str(calendar().next_session(day).date())


def asof_now():
    now = pd.Timestamp.now(tz='Asia/Shanghai')
    day = now.normalize().tz_localize(None)
    if now.hour * 60 + now.minute < 15 * 60 + 10:
        day -= pd.Timedelta(days=1)
    return str(calendar().date_to_session(day, direction='previous').date())


def get_json(url, params=None):
    err = None
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=25,
                             headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err = e
            time.sleep(i + 0.3)
    raise RuntimeError(f'{url}: {err}')


def history(code, day, adjusted=True, index=False, start='2024-01-01'):
    folder = CACHE / ('index' if index else ('qfq' if adjusted else 'raw')) / start
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f'{code}.parquet'
    if p.exists():
        old = pd.read_parquet(p)
        old = old[old.date <= pd.Timestamp(day)]
        if len(old) and str(old.date.max().date()) == day:
            return old
    market = '1' if index or code.startswith(('5', '6', '9')) else '0'
    j = get_json('https://push2his.eastmoney.com/api/qt/stock/kline/get', {
        'secid': f'{market}.{code}', 'klt': 101, 'fqt': 1 if adjusted and not index else 0,
        'beg': start.replace('-', ''), 'end': day.replace('-', ''),
        'fields1': 'f1,f2,f3', 'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'})
    d = j.get('data') or {}
    rows = [x.split(',') for x in d.get('klines', [])]
    if not rows:
        raise RuntimeError(f'{code}: no history through {day}')
    df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume',
                                     'amount', 'amplitude', 'pct', 'change', 'turnover'])
    df['date'] = pd.to_datetime(df.date)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['code'], df['name'] = code, d.get('name', code)
    if df.date.duplicated().any() or not np.isfinite(df.close).all() or (df.close <= 0).any():
        raise RuntimeError(f'{code}: malformed history')
    df.to_parquet(p, index=False)
    return df


def histories(codes, day, adjusted=True, start='2024-01-01', workers=6):
    frames, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        todo = {ex.submit(history, c, day, adjusted, False, start): c for c in sorted(set(codes))}
        for i, f in enumerate(as_completed(todo), 1):
            c = todo[f]
            try:
                frames[c] = f.result()
            except Exception as e:
                errors[c] = str(e)
            if i % 200 == 0:
                print(f'data {i}/{len(todo)}, failures={len(errors)}', flush=True)
    return frames, errors


def stock_spot(day):
    import akshare as ak
    p = CACHE / f'spot_{day}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    spot = ak.stock_zh_a_spot_em()
    spot['代码'] = spot['代码'].astype(str).str.zfill(6)
    spot = spot[~spot['名称'].str.contains('ST|退', case=False, na=False)]
    spot = spot[spot['代码'].str[:2].isin(['60', '00', '30', '68']) & (spot['最新价'] > 2)]
    spot = spot.drop_duplicates('代码').sort_values('总市值', ascending=False)
    if len(spot) < 1600:
        raise RuntimeError(f'incomplete stock snapshot: {len(spot)}')
    spot['code6'] = spot['代码']
    # Snapshot only usable for the most recently completed session; never backdate live caps.
    if day != asof_now():
        raise RuntimeError('cannot backdate live market capitalization snapshot')
    spot.to_parquet(p, index=False)
    return spot


def board_mapping(day):
    import akshare as ak
    folder = CACHE / 'boards' / day
    folder.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for kind, listing, members in [('ind', ak.stock_board_industry_name_em, ak.stock_board_industry_cons_em),
                                    ('con', ak.stock_board_concept_name_em, ak.stock_board_concept_cons_em)]:
        names = listing()['板块名称'].tolist()
        def one(name):
            import hashlib
            p = folder / f'{kind}_{hashlib.sha256(name.encode()).hexdigest()[:20]}.parquet'
            if p.exists():
                return pd.read_parquet(p)
            for attempt in range(3):
                try:
                    d = members(symbol=name)
                    d = pd.DataFrame({'code': d['代码'].astype(str).str.zfill(6), 'board': name, 'kind': kind})
                    if d.empty:
                        raise ValueError('empty membership')
                    d.to_parquet(p, index=False)
                    return d
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        with ThreadPoolExecutor(max_workers=5) as ex:
            fs = {ex.submit(one, n): n for n in names}
            for f in as_completed(fs):
                try:
                    rows.append(f.result())
                except Exception:
                    failed.append(fs[f])
    if failed:
        raise RuntimeError(f'incomplete board membership: {len(failed)} failures; {failed[:5]}')
    return pd.concat(rows, ignore_index=True).drop_duplicates(['code', 'board', 'kind'])


def execution_bars(codes, day):
    frames, errors = histories(codes, day, adjusted=False,
                               start=str((pd.Timestamp(day) - pd.Timedelta(days=40)).date()))
    bars = {}
    for c, f in frames.items():
        row = f[f.date == pd.Timestamp(day)]
        if row.empty:
            continue
        row = row.iloc[-1]
        # pct is exchange-reported against the ex-rights reference, usable for limit checks.
        prev = float(row.close) / (1 + float(row.pct) / 100)
        bars[c] = {k: float(row[k]) for k in ['open', 'close', 'high', 'low', 'volume', 'amount']}
        bars[c]['prev'] = prev
        past = f[f.date < pd.Timestamp(day)]
        bars[c]['previous_raw_close'] = float(past.iloc[-1].close) if len(past) else prev
    return bars, errors
