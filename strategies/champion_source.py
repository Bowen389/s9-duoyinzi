#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冠军策略 · 独立可执行实现（单文件，无外部仓库依赖）
=================================================
基于 output/策略规格_5版.json 的冠军配置：
  持有3日 Top20 收盘卖 · HS300>MA120 门控 · 窗口 量比10/换手40/前高120
  题材权重 mom5 −0.5 / zt −0.5 / rank −0.5 / rs 2.0 / prem −0.5
  数据区间 2024-01-01 ~ 2026-11-05 · 复利口径 · HS300 基准

用法：
  python3 champion_standalone.py                       # 完整回测（默认 champion）
  python3 champion_standalone.py --version runner      # 指定版本（5版任选）
  python3 champion_standalone.py --mode signal --date 2026-11-05   # 查询某日收盘候选
  python3 champion_standalone.py --end 20261105        # 指定数据截止日
  python3 champion_standalone.py --refresh             # 强制重新拉取数据

输出（--out 目录，默认 ./standalone_out）：
  trades_{version}.csv  逐笔交易（含名称/买卖价/收益）
  nav_{version}.csv     日净值 + HS300
  monthly_{version}.csv 月度收益
  summary_{version}.json 指标（夏普/年化/回撤/累计/胜率/盈亏比/笔数）

数据缓存：./cache/（spot、universe、逐股日线、指数、板块成分），可断点续跑。
依赖：akshare、pandas、numpy、pyarrow（缺失时自动安装）。
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import pyarrow  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyarrow==26.0.1'])

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, 'cache')

# ---------------- 策略参数（与策略规格一致） ----------------
START = '20240101'
END_DEFAULT = '20261105'
N_UNIVERSE = 800

BASE_WEIGHTS = {'vol_ratio': 1.695, 'turnover': 1.687, 'open_str': 1.375,
                'zt_count': 1.233, 'position': 1.205, 'amplitude': -0.598,
                'mktcap': -0.839}
BASE_NEW = {'theme_mom5': 1.0, 'theme_zt': 1.0, 'theme_rank': 0.5,
            'theme_rs': 0.5, 'zt_premium': 0.5}
FACTORS = ['theme_mom5', 'theme_zt', 'theme_rank', 'theme_rs', 'zt_premium']

VERSIONS = {
    'champion': {'hold': 3, 'top_n': 20, 'sell_at': 'close', 'gate': 'ma120',
                 'windows': {'vol_ma': 10, 'to_window': 40, 'pos_window': 120},
                 'weights': {'theme_mom5': -0.5, 'theme_zt': -0.5,
                             'theme_rank': -0.5, 'theme_rs': 2.0,
                             'zt_premium': -0.5},
                 'label': '冠军配置'},
    'balanced': {'hold': 3, 'top_n': 20, 'sell_at': 'close', 'gate': 'ma20',
                 'windows': {'vol_ma': 5, 'to_window': 20, 'pos_window': 120},
                 'weights': {'theme_mom5': 1.0, 'theme_zt': 0.0,
                             'theme_rank': -0.5, 'theme_rs': 0.5,
                             'zt_premium': -0.5},
                 'label': '稳健配置'},
    'runner': {'hold': 3, 'top_n': 20, 'sell_at': 'close', 'gate': 'ma60',
               'windows': {'vol_ma': 5, 'to_window': 20, 'pos_window': 120},
               'weights': {'theme_mom5': 1.0, 'theme_zt': -0.5,
                           'theme_rank': 0.0, 'theme_rs': 2.0,
                           'zt_premium': -0.5},
               'label': '奔跑配置'},
    'original': {'hold': 3, 'top_n': 20, 'sell_at': 'open', 'gate': 'ma20',
                 'windows': {'vol_ma': 5, 'to_window': 20, 'pos_window': 120},
                 'weights': {'theme_mom5': 1.0, 'theme_zt': 0.0,
                             'theme_rank': -0.5, 'theme_rs': 0.5,
                             'zt_premium': -0.5},
                 'label': '原版配置'},
    'wild': {'hold': 1, 'top_n': 2, 'sell_at': 'close', 'gate': 'ma120',
             'windows': {'vol_ma': 3, 'to_window': 20, 'pos_window': 60},
             'weights': {'theme_mom5': 1.0, 'theme_zt': 1.0,
                         'theme_rank': -0.5, 'theme_rs': 0.5,
                         'zt_premium': -0.5},
             'label': '狂暴配置'},
}

# 成本
COMMISSION = 0.00025     # 佣金 双边
STAMP_TAX = 0.0005       # 印花税 卖出
TRANSFER_FEE = 0.00001   # 过户费 双边
SLIPPAGE = 0.0015        # 滑点 单边
BUY_COST = COMMISSION + TRANSFER_FEE + SLIPPAGE
SELL_COST = COMMISSION + TRANSFER_FEE + STAMP_TAX + SLIPPAGE

LIMIT_PCT = {'60': 9.8, '00': 9.8, '30': 19.8, '68': 19.8}
TO_BREAKS = [1.0, 3.0, 7.0]


# ================= 数据获取（带缓存） =================
def ensure_akshare():
    try:
        import akshare  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'akshare'])


def load_spot():
    """全A快照 -> 股票池（大盘前800 + 小盘后800）"""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, 'spot.csv')
    if os.path.exists(fp):
        spot = pd.read_csv(fp, dtype={'代码': str})
    else:
        import akshare as ak
        spot = ak.stock_zh_a_spot_em()
        spot['代码'] = spot['代码'].astype(str).str.zfill(6)
        spot.to_csv(fp, index=False)
    spot['code6'] = spot['代码'].str.zfill(6)
    spot = spot[~spot['名称'].str.contains('ST|退', na=False, regex=True)]
    spot = spot[spot['最新价'] > 2]
    spot = spot[spot['code6'].str[:2].isin(['60', '00', '30', '68'])]
    spot = spot.drop_duplicates('code6')
    spot['总市值'] = spot['总市值'].astype(float)
    spot = spot.sort_values('总市值', ascending=False)
    big = spot.head(N_UNIVERSE)
    small = spot.tail(N_UNIVERSE).sort_values('总市值')
    big[['code6', '名称', '总市值', '换手率']].to_csv(
        os.path.join(CACHE, 'universe_large.csv'), index=False)
    small[['code6', '名称', '总市值', '换手率']].to_csv(
        os.path.join(CACHE, 'universe_small.csv'), index=False)
    print(f'股票池: 大盘 {len(big)} / 小盘 {len(small)}')
    return spot


def fetch_hist(codes, start, end, workers=6):
    """逐股日线（qfq），缓存 data/hist/{code}.parquet，断点续传"""
    hist_dir = os.path.join(CACHE, 'hist')
    os.makedirs(hist_dir, exist_ok=True)
    todo = [c for c in codes if not os.path.exists(os.path.join(hist_dir, f'{c}.parquet'))]
    print(f'待抓取 {len(todo)} 只（已缓存 {len(codes)-len(todo)}）', flush=True)
    fail = []

    def work(c):
        import akshare as ak
        for a in range(3):
            try:
                df = ak.stock_zh_a_hist(symbol=c, period='daily',
                                        start_date=start, end_date=end, adjust='qfq')
                if df is not None and len(df) > 30:
                    df['code'] = c
                    df.to_parquet(os.path.join(hist_dir, f'{c}.parquet'))
                    return c
            except Exception as e:
                msg = str(e)
                if '限流' in msg or 'Push2RateLimit' in msg:
                    print(f'  ⚠ {c} 东财限流，跳过', flush=True)
                    fail.append(c)
                    return None
                time.sleep(1 + a * 2)
        fail.append(c)
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if i % 100 == 0:
                print(f'  进度 {i}/{len(todo)} 失败 {len(fail)}', flush=True)
    return fail


def fetch_boards():
    """行业+概念板块成分 -> stock_boards.csv（code, board, kind）"""
    import akshare as ak
    boards_dir = os.path.join(CACHE, 'boards')
    os.makedirs(boards_dir, exist_ok=True)

    def norm(code):
        c = str(code).strip().lower()
        for p in ('sh', 'sz', 'bj'):
            if c.startswith(p):
                c = c[2:]
        return c.zfill(6)

    def sanitize(name):
        return ''.join(ch for ch in str(name) if ch.isalnum() or ch in '._-') or 'board'

    def fetch_members(kind, board):
        fn = {'ind': ak.stock_board_industry_cons_em,
              'con': ak.stock_board_concept_cons_em}[kind]
        for a in range(3):
            try:
                df = fn(symbol=board)
                if df is not None and len(df):
                    df['code'] = df['代码'].map(norm)
                    return df[['code', '名称']]
            except Exception:
                time.sleep(1 + a * 2)
        return None

    rows = []
    for kind, fn_list, tag in [('ind', None, '行业'), ('con', None, '概念')]:
        board_df = (ak.stock_board_industry_name_em() if kind == 'ind'
                    else ak.stock_board_concept_name_em())
        boards = board_df['板块名称'].tolist()
        done = {f[:-4] for f in os.listdir(boards_dir) if f.endswith('.csv')}
        todo = [b for b in boards if f'{kind}_{sanitize(b)}' not in done]
        print(f'{tag}板块: {len(boards)}, 待抓 {len(todo)}', flush=True)
        fail = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(fetch_members, kind, b): b for b in todo}
            for i, f in enumerate(as_completed(futs)):
                b, df = f.result()
                if df is not None:
                    fn = os.path.join(boards_dir, f'{kind}_{sanitize(b)}.csv')
                    df.to_csv(fn, index=False)
                    rows.append(df.assign(board=b, kind=kind)[['code', 'board', 'kind']])
                else:
                    fail.append(b)
        print(f'{tag}完成 失败 {len(fail)}', flush=True)
    allb = pd.concat(rows)
    allb.to_csv(os.path.join(CACHE, 'stock_boards.csv'), index=False)
    print(f'板块映射: {allb["code"].nunique()} 只股票, {allb["board"].nunique()} 个板块')
    return allb


def fetch_index(start, end):
    import akshare as ak
    fp = os.path.join(CACHE, 'index_000300.csv')
    if os.path.exists(fp):
        idx = pd.read_csv(fp)
        idx['日期'] = pd.to_datetime(idx['日期'])
        if idx['日期'].max() >= pd.Timestamp(end):
            return idx
    idx = ak.index_zh_a_hist(symbol='000300', period='daily',
                             start_date=start, end_date=end)
    idx['日期'] = pd.to_datetime(idx['日期'])
    idx.to_csv(fp, index=False)
    print(f'HS300: {idx["日期"].min().date()} ~ {idx["日期"].max().date()}')
    return idx


def build_panel():
    frames = [pd.read_parquet(os.path.join(CACHE, 'hist', f))
              for f in sorted(os.listdir(os.path.join(CACHE, 'hist')))
              if f.endswith('.parquet')]
    panel = pd.concat(frames, ignore_index=True)
    panel['日期'] = pd.to_datetime(panel['日期'])
    panel['code'] = panel['code'].astype(str).str.zfill(6)
    return panel.sort_values(['日期', 'code']).reset_index(drop=True)


# ================= 因子计算 =================
def piecewise_reverse(s, breaks):
    """分段反向：段内 min-max 归一后取反（值越大换手越低越好）"""
    out = pd.Series(np.nan, index=s.index)
    edges = [-np.inf] + list(breaks) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() < 2:
            continue
        v = s[m]
        out.loc[m] = -((v - v.min()) / (v.max() - v.min() + 1e-12))
    return out


def compute_base_factors(panel, windows):
    """7 基础因子 + ret_next（移植自 factors.py，含截面 z-score）"""
    df = panel.copy()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['limit_pct'] = df['code'].str[:2].map(LIMIT_PCT)
    df = df.sort_values(['code', '日期'])
    grp = df.groupby('code', group_keys=False)

    df['vol_ma'] = grp['成交量'].transform(lambda s: s.rolling(windows['vol_ma'], min_periods=3).mean())
    df['vol_ratio'] = df['成交量'] / df['vol_ma']

    df['turnover'] = grp['换手率'].transform(lambda s: s.rolling(windows['to_window'], min_periods=3).mean())
    df['turnover_score'] = df.groupby('code', group_keys=False)['turnover'].apply(
        lambda s: piecewise_reverse(s, TO_BREAKS))

    df['prev_close'] = grp['收盘'].shift(1)
    df['open_str'] = (df['开盘'] - df['prev_close']) / df['prev_close']

    is_zt = (df['涨跌幅'] >= df['limit_pct']).astype(int)
    cnt_cum = is_zt.groupby(df['code']).cumsum()
    seg_start = cnt_cum.mask(is_zt == 1).groupby(df['code']).ffill().fillna(0)
    df['zt_count'] = cnt_cum - seg_start

    df['ma5'] = grp['收盘'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df['ma10'] = grp['收盘'].transform(lambda s: s.rolling(10, min_periods=10).mean())
    df['ma20'] = grp['收盘'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['align'] = ((df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])).astype(float)
    hi = grp['最高'].transform(lambda s: s.rolling(windows['pos_window'], min_periods=20).max())
    df['position'] = 0.4 * df['align'] + 0.6 * (df['收盘'] / hi)
    df['amplitude'] = df['振幅'] / 100.0

    df['ret_next'] = grp['收盘'].transform(lambda s: s.shift(-1) / s - 1)
    df = df.dropna(subset=['vol_ratio', 'turnover_score', 'open_str', 'zt_count'])

    def zs(col):
        g = df.groupby('日期')[col]
        return ((df[col] - g.transform('mean')) / (g.transform('std') + 1e-12)).clip(-3, 3)

    df['score_raw'] = (
        1.695 * zs('vol_ratio')
        + 1.687 * zs('turnover_score')
        + 1.375 * zs('open_str')
        + 1.233 * zs('zt_count')
        + 1.205 * zs('position')
        + (-0.598) * zs('amplitude')
        + (-0.839) * zs('mktcap')
    )
    return df


def compute_mktcap(panel, spot):
    """市值因子（log10 总市值，静态快照）"""
    m = spot[['code6', '总市值']].copy()
    m['code6'] = m['code6'].astype(str).str.zfill(6)
    panel = panel.copy()
    panel['mktcap'] = panel['code'].map(m.set_index('code6')['总市值'])
    panel['mktcap'] = np.log10(panel['mktcap'].astype(float).clip(lower=1e8))
    return panel


def build_board_day(daily_stats, mapping):
    """板块-日统计（移植自 themes.py）"""
    df = daily_stats.merge(mapping, on='code', how='inner')
    df['ztf'] = (df['zt_count'] >= 1).astype('int8')
    bd = df.groupby(['board', '日期'], sort=False).agg(
        ret_1d=('ret_1d', 'mean'), ztf=('ztf', 'sum'), n=('ztf', 'size')).reset_index()
    bd['zt_cnt'] = bd['ztf'].astype(int)
    bd['zt_ratio'] = bd['zt_cnt'] / bd['n'].clip(lower=1)
    bd = bd.sort_values(['board', '日期']).reset_index(drop=True)
    bd['ret_5d'] = bd.groupby('board')['ret_1d'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    bd['rank_pct'] = bd.groupby('日期')['ret_1d'].rank(pct=True)
    bd = bd.drop(columns=['ztf'])
    return bd[['board', '日期', 'ret_1d', 'zt_ratio', 'ret_5d', 'rank_pct']]


def daily_stats_of(panel):
    df = panel[['日期', 'code', '涨跌幅', '收盘', '最高']].copy()
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['limit_pct'] = df['code'].str[:2].map(LIMIT_PCT)
    df = df.sort_values(['code', '日期']).reset_index(drop=True)
    is_zt = (df['涨跌幅'] >= df['limit_pct']).astype(int)
    cnt_cum = is_zt.groupby(df['code']).cumsum()
    seg_start = cnt_cum.mask(is_zt == 1).groupby(df['code']).ffill().fillna(0)
    df['zt_count'] = cnt_cum - seg_start
    df['ret_1d'] = df['涨跌幅'] / 100.0
    df['broken'] = (df['zt_count'] >= 1) & (df['收盘'] < df['最高'] * 0.999)
    return df[['日期', 'code', 'ret_1d', 'zt_count', 'broken']]


def add_theme_factors(fdf, bd, mapping):
    """题材因子（按季度分块合并，省内存）"""
    bd = bd.rename(columns={'ret_1d': 'board_ret'})
    parts = []
    for ym, chunk in fdf.groupby(fdf['日期'].dt.to_period('Q')):
        df = chunk[['日期', 'code', '涨跌幅']].merge(mapping, on='code', how='inner')
        df['ret_1d'] = df['涨跌幅'] / 100.0
        df = df.merge(bd, on=['board', '日期'], how='left')
        df = df.rename(columns={'ret_1d_x': 'ret_1d', 'ret_1d_y': 'board_ret'})
        theme = df.groupby(['code', '日期']).agg(
            theme_mom5=('ret_5d', 'mean'), theme_zt=('zt_ratio', 'mean'),
            theme_rank=('rank_pct', 'mean'), theme_rs=('ret_1d', 'first')).reset_index()
        theme['theme_rs'] = theme['theme_rs'] - df.groupby(['code', '日期'])['board_ret'].mean().values
        parts.append(theme)
        del df, theme
    stock_theme = pd.concat(parts)
    return fdf.merge(stock_theme, on=['code', '日期'], how='left')


def add_zt_ecofactors(fdf):
    zt_cnt = fdf.groupby('日期').apply(
        lambda g: int((g['zt_count'] >= 1).sum()), include_groups=False)
    broken = (fdf['zt_count'] >= 1) & (fdf['收盘'] < fdf['最高'] * 0.999)
    tot = (fdf['zt_count'] >= 1).astype(int) + broken.astype(int)
    broken_ratio = fdf.assign(broken=broken, tot=tot).groupby('日期').apply(
        lambda g: g['broken'].sum() / max(g['tot'].sum(), 1), include_groups=False)
    fdf = fdf.merge(pd.DataFrame({'mkt_zt': zt_cnt, 'mkt_broken': broken_ratio})
                    .reset_index(), on='日期', how='left')
    prem = fdf[fdf['zt_count'] >= 1].groupby('日期')['ret_next'].mean()
    fdf = fdf.merge(prem.rename('zt_premium').reset_index(), on='日期', how='left')
    return fdf


def compute_factors(panel, windows, weights, mapping, bd, spot):
    """完整因子 + 最终得分（基础7因子 z 值 + 版本题材权重）
    panel: 大盘池(前800)；bd: 板块-日统计（由全池构建）"""
    fdf = compute_mktcap(panel, spot)
    fdf = compute_base_factors(fdf, windows)
    fdf = add_theme_factors(fdf, bd, mapping)
    fdf = add_zt_ecofactors(fdf)

    def zs(col):
        g = fdf.groupby('日期')[col]
        return ((fdf[col] - g.transform('mean')) / (g.transform('std') + 1e-12)).clip(-3, 3)

    zcols = {}
    for k in FACTORS:
        zcols[k] = zs(k)
        fdf[f'z_{k}'] = zcols[k]
    fdf['score'] = fdf['score_raw'] + sum(weights[k] * zcols[k] for k in FACTORS)
    return fdf


# ================= 回测 =================
def build_trend(index, gate):
    ma = int(gate[2:])
    idx = index.sort_values('日期').set_index('日期')
    trend = idx['收盘'] > idx['收盘'].rolling(ma, min_periods=ma).mean()
    return trend


def run_backtest(version, fdf, names, index, out_dir):
    cfg = VERSIONS[version]
    px = fdf.pivot_table(index='日期', columns='code',
                         values=['开盘', '收盘', '最高', '最低'], aggfunc='first')
    dates = px.index.sort_values()
    sig = fdf.pivot_table(index='日期', columns='code', values='score', aggfunc='first')
    sig = sig.reindex(index=dates, columns=px.columns.get_level_values(1).unique())
    trend = build_trend(index, cfg['gate']).reindex(dates).fillna(True)
    bench = index.sort_values('日期').set_index('日期')['收盘']
    bench_nav = bench.reindex(dates) / bench.reindex(dates).iloc[0]

    trades, cycle_rets = [], []
    for i in range(len(dates) - 1 - cfg['hold']):
        t = dates[i]
        if not bool(trend.loc[t]):
            continue
        row = sig.loc[t].dropna()
        if len(row) < cfg['top_n']:
            continue
        picks = row.sort_values(ascending=False).head(cfg['top_n']).index.tolist()
        t1, t2 = dates[i + 1], dates[i + 1 + cfg['hold']]
        rets = []
        for c in picks:
            b = px.loc[t1, ('开盘', c)]
            if not np.isfinite(b) or b <= 0:
                continue
            if abs(b - px.loc[t1, ('最高', c)]) / b < 0.002:   # 一字涨停
                continue
            if cfg['sell_at'] == 'open':
                spx = px.loc[t2, ('开盘', c)]
                if not np.isfinite(spx) or spx <= 0:
                    continue
                if abs(spx - px.loc[t2, ('最低', c)]) / spx < 0.002:   # 一字跌停
                    continue
            else:
                spx = px.loc[t2, ('收盘', c)]
                if not np.isfinite(spx) or spx <= 0:
                    continue
            r = (spx / (1 + SELL_COST)) / (b / (1 - BUY_COST)) - 1
            trades.append({'buy_date': t1.date().isoformat(),
                           'sell_date': t2.date().isoformat(),
                           'code': c, 'name': names.get(c, c),
                           'buy_px': round(float(b), 3),
                           'sell_px': round(float(spx), 3),
                           'ret': float(r), 'hold': cfg['hold']})
            rets.append(r)
        if rets:
            cycle_rets.append((t, float(np.mean(rets)) / cfg['hold']))

    tr = pd.DataFrame(trades)
    tr.to_csv(os.path.join(out_dir, f'trades_{version}.csv'), index=False, encoding='utf-8-sig')

    daily = pd.DataFrame({'ret': 0.0}, index=dates)
    for t, r in cycle_rets:
        daily.loc[t, 'ret'] = r
    nav = (1 + daily['ret']).cumprod()
    nav_df = pd.DataFrame({'nav': nav, 'bench': bench_nav, 'ret': daily['ret'], 'date': nav.index})
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.to_csv(os.path.join(out_dir, f'nav_{version}.csv'), index=False, encoding='utf-8-sig')

    cyc = daily.loc[daily['ret'] != 0]
    span = max((cyc.index.max() - cyc.index.min()).days / 365.25, 0.25)
    rt = tr['ret']
    summary = {
        'version': version, 'label': VERSIONS[version]['label'],
        'config': f"hold{cfg['hold']} top{cfg['top_n']} {cfg['sell_at']} gate{cfg['gate']}",
        'windows': cfg['windows'], 'weights': cfg['weights'],
        'start': nav.index.min().date().isoformat(),
        'end': nav.index.max().date().isoformat(),
        'span_years': round(span, 2),
        'total': float(nav.iloc[-1] - 1),
        'ann': float(nav.iloc[-1] ** (1 / span) - 1),
        'mdd': float((nav / nav.cummax() - 1).min()),
        'sharpe': float(cyc['ret'].mean() / (cyc['ret'].std() + 1e-12)
                        * np.sqrt(len(cyc) / span)),
        'win_trade': float((rt > 0).mean()),
        'win_cycle': float((cyc['ret'] > 0).mean()),
        'pf': float(rt[rt > 0].sum() / abs(rt[rt < 0].sum())),
        'avg_trade': float(rt.mean()),
        'avg_win': float(rt[rt > 0].mean()),
        'avg_loss': float(rt[rt < 0].mean()),
        'bench_total': float(bench_nav.iloc[-1] - 1),
        'n_trades': len(tr), 'n_cycles': len(cyc),
    }
    json.dump(summary, open(os.path.join(out_dir, f'summary_{version}.json'), 'w'),
              ensure_ascii=False, indent=1)
    g = nav_df.set_index('date')
    nm = g['nav'].groupby(g.index.to_period('M')).last()
    bm = g['bench'].groupby(g.index.to_period('M')).last()
    mo = pd.DataFrame({'strategy': nm / nm.shift(1) - 1, 'bench': bm / bm.shift(1) - 1}).dropna()
    mo.index = mo.index.astype(str)
    mo.to_csv(os.path.join(out_dir, f'monthly_{version}.csv'))
    return summary


# ================= 信号查询 =================
def cmd_signal(version, date, top):
    cfg = VERSIONS[version]
    spot = load_spot()
    panel = build_panel()
    idx = fetch_index(START, END_DEFAULT)
    mapping = pd.read_csv(os.path.join(CACHE, 'stock_boards.csv'), dtype={'code': str})
    mapping['code'] = mapping['code'].str.zfill(6)
    bd_cache = os.path.join(CACHE, 'board_day.parquet')
    if os.path.exists(bd_cache):
        bd = pd.read_parquet(bd_cache)
    else:
        ds = daily_stats_of(panel)
        bd = build_board_day(ds, mapping)
    big = pd.read_csv(os.path.join(CACHE, 'universe_large.csv'), dtype={'code6': str})
    big['code6'] = big['code6'].str.zfill(6)
    panel_large = panel[panel['code'].isin(set(big['code6']))].reset_index(drop=True)
    fdf = compute_factors(panel_large, cfg['windows'], cfg['weights'], mapping, bd, spot)

    dt = pd.Timestamp(date)
    sig = fdf.pivot_table(index='日期', columns='code', values='score', aggfunc='first')
    if dt not in sig.index:
        near = sig.index[sig.index <= dt].max()
        print(f'（{date} 无完整得分——末日 zt_premium 缺失，回退至最近信号日 {near.date()}）')
        dt = near
    trend = build_trend(idx, cfg['gate'])
    on = bool(trend.reindex([dt]).fillna(False).iloc[0]) if dt in trend.index else False
    day = sig.loc[dt].dropna()
    px = fdf.pivot_table(index='日期', columns='code', values='收盘', aggfunc='first')
    closes = px.loc[dt]
    print(f'\n📅 {dt.date()} 收盘 · {VERSIONS[version]["label"]}（{version}）')
    print(f'  门控 HS300>{cfg["gate"]}: {"开" if on else "关"}')
    print('─' * 70)
    print(f'  {"#":>3} 代码      得分        收盘    板块')
    picks = day.sort_values(ascending=False).head(top)
    for i, (code, sc) in enumerate(picks.items(), 1):
        bs = mapping.loc[mapping['code'] == code, 'board'].head(2).tolist()
        print(f'  {i:>3} {code}  {sc:+.3f}  {closes.get(code, float("nan")):>8.2f}  '
              f'{("、".join(bs)) if bs else ""}')
    print('─' * 70)
    print('  （门控关闭时以上仅为得分排名，不构成买入信号）')


# ================= 主流程 =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='champion',
                    help='champion|balanced|runner|original|wild，默认 champion')
    ap.add_argument('--mode', default='backtest', choices=['backtest', 'signal'])
    ap.add_argument('--date', default='2026-11-05', help='signal 模式的日期')
    ap.add_argument('--top', type=int, default=20, help='signal 模式显示前 N 只')
    ap.add_argument('--end', default=END_DEFAULT, help='数据截止日 YYYYMMDD')
    ap.add_argument('--out', default=os.path.join(BASE, 'standalone_out'))
    ap.add_argument('--refresh', action='store_true', help='强制重新拉取数据')
    args = ap.parse_args()

    ensure_akshare()
    if args.version not in VERSIONS:
        sys.exit(f'未知版本 {args.version}，可选: {list(VERSIONS)}')

    if args.mode == 'signal':
        cmd_signal(args.version, args.date, args.top)
        return

    os.makedirs(args.out, exist_ok=True)
    if args.refresh:
        for f in ['spot.csv', 'index_000300.csv', 'stock_boards.csv']:
            p = os.path.join(CACHE, f)
            if os.path.exists(p):
                os.remove(p)
        import shutil
        shutil.rmtree(os.path.join(CACHE, 'hist'), ignore_errors=True)
        shutil.rmtree(os.path.join(CACHE, 'boards'), ignore_errors=True)

    t0 = time.time()
    print('[1/5] 拉取快照与股票池 ...', flush=True)
    spot = load_spot()
    print('[2/5] 拉取个股日线 ...', flush=True)
    big = pd.read_csv(os.path.join(CACHE, 'universe_large.csv'), dtype={'code6': str})
    small = pd.read_csv(os.path.join(CACHE, 'universe_small.csv'), dtype={'code6': str})
    big['code6'] = big['code6'].str.zfill(6)
    small['code6'] = small['code6'].str.zfill(6)
    codes = list(dict.fromkeys(big['code6'].tolist() + small['code6'].tolist()))
    fetch_hist(codes, START, args.end)
    panel = build_panel()
    print(f'  panel: {panel.shape}, {panel["日期"].min().date()} ~ {panel["日期"].max().date()}')
    print('[3/5] 拉取指数与板块 ...', flush=True)
    idx = fetch_index(START, args.end)
    if not os.path.exists(os.path.join(CACHE, 'stock_boards.csv')):
        fetch_boards()
    mapping = pd.read_csv(os.path.join(CACHE, 'stock_boards.csv'), dtype={'code': str})
    mapping['code'] = mapping['code'].str.zfill(6)
    print('[4/5] 计算因子 ...', flush=True)
    big = pd.read_csv(os.path.join(CACHE, 'universe_large.csv'), dtype={'code6': str})
    big['code6'] = big['code6'].str.zfill(6)
    panel_large = panel[panel['code'].isin(set(big['code6']))].reset_index(drop=True)
    # 板块-日统计用全池；打分与回测用大盘池
    ds = daily_stats_of(panel)
    bd = build_board_day(ds, mapping)
    bd.to_parquet(os.path.join(CACHE, 'board_day.parquet'))
    cfg = VERSIONS[args.version]
    fdf = compute_factors(panel_large, cfg['windows'], cfg['weights'], mapping, bd, spot)
    print(f'  因子表: {fdf.shape}', flush=True)
    print('[5/5] 回测 ...', flush=True)
    names = spot.set_index('code6')['名称'].to_dict()
    summary = run_backtest(args.version, fdf, names, idx, args.out)
    s = summary
    print('\n' + '=' * 64)
    print(f'【{s["label"]} {args.version}】{s["config"]}')
    print(f'  窗口 {s["windows"]} | 题材权重 {s["weights"]}')
    print(f'  夏普 {s["sharpe"]:.2f}  年化 {s["ann"]*100:.1f}%  最大回撤 {s["mdd"]*100:.1f}%')
    print(f'  累计 {s["total"]*100:+.1f}%  基准(HS300) {s["bench_total"]*100:+.1f}%')
    print(f'  单笔胜率 {s["win_trade"]*100:.1f}%  调仓胜率 {s["win_cycle"]*100:.1f}%  '
          f'盈亏比 {s["pf"]:.2f}  笔数 {s["n_trades"]:,}  周期 {s["n_cycles"]}')
    print(f'  输出: {args.out}/ (trades/nav/monthly/summary_{args.version}.*)')
    print(f'  耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
