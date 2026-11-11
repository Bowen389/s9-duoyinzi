"""Two-slot ETF rotation. Produces an observation ranking even with a closed gate."""
import re

import numpy as np
import pandas as pd

from . import data

GOLD = '518880'
DEFENSE = {'511360': 'short', '511010': 'government', '511260': 'government', GOLD: 'gold'}
EXCLUDED = re.compile(r'货币|现金|日利|添益|保证金|债ETF|债券ETF')


def dual_signal(day, panel, account):
    codes = panel['codes']
    names = panel['names']
    lookup = {c: i for i, c in enumerate(codes)}
    if set(DEFENSE) - lookup.keys():
        raise ValueError('DUAL55 缺少当日防守 ETF 行情')
    close = panel['fresh_close']
    if len(close) < 75:
        raise ValueError('DUAL55 历史不足 75 个交易日')
    history = close[-55:]
    valid = (np.isfinite(history) & (history > 0)).all(axis=0)
    logs = np.log(np.where(valid, history, 1.))
    x = np.arange(55) - 27
    covariance = x @ logs
    variance = np.sum((logs - logs.mean(axis=0)) ** 2, axis=0)
    fit = np.divide(covariance**2, (x @ x) * variance, out=np.zeros_like(variance), where=variance > 0)
    score = covariance / (x @ x) * 252 * np.clip(fit, 0, 1)
    eligible = valid & (panel['age'][-1] >= 60) & (panel['amount'][-10:].mean(axis=0) >= 10_000_000)
    short_positive = close[-1] > close[-10]
    held = {lot['code'] for lot in account['lots']}
    attack = sorted((c for c in codes if c not in DEFENSE and not c.startswith('511')
                     and not EXCLUDED.search(names.get(c, '')) and eligible[lookup[c]]
                     and short_positive[lookup[c]] and score[lookup[c]] > 0
                     and close[-1, lookup[c]] > history[:, lookup[c]].mean()),
                    key=lambda c: (-score[lookup[c]] * (1.2 if c in held else 1.), c))
    picks = [dict(code=str(c), name=names.get(c, c), score=float(score[lookup[c]]),
                  ref_close=float(close[-1, lookup[c]])) for c in attack[:10]]
    benchmark = panel['bench'].iloc[-75:]
    ma75 = float(benchmark.mean())
    gate = bool(benchmark.iloc[-1] > ma75)
    gold_idx = lookup[GOLD]
    blocked = bool(account.get('gold_blocked', False)) and not bool(short_positive[gold_idx])
    if GOLD in held:
        anchor = account.get('gold_anchor')
        if not anchor:
            raise ValueError('DUAL55 缺少黄金首次买入成本')
        entry = pd.Timestamp(anchor['date'])
        raw = data.history(GOLD, day, adjusted=False)
        bar = raw.loc[raw.date == entry, 'close']
        if bar.empty or entry not in panel['dates']:
            raise ValueError('DUAL55 缺少黄金首笔买入日行情')
        adjusted_entry = close[panel['dates'].get_loc(entry), gold_idx]
        basis = anchor['price'] * adjusted_entry / float(bar.iloc[-1])
        if not np.isfinite(basis) or basis <= 0:
            raise ValueError('DUAL55 黄金复权成本无效')
        blocked |= bool(close[-1, gold_idx] <= basis * .95)
    target = [str(c) for c in attack[:1]] if gate else []
    groups = set()
    defensive = []
    for c in sorted(DEFENSE, key=lambda c: (-score[lookup[c]], c)):
        j = lookup[c]
        if not eligible[j] or (c == GOLD and blocked):
            continue
        defensive.append(dict(code=c, name=names.get(c, c), score=float(score[j])))
        if len(target) < 2 and DEFENSE[c] not in groups:
            target.append(c)
            groups.add(DEFENSE[c])
    account['gold_blocked'] = blocked
    return dict(strategy='DUAL55', label='ETF 双席趋势', status='ok', asof=day,
                exec_day=data.next_session(day), gate_on=gate, picks=picks, defense_picks=defensive,
                benchmark_close=float(benchmark.iloc[-1]), benchmark_ma75=ma75, gold_blocked=blocked,
                rebalance=True, weights={c: .5 for c in target}, target_codes=target)
