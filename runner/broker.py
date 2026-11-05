"""Cash-constrained paper broker. No brokerage connection, no orders sent externally."""
from copy import deepcopy
import math

START = '2026-11-06'
STOCKS = ['champion', 'balanced', 'runner', 'original', 'wild']
ETFS = ['B0', 'K3', 'K5']


def initial(name):
    return dict(strategy=name, start=START, initial_cash=200000., cash=200000., lots=[],
                last_settled=None, trades=[], nav=[], events=[], last_exit={}, peak=200000.)


def fee(value, side, stock):
    commission = max(5., value * (0.00025 if stock else 0.0001))
    return commission + (value * 0.00001 if stock else 0.) + (value * .0005 if stock and side == 'SELL' else 0.)


def price(bar, side, phase, stock, code):
    if not bar or bar['volume'] <= 0 or bar['amount'] <= 0:
        return None
    p = bar[phase]
    if not math.isfinite(p) or p <= 0:
        return None
    limit = .20 if stock and code.startswith(('30', '68')) else .10
    upper, lower = round(bar['prev'] * (1 + limit) + 1e-8, 2 if stock else 3), round(bar['prev'] * (1 - limit) + 1e-8, 2 if stock else 3)
    # Conservative: do not assume queue priority at a limit price.
    if (side == 'BUY' and p >= upper - .00001) or (side == 'SELL' and p <= lower + .00001):
        return None
    slip = .0015 if stock else .0002
    return min(bar['high'], p * (1 + slip)) if side == 'BUY' else max(bar['low'], p * (1 - slip))


def equity(a):
    return a['cash'] + sum(x['qty'] * x['mark'] for x in a['lots'])


def buy(a, code, day, budget, bar, stock, phase='open', due=None, sell_at=None):
    p = price(bar, 'BUY', phase, stock, code)
    if p is None:
        a['events'].append(dict(date=day, code=code, event='BUY_UNFILLED', reason='missing/suspended/limit'))
        return
    # STAR board minimum 200; subsequent one-share increments. Others minimum/step 100.
    step, minimum = (1, 200) if stock and code.startswith('68') else (100, 100)
    q = math.floor(min(budget, a['cash']) / p / step) * step
    while q >= minimum and q * p + fee(q * p, 'BUY', stock) > min(budget, a['cash']) + 1e-7:
        q -= step
    if q < minimum:
        a['events'].append(dict(date=day, code=code, event='BUY_UNFILLED', reason='insufficient lot budget'))
        return
    cost = q * p + fee(q * p, 'BUY', stock)
    a['cash'] -= cost
    a['lots'].append(dict(code=code, qty=q, entry=day, entry_px=p, cost=cost,
                          mark=p, due=due, sell_at=sell_at))
    a['trades'].append(dict(date=day, phase=phase, code=code, side='BUY', qty=q, price=p, fee=cost-q*p, cash=a['cash']))


def sell(a, lot, day, bar, stock, phase='open', qty=None):
    if lot['entry'] >= day:
        return
    p = price(bar, 'SELL', phase, stock, lot['code'])
    if p is None:
        a['events'].append(dict(date=day, code=lot['code'], event='SELL_DEFERRED', reason='missing/suspended/limit'))
        return
    q = min(lot['qty'], qty if qty is not None else lot['qty'])
    cost_basis = lot['cost'] * q / lot['qty']
    proceeds = q * p - fee(q*p, 'SELL', stock)
    a['cash'] += proceeds
    a['trades'].append(dict(date=day, phase=phase, code=lot['code'], side='SELL', qty=q,
                            price=p, fee=q*p-proceeds, pnl=proceeds-cost_basis, cash=a['cash']))
    lot['qty'] -= q
    lot['cost'] -= cost_basis
    if not lot['qty']:
        a['lots'].remove(lot)
        if not any(x['code'] == lot['code'] for x in a['lots']):
            a['last_exit'][lot['code']] = day


def settle(account, day, signal, bars, trading_days, cfg=None):
    if account['last_settled'] and day <= account['last_settled']:
        return account
    a = deepcopy(account)
    stock = a['strategy'] in STOCKS
    # Corporate actions are quarantined rather than silently manufacturing P&L.
    blocked = set(a.get('corporate_review', []))
    for lot in a['lots']:
        b = bars.get(lot['code'])
        if b and abs(b['prev']/b.get('previous_raw_close', b['prev'])-1) > .0002:
            blocked.add(lot['code'])
            a['events'].append(dict(date=day, code=lot['code'], event='CORPORATE_ACTION_REVIEW',
                                    reason='ex-rights/dividend detected; do not assume shares/cash adjustment'))
    a['corporate_review'] = sorted(blocked)
    safe = {c: b for c, b in bars.items() if c not in blocked}
    if stock:
        for lot in list(a['lots']):
            if lot['due'] <= day and lot['sell_at'] == 'open':
                sell(a, lot, day, safe.get(lot['code']), True)
        if signal and signal['exec_day'] == day and signal['status'] == 'ok' and signal['gate_on']:
            # hold=3 exits entry+3 sessions: four overlapping cohorts including entry day.
            cohort = equity(a) / (cfg['hold'] + 1)
            budget = min(cohort, a['cash']) / cfg['top_n']
            due = trading_days[trading_days.index(day) + cfg['hold']]
            for x in signal['picks'][:cfg['top_n']]:
                buy(a, x['code'], day, budget, safe.get(x['code']), True, due=due, sell_at=cfg['sell_at'])
        for lot in list(a['lots']):
            if lot['due'] <= day and lot['sell_at'] == 'close':
                sell(a, lot, day, safe.get(lot['code']), True, 'close')
    elif signal and signal['exec_day'] == day and signal['status'] == 'ok' and signal['rebalance']:
        eq = equity(a)
        target = signal['weights']
        # Sells before buys; failed sells consume cash and do not enable imaginary buying power.
        wanted = {}
        for c, w in target.items():
            b = safe.get(c)
            if b:
                wanted[c] = max(0, int(eq*w/b['open']/100)*100)
        for lot in list(a['lots']):
            c = lot['code']
            if c in target and c not in wanted:
                continue
            total = sum(x['qty'] for x in a['lots'] if x['code']==c)
            reduction = max(0, total - wanted.get(c, 0))
            if reduction:
                sell(a, lot, day, safe.get(c), False, qty=min(lot['qty'], reduction))
        for c, w in target.items():
            b = safe.get(c)
            held = sum(x['qty'] for x in a['lots'] if x['code']==c)
            if b and wanted.get(c, 0) > held:
                buy(a, c, day, (wanted[c]-held)*b['open'], b, False)
    for lot in a['lots']:
        if lot['code'] in bars:
            lot['mark'] = bars[lot['code']]['close']
        else:
            a['events'].append(dict(date=day, code=lot['code'], event='STALE_MARK', reason='no current bar; last mark retained'))
    value = equity(a)
    a['peak'] = max(a['peak'], value)
    a['last_settled'] = day
    a['nav'].append(dict(date=day, cash=a['cash'], equity=value, return_pct=(value/200000-1)*100,
                         positions=len(set(x['code'] for x in a['lots'])), valuation_review=bool(blocked)))
    assert a['cash'] >= -1e-6
    return a
