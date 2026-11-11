import gc
import numpy as np
import pandas as pd
from . import data
from .broker import equity
from strategies import champion_source as ch, s9_source as s9
from .s9_decision import decide
from .dual55 import DEFENSE, dual_signal


def causal_piecewise(s, breaks):
    out = pd.Series(np.nan, index=s.index)
    edges = [-np.inf] + list(breaks) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = s[(s >= lo) & (s < hi)]
        out.loc[v.index] = -(v-v.cummin())/(v.cummax()-v.cummin()+1e-12)
    return out


def causal_ecology(fdf):
    # Original premium is a date-level constant and therefore has zero cross-sectional
    # z-score. Preserve its zero contribution without requiring an unknown next close.
    fdf = fdf.copy()
    fdf['zt_premium'] = 0.
    return fdf


def stock_signals(day):
    spot = data.stock_spot(day)
    big = spot.head(800).code6.tolist()
    small = spot.tail(800).code6.tolist()
    codes = sorted(set(big + small))
    frames, errs = data.histories(codes, day)
    active = set(spot.loc[spot['成交量'] > 0, 'code6']) & set(codes)
    bad = [c for c in active if c not in frames or str(frames[c].date.max().date()) != day]
    for c in bad:
        errs[c] = errs.get(c, 'missing current-session history; excluded from ranking')
        frames.pop(c, None)
    if bad:
        print(f'::warning::Excluded {len(bad)} active stocks with missing/stale histories: {bad[:10]}', flush=True)
    if len(frames) < len(codes)*.98:
        raise RuntimeError(f'stock coverage {len(frames)}/{len(codes)}')
    mapping = data.board_mapping(day)
    unmapped = set(big) - set(mapping.code)
    if unmapped:
        for c in unmapped:
            errs[c] = 'board membership missing; excluded from ranking'
            frames.pop(c, None)
        print(f'::warning::Excluded {len(unmapped)} stocks without board membership', flush=True)
    if len(frames) < len(codes)*.98:
        raise RuntimeError(f'stock coverage after exclusions {len(frames)}/{len(codes)}')
    big = [c for c in big if c in frames]
    active &= set(frames)
    panel = pd.concat(frames.values(), ignore_index=True).rename(columns={
        'date':'日期', 'open':'开盘', 'close':'收盘', 'high':'最高', 'low':'最低',
        'volume':'成交量', 'amount':'成交额', 'amplitude':'振幅', 'pct':'涨跌幅', 'turnover':'换手率'})
    bd = ch.build_board_day(ch.daily_stats_of(panel), mapping)
    idx = data.history('000300', day, index=True).rename(columns={'date':'日期','close':'收盘'})
    if str(idx['日期'].max().date()) != day:
        raise RuntimeError('stale HS300')
    ch.piecewise_reverse = causal_piecewise
    ch.add_zt_ecofactors = causal_ecology
    names = spot.set_index('code6')['名称'].to_dict()
    out = {}
    for version, cfg in ch.VERSIONS.items():
        fdf = ch.compute_factors(panel[panel.code.isin(big)].copy(), cfg['windows'], cfg['weights'], mapping, bd, spot)
        today = fdf[(fdf['日期']==pd.Timestamp(day)) & fdf.code.isin(active)].copy()
        today = today.replace([np.inf, -np.inf], np.nan).dropna(subset=['score'])
        today = today.sort_values(['score','code'], ascending=[False,True])
        if len(today) < cfg['top_n']:
            raise RuntimeError(f'{version}: not enough current scores')
        gate = ch.build_trend(idx, cfg['gate'])
        picks = [dict(code=r.code, name=names.get(r.code,r.code), score=float(r.score), ref_close=float(r['收盘']))
                 for _,r in today.head(cfg['top_n']).iterrows()]
        out[version] = dict(strategy=version, label=cfg['label'], status='ok', asof=day,
                            exec_day=data.next_session(day), gate_on=bool(gate.loc[pd.Timestamp(day)]),
                            picks=picks, config=cfg, coverage=dict(loaded=len(frames), expected=len(codes), inactive_errors=errs,
                            vendors=sorted(set(panel['source'])) if 'source' in panel else [],
                            board_catalog_asof=str(mapping['catalog_asof'].iloc[0]) if 'catalog_asof' in mapping else day),
                            live_patch='causal turnover expanding bounds; constant premium contributes zero; no stale-date fallback')
        del fdf
        gc.collect()
    return out


def etf_panel(day, protected=()):
    codes = [x.strip() for x in (data.ROOT/'strategies/etf_codes.txt').read_text().splitlines() if x.strip() and not x.startswith('_')]
    codes = sorted(set(codes) | set(s9.POOL) | {'511260','518880','511880'} | set(DEFENSE))
    frames, errs = data.histories(codes, day, start='2016-01-01')
    # Never forward-fill stale current-session data into an actionable ranking.
    for c, frame in list(frames.items()):
        if frame.empty or str(frame.date.max().date()) != day:
            errs[c] = 'missing current-session history; excluded from ranking'
            frames.pop(c)
    from .eligibility import classify_unavailable
    mandatory=set(s9.POOL) | {'511260','518880','511880'} | set(protected)
    excluded=classify_unavailable(errs,day,protected=mandatory,min_age=60) if errs else {}
    unresolved={c:e for c,e in errs.items() if c not in excluded}
    expected=len(codes)-len(excluded)
    health=dict(asof=day,total_manifest=len(codes),loaded=len(frames),
                eligible_data_expected=expected,verified_young_or_unlisted=excluded,
                unresolved=unresolved,coverage=len(frames)/expected if expected else 0.)
    from .main import write_json
    write_json(data.ROOT/'reports'/'data_quality'/f'etf-{day}.json',health)
    print(f'ETF completeness: {len(frames)}/{expected}; manifest {len(codes)}, '
          f'verified age-ineligible {len(excluded)}, unresolved {len(unresolved)}',flush=True)
    if len(frames) < .98*expected:
        raise RuntimeError(f'ETF history coverage {len(frames)}/{expected} eligible records; '
                           f'{len(excluded)} verified age-ineligible; unresolved {list(unresolved)[:10]}')
    # Missing static-pool members are excluded by the panel code index.
    # Held and defensive instruments still require fresh prices.
    for c in sorted({'511260','518880','511880'} | set(protected)):
        if c not in frames:
            raise RuntimeError(f'ETF core/defensive data missing: {c}')
    idx = data.history('000300', day, index=True, start='2016-01-01')
    if str(idx.date.max().date()) != day:
        raise RuntimeError('stale HS300')
    cal = pd.DatetimeIndex(idx.date)
    daily = pd.concat(frames.values(),ignore_index=True)
    codes = np.array(sorted(frames))
    fields = {}
    for k in ['open','high','low','close','amount']:
        w = daily.pivot(index='date',columns='code',values=k).reindex(index=cal, columns=codes)
        fields[k] = (w.fillna(0) if k=='amount' else w.ffill(limit=5)).to_numpy(float)
    fresh_close = daily.pivot(index='date',columns='code',values='close').reindex(index=cal, columns=codes)
    fields['fresh_close'] = fresh_close.to_numpy(float)
    fields['age'] = fresh_close.notna().cummax().cumsum().to_numpy()
    extra = frames['511880'].set_index('date').close.reindex(cal).ffill()
    return dict(dates=cal,codes=codes,names={c:f.iloc[-1]['name'] for c,f in frames.items()},
                bench=pd.Series(idx.close.to_numpy(),index=cal),
                cash_ret=extra.pct_change().fillna(0).clip(-.05,.05).to_numpy(), data_health=health, **fields), unresolved


def etf_signals(day, accounts):
    protected={x['code'] for a in accounts.values() for x in a['lots']}
    panel, errs = etf_panel(day,protected)
    t = len(panel['dates'])-1
    out = {}
    for profile in s9.PROFILES:
        a = accounts[profile]
        params={**s9.DEFAULT, **s9.S9, **s9.PROFILES[profile]}
        bt=s9.Backtest(panel,static_pool=s9.POOL,**params)
        held_codes=sorted(set(x['code'] for x in a['lots']))
        if any(c not in bt.cidx for c in held_codes):
            raise RuntimeError('held ETF absent from factor panel')
        held=np.array([bt.cidx[c] for c in held_codes],dtype=int)
        entry_px=np.full(len(bt.codes),np.nan)
        entry_t=np.full(len(bt.codes),-1)
        last_exit=np.full(len(bt.codes),-10**6)
        for c in held_codes:
            lots=[x for x in a['lots'] if x['code']==c]
            # Convert raw fill cost to the adjusted price scale used for trend/stop factors.
            i=bt.cidx[c]
            entry=min(x['entry'] for x in lots)
            j=int(panel['dates'].get_indexer([pd.Timestamp(entry)])[0])
            if j<0:
                raise RuntimeError('paper entry outside ETF factor calendar')
            qty=sum(x['qty'] for x in lots)
            raw_average=sum(x['entry_px']*x['qty'] for x in lots)/qty
            raw_entry=data.history(c,day,adjusted=False)
            raw_row=raw_entry[raw_entry.date==pd.Timestamp(entry)]
            if raw_row.empty:
                raise RuntimeError('missing raw ETF entry bar')
            entry_px[i]=raw_average*(bt.close[j,i]/float(raw_row.iloc[-1].close))
            entry_t[i]=j
        for c,d in a['last_exit'].items():
            if c in bt.cidx:
                j=panel['dates'].get_indexer([pd.Timestamp(d)])[0]
                if j>=0:
                    last_exit[bt.cidx[c]]=j
        target, weights, why, exits=decide(bt,t,held,entry_px,entry_t,last_exit)
        # Preserve source semantics: no daily weight reset when target membership is unchanged.
        changed=weights is not None
        wmap={str(bt.codes[c]):float(w) for c,w in zip(target,weights)} if changed else {}
        # Re-attempt failed/pending reductions: account state will still differ from target.
        rank=bt._rank(t)
        picks=[dict(code=str(bt.codes[i]),name=str(bt.names.get(bt.codes[i],'')), score=float(bt.score[t,i])) for i in rank[:10]]
        out[profile]=dict(strategy=profile,label='ETF S9 '+profile,status='ok',asof=day,
                          exec_day=data.next_session(day),gate_on=bool(bt.regime[t]),picks=picks,
                          weights=wmap,rebalance=changed,target_codes=[str(bt.codes[i]) for i in target],
                          reasons={str(bt.codes[c]):v for c,v in why.items()},params=params,
                          data_warnings=errs,data_health=panel['data_health'],live_warmup='up to 640 actual sessions per ETF; no inherited historical holdings')
        # Source cooldown begins at the close decision, even if exit executes next session.
        for c in np.flatnonzero(exits > -10**6):
            a['last_exit'][str(bt.codes[c])]=str(panel['dates'][exits[c]].date())
        del bt
        gc.collect()
    try:
        if accounts['DUAL55']['last_settled'] != day:
            raise ValueError('DUAL55 account unsettled')
        out['DUAL55'] = dual_signal(day, panel, accounts['DUAL55'])
    except Exception as exc:
        out['DUAL55'] = dict(strategy='DUAL55', label='ETF 双席趋势', status='data_error',
                             asof=day, exec_day=data.next_session(day), error=str(exc), picks=[])
    return out
