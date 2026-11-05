"""Read-only retrospective analysis. Never writes signals or paper accounts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import data
from .signals import causal_piecewise,causal_ecology
from strategies import champion_source as ch

ASOF='2026-11-05'
DAYS=['2026-11-02','2026-11-03','2026-11-04','2026-11-05']


def main():
    cache=data.CACHE
    spot_path=cache/f'spot_{ASOF}.parquet'
    mapping_path=cache/f'board_mapping_{ASOF}.parquet'
    if not spot_path.exists() or not mapping_path.exists():
        raise RuntimeError('Production cache not restored; refuse to replace historical snapshot with live data')
    spot=pd.read_parquet(spot_path)
    big=spot.head(800).code6.tolist()
    universe=sorted(set(big+spot.tail(800).code6.tolist()))
    frames=[]
    for c in universe:
        p=cache/'qfq'/'2024-01-01'/f'{c}.parquet'
        if not p.exists():
            raise RuntimeError(f'production history cache missing {c}')
        f=pd.read_parquet(p)
        f=f[f.date<=pd.Timestamp(ASOF)]
        frames.append(f)
    panel=pd.concat(frames,ignore_index=True).rename(columns={
        'date':'日期','open':'开盘','close':'收盘','high':'最高','low':'最低',
        'volume':'成交量','amount':'成交额','amplitude':'振幅','pct':'涨跌幅','turnover':'换手率'})
    mapping=pd.read_parquet(mapping_path)
    ch.piecewise_reverse=causal_piecewise
    ch.add_zt_ecofactors=causal_ecology
    cfg=ch.VERSIONS['wild']
    bd=ch.build_board_day(ch.daily_stats_of(panel),mapping)
    fdf=ch.compute_factors(panel[panel.code.isin(big)].copy(),cfg['windows'],cfg['weights'],mapping,bd,spot)
    idx=data.history('000300',ASOF,index=True).rename(columns={'date':'日期','close':'收盘'})
    gate=ch.build_trend(idx,cfg['gate'])
    names=spot.set_index('code6')['名称'].to_dict()
    selected={}
    for day in DAYS:
        d=fdf[(fdf['日期']==pd.Timestamp(day)) & (fdf['成交量']>0)].copy()
        if day==ASOF:
            active=set(spot.loc[spot['成交量']>0,'code6'])
            d=d[d.code.isin(active)]
        d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=['score'])
        selected[day]=d.sort_values(['score','code'],ascending=[False,True]).head(2)
    live=json.loads((data.ROOT/'signals'/ASOF/'wild.json').read_text())
    latest=selected[ASOF]
    if latest.code.tolist()!=[x['code'] for x in live['picks']]:
        raise RuntimeError('Replay does not reproduce recorded 11/05 candidates')
    if not np.allclose(latest.score.to_numpy(),[x['score'] for x in live['picks']],atol=1e-8,rtol=1e-8):
        raise RuntimeError('Replay scores differ from production snapshot')
    codes=sorted({r.code for d in selected.values() for _,r in d.iterrows()})
    raw,errors=data.histories(codes,ASOF,adjusted=False,start='2026-10-01',workers=4)
    if errors:
        raise RuntimeError(f'Raw OHLC unavailable: {errors}')
    def px(c,day,field):
        if day>ASOF:
            return None
        d=raw[c][raw[c].date==pd.Timestamp(day)]
        return float(d.iloc[-1][field]) if len(d) else None
    rows=[]
    for day,df in selected.items():
        entry=data.next_session(day)
        exitday=data.next_session(entry)
        for rank,(_,r) in enumerate(df.iterrows(),1):
            c=r.code
            so,sc=px(c,day,'open'),px(c,day,'close')
            bo,bc=px(c,entry,'open'),px(c,entry,'close')
            xc=px(c,exitday,'close')
            rows.append(dict(signal_date=day,record_type='actual' if day==ASOF else 'recomputed_fixed_1105_universe',
                rank=rank,code=c,name=names[c],score=float(r.score),gate_on=bool(gate.loc[pd.Timestamp(day)]),
                signal_open=so,signal_close=sc,signal_intraday_pct=(sc/so-1)*100 if so and sc else None,
                entry_day=entry,entry_open=bo,entry_close=bc,entry_intraday_pct=(bc/bo-1)*100 if bo and bc else None,
                exit_day=exitday,exit_close=xc,wild_gross_pct=(xc/bo-1)*100 if xc and bo else None))
    result=dict(asof=ASOF,scope='last four trading days',
        caveat='11/02-11/04 are retrospective rankings with 11/05 frozen universe, market caps and board mapping; not archived live signals. 11/05 exactly reproduces archived codes and scores. OHLC is unadjusted; hypothetical gross returns omit costs and gate execution.',rows=rows)
    out=data.ROOT/'research_output'
    out.mkdir(exist_ok=True)
    (out/'wild_last4.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print('WILD_REPLAY_JSON '+json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
