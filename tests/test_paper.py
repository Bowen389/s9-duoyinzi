import numpy as np
import pandas as pd
from runner.broker import initial, settle, price, buy, sell, equity
from runner.s9_decision import decide
from strategies import s9_source as s9


def bar(p=10,prev=10):
    return dict(open=p,close=p,high=p*1.02,low=p*.98,prev=prev,previous_raw_close=prev,volume=1e6,amount=1e7)


def test_cash_lots_tplusone_and_idempotence():
    a=initial('wild')
    buy(a,'600001','2026-11-06',200000,bar(),True,due='2026-11-09',sell_at='close')
    assert a['cash']>=0
    assert a['lots'][0]['qty']%100==0
    lot=a['lots'][0]
    sell(a,lot,'2026-11-06',bar(),True)
    assert len(a['trades'])==1
    a=settle(a,'2026-11-09',None,{'600001':bar()},['2026-11-06','2026-11-09'])
    assert not a['lots'] and len(a['trades'])==2
    b=settle(a,'2026-11-09',None,{},[])
    assert b==a


def test_limit_sell_defers_and_cannot_create_buying_power():
    a=initial('wild')
    buy(a,'600001','2026-11-06',100000,bar(),True,due='2026-11-09',sell_at='close')
    a=settle(a,'2026-11-09',None,{'600001':bar(9)},['2026-11-06','2026-11-09'])
    assert a['lots'] and any(e['event']=='SELL_DEFERRED' for e in a['events'])
    assert price(bar(11),'BUY','open',True,'600001') is None


def test_corporate_action_quarantines_and_persists():
    a=initial('wild')
    buy(a,'600001','2026-11-06',100000,bar(),True,due='2026-11-09',sell_at='close')
    b=bar(5,prev=5);b['previous_raw_close']=10
    a=settle(a,'2026-11-09',None,{'600001':b},['2026-11-06','2026-11-09'])
    assert a['corporate_review']==['600001']
    a=settle(a,'2026-11-10',None,{'600001':bar(5,5)},['2026-11-06','2026-11-09','2026-11-10'])
    assert a['lots'] and a['lots'][0]['mark']==5


def test_cohort_budget_and_entry_plus_hold():
    a=initial('wild')
    sig=dict(status='ok',exec_day='2026-11-06',gate_on=True,picks=[{'code':'600001'},{'code':'600002'}])
    days=['2026-11-06','2026-11-09','2026-11-10']
    a=settle(a,days[0],sig,{'600001':bar(),'600002':bar()},days,dict(hold=1,top_n=2,sell_at='close'))
    assert a['cash']>=100000 and len(a['lots'])==2
    assert all(x['due']=='2026-11-09' for x in a['lots'])


def test_star_minimum_and_one_share_increment():
    a=initial('champion')
    buy(a,'688001','2026-11-06',1000,bar(),True)
    assert not a['lots']
    buy(a,'688001','2026-11-06',2345,bar(),True)
    assert a['lots'][0]['qty']>=200 and a['lots'][0]['qty']%100!=0


def panel():
    rng=np.random.default_rng(14)
    dates=pd.bdate_range('2024-01-01',periods=380)
    codes=np.array(['510300','510500','512880','159915','511260','518880'])
    close=np.exp(np.cumsum(rng.normal(.001,.009,(len(dates),len(codes))),axis=0))*3
    return dict(dates=dates,codes=codes,names={c:c for c in codes},close=close,
                open=close*.999,high=close*1.01,low=close*.99,
                amount=np.ones_like(close)*1e8,bench=pd.Series(np.arange(len(dates))+3000.,index=dates),
                cash_ret=np.zeros(len(dates)))


def test_s9_decision_matches_original_engine_from_empty():
    P=panel()
    for profile in s9.PROFILES:
        par={**s9.DEFAULT,**s9.S9,**s9.PROFILES[profile]}
        for t in [150,230,330]:
            bt=s9.Backtest(P,start=str(P['dates'][t].date()),end=str(P['dates'][t+1].date()),static_pool=list(P['codes']),**par)
            r=bt.run()
            target,weights,_,_=decide(bt,t,np.array([],dtype=int),np.full(6,np.nan),np.full(6,-1),np.full(6,-10**6))
            assert {bt.codes[c] for c in target} == set(r['hold_path'][1])
            assert weights is None or sum(weights)<=1.000001


def test_causal_turnover_does_not_change_past_with_future_data():
    from runner.signals import causal_piecewise
    x=pd.Series([1.2,1.8,2.2,1.5,6.,9.,.5])
    for n in range(2,len(x)):
        pd.testing.assert_series_equal(causal_piecewise(x[:n],[1.,3.,7.]),causal_piecewise(x,[1.,3.,7.])[:n])


def test_trading_calendar_skips_weekend():
    from runner.data import next_session
    assert next_session('2026-11-06')=='2026-11-09'


def test_stock_pipeline_emits_current_date_for_all_five_versions(monkeypatch):
    from runner import signals as s
    dates=pd.bdate_range('2026-01-01',periods=150)
    day=str(dates[-1].date())
    codes=[f'600{i:03d}' for i in range(40)]
    spot=pd.DataFrame({'代码':codes,'code6':codes,'名称':codes,'成交量':1e6,
                       '总市值':np.linspace(1e9,1e11,len(codes))})
    rng=np.random.default_rng(11)
    frames={}
    for c in codes:
        close=10*np.exp(np.cumsum(rng.normal(.001,.015,len(dates))))
        frames[c]=pd.DataFrame(dict(date=dates,code=c,open=close*.998,close=close,
                                   high=close*1.01,low=close*.99,volume=rng.uniform(1e5,1e6,len(dates)),
                                   amount=1e8,amplitude=2.,pct=rng.normal(0,2,len(dates)),
                                   turnover=rng.uniform(.5,10,len(dates))))
    monkeypatch.setattr(s.data,'stock_spot',lambda d:spot)
    monkeypatch.setattr(s.data,'histories',lambda codes,d:(frames,{}))
    monkeypatch.setattr(s.data,'board_mapping',lambda d:pd.DataFrame(dict(code=codes,board=['sample']*len(codes),kind='ind')))
    monkeypatch.setattr(s.data,'history',lambda *a,**kw:pd.DataFrame(dict(date=dates,close=np.arange(len(dates))+3000.)))
    monkeypatch.setattr(s.data,'next_session',lambda d:'2026-11-06')
    result=s.stock_signals(day)
    assert len(result)==5
    assert all(x['status']=='ok' and x['asof']==day and x['gate_on'] for x in result.values())
    assert len(result['wild']['picks'])==2
    assert all(len(result[x]['picks'])==20 for x in ['champion','balanced','runner','original'])
