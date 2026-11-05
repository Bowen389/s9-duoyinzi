import pandas as pd
import pytest
from runner import data


def test_tencent_rich_fields_units_are_real_vendor_values(monkeypatch):
    j={'data':{'sh510300':{'qfqday':[['2026-11-04','4','4.1','4.2','3.9','1234',{},'1.25','100.5','']],
                            'qt':{'sh510300':['1','ETF名称']}}}}
    monkeypatch.setattr(data,'get_json',lambda *a,**k:j)
    f=data._tx_piece('510300','2026-11-04','2024-01-01',True,False)
    assert f.iloc[0].amount==1005000
    assert f.iloc[0].turnover==1.25
    assert f.iloc[0].volume==1234


def test_short_legacy_kline_does_not_fabricate_turnover_or_amount(monkeypatch):
    monkeypatch.setattr(data,'get_json',lambda *a,**k:{'data':{'sh510300':{'qfqday':[['2026-11-04','4','4','4','4','10']]}}})
    with pytest.raises(RuntimeError,match='missing amount/turnover'):
        data._tx_piece('510300','2026-11-04','2024-01-01',True,False)


def test_outage_probes_skip_remaining_thousands(monkeypatch):
    called=[]
    def down(c,*a,**k):
        called.append(c)
        raise RuntimeError('upstream disconnected')
    monkeypatch.setattr(data,'history',down)
    codes=[f'{i:06d}' for i in range(1777)]
    frames,errors=data.histories(codes,'2026-11-05')
    assert not frames and len(errors)==1777 and len(called)==3


def test_host_circuit_stops_retrying_on_rate_limit(monkeypatch):
    data._HOSTS.clear()
    called=[]
    def disconnected(*a,**k):
        called.append(1)
        raise data.requests.exceptions.ConnectionError('disconnected')
    monkeypatch.setattr(data.requests,'get',disconnected)
    monkeypatch.setattr(data.time,'sleep',lambda _:None)
    for _ in range(10):
        with pytest.raises(RuntimeError):
            data.get_json('https://unavailable.test/data')
    assert len(called)<=8
    data._HOSTS.clear()


def test_sina_market_cap_is_converted_from_ten_thousand_yuan(monkeypatch, tmp_path):
    monkeypatch.setattr(data,"CACHE",tmp_path)
    def reply(url,params):
        if url.endswith('StockCount'):
            return '2'
        return [dict(code=c,name=c,trade='10',volume=1000,amount=10000,turnoverratio=.3,mktcap=12345) for c in ['600001','600002']]
    monkeypatch.setattr(data,'get_json',reply)
    f=data._sina_spot()
    assert len(f)==2 and (f['总市值']==123450000).all()


def test_qfq_incremental_refresh_rescales_earlier_prices(monkeypatch):
    old=pd.DataFrame(dict(date=pd.to_datetime(['2026-11-02','2026-11-03','2026-11-04']),
                         open=[10.,10.,10.],close=[10.,10.,10.],high=[11.,11.,11.],low=[9.,9.,9.],
                         volume=100,amount=1000,turnover=1,code='510300',source='Tencent rich kline',name='sample'))
    fresh=old.iloc[1:].copy()
    fresh[['open','close','high','low']]*=.9
    end=fresh.iloc[-1:].copy();end['date']=pd.Timestamp('2026-11-05')
    fresh=pd.concat([fresh,end],ignore_index=True)
    monkeypatch.setattr(data,'_tx_piece',lambda *a,**k:fresh)
    f=data._tencent('510300','2026-11-05','2024-01-01',True,False,old)
    assert len(f)==4 and float(f.iloc[0].close)==9
    assert float(f.iloc[0].amount)==1000
