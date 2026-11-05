"""Validated vendor data with rate limits, fail-fast probes and resumable caches."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import urlparse
import json
import time
import requests
import numpy as np
import pandas as pd
import exchange_calendars as xc

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)
_LOCK = Lock()
_HOSTS = {}


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
    host = urlparse(url).netloc
    err = None
    for i in range(2):
        with _LOCK:
            state = _HOSTS.setdefault(host, {'fails': 0, 'blocked_until': 0., 'next': 0.})
            now = time.monotonic()
            if state['blocked_until'] > now:
                raise RuntimeError(f'{host}: circuit open after vendor connection/rate-limit errors; batch skipped')
            delay = max(0., state['next'] - now)
            state['next'] = max(now, state['next']) + .25
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, params=params, timeout=(5, 12),
                             headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.qq.com/'})
            r.raise_for_status()
            j = r.json()
            with _LOCK:
                state['fails'] = 0
            return j
        except Exception as e:
            err = e
            with _LOCK:
                state['fails'] += 1
                if state['fails'] >= 3 or getattr(getattr(e, 'response', None), 'status_code', None) == 429:
                    # Respect rate limits: do not rotate Eastmoney subdomains to bypass a ban.
                    state['blocked_until'] = time.monotonic() + 3600
            if i == 0:
                time.sleep(.4)
    raise RuntimeError(f'{host}: {type(err).__name__}: {err}')


def _tx_piece(code, day, start, adjusted, index):
    symbol = ('sh' if index or code.startswith(('5', '6', '9')) else 'sz') + code
    j = get_json('https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get',
                 {'param': f'{symbol},day,{start},{day},640,{"qfq" if adjusted and not index else ""}'})
    d = (j.get('data') or {}).get(symbol) or {}
    rows = d.get('qfqday') if adjusted and not index else d.get('day')
    if rows is None:
        rows = d.get('day', [])
    # The old fqkline endpoint returns only six fields. Do not invent amount/turnover.
    if not rows or any(len(x) < 9 for x in rows):
        raise RuntimeError(f'{code}: Tencent rich kline missing amount/turnover')
    df = pd.DataFrame([{ 'date': x[0], 'open': x[1], 'close': x[2], 'high': x[3],
                         'low': x[4], 'volume': x[5], 'turnover': x[7], 'amount': x[8] } for x in rows])
    df['date'] = pd.to_datetime(df.date)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Tencent volume: lots except STAR shares. Normalize to the EM lot unit used by source.
    if code.startswith(('688', '689')):
        df['volume'] /= 100.
    df['amount'] *= 10000.  # Tencent 10,000 CNY -> CNY; turnover already in percent.
    qt = (d.get('qt') or {}).get(symbol) or []
    df['name'] = qt[1] if len(qt)>1 else code
    df['code'], df['source'] = code, 'Tencent rich kline'
    return df


def _tencent(code, day, start, adjusted, index, old=None):
    # ETF live decisions only need 120-session age plus <=60-session features. 640 bars
    # retain an ample warmup without downloading ten years for 1,777 assets each run.
    etf = code.startswith(('15', '51', '52', '56', '58')) and not index
    if old is not None and len(old) and old.date.max() < pd.Timestamp(day):
        overlap_start = str((old.date.max()-pd.Timedelta(days=40)).date())
        fresh = _tx_piece(code, day, overlap_start, adjusted, index)
        overlap = old[['date','close']].merge(fresh[['date','close']],on='date',suffixes=('_old','_new'))
        if len(overlap):
            ratio = overlap.close_new / overlap.close_old
            if np.isfinite(ratio).all() and ratio.max()/ratio.min() < 1.0002:
                prior = old[old.date < fresh.date.min()].copy()
                if adjusted and not index:
                    prior[['open','close','high','low']] *= float(ratio.median())
                return pd.concat([prior,fresh],ignore_index=True).drop_duplicates('date',keep='last')
    if etf or index:
        df = _tx_piece(code, day, start, adjusted, index)
    else:
        # Preserve the uploaded stock factor's 2024-start expanding turnover bounds.
        parts=[]
        for year in range(pd.Timestamp(start).year,pd.Timestamp(day).year+1,2):
            lo=max(pd.Timestamp(start),pd.Timestamp(f'{year}-01-01'))
            hi=min(pd.Timestamp(day),pd.Timestamp(f'{year+1}-12-31'))
            parts.append(_tx_piece(code,str(hi.date()),str(lo.date()),adjusted,index))
        df=pd.concat(parts,ignore_index=True).drop_duplicates('date',keep='last')
    return df


def _eastmoney(code, day, start, adjusted, index):
    market = '1' if index or code.startswith(('5', '6', '9')) else '0'
    j = get_json('https://push2his.eastmoney.com/api/qt/stock/kline/get', {
        'secid': f'{market}.{code}', 'klt':101,'fqt':1 if adjusted and not index else 0,
        'beg':start.replace('-',''),'end':day.replace('-',''),'fields1':'f1,f2,f3',
        'fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'})
    d=j.get('data') or {}
    rows=[x.split(',') for x in d.get('klines',[])]
    if not rows:
        raise RuntimeError(f'{code}: no Eastmoney history')
    df=pd.DataFrame(rows,columns=['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover'])
    df['date']=pd.to_datetime(df.date)
    for c in df.columns[1:]:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df['code'],df['name'],df['source']=code,d.get('name',code),'Eastmoney'
    return df


def history(code, day, adjusted=True, index=False, start='2024-01-01'):
    folder=CACHE/('index' if index else ('qfq' if adjusted else 'raw'))/start
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/f'{code}.parquet'
    old=pd.read_parquet(p) if p.exists() else None
    if old is not None:
        old=old[old.date<=pd.Timestamp(day)]
        if len(old) and str(old.date.max().date())==day and 'source' in old:
            return old
    errors=[]
    for fn in [lambda:_tencent(code,day,start,adjusted,index,old),lambda:_eastmoney(code,day,start,adjusted,index)]:
        try:
            df=fn().sort_values('date').reset_index(drop=True)
            df=df[(df.date>=pd.Timestamp(start)) & (df.date<=pd.Timestamp(day))].copy()
            if df.empty or df.date.duplicated().any() or not np.isfinite(df[['open','high','low','close','volume','amount','turnover']]).all().all():
                raise RuntimeError('invalid/missing OHLCV, turnover or amount')
            if (df.close<=0).any() or (df.amount<0).any():
                raise RuntimeError('nonpositive close or negative amount')
            if 'pct' not in df:
                prev=df.close.shift(1)
                df['pct']=df.close.pct_change()*100
                df['amplitude']=(df.high-df.low)/prev*100
                df['change']=df.close.diff()
                # First row is warmup, never used for live signals/execution.
                df.loc[df.index[0],['pct','amplitude','change']]=0.
            tmp=p.with_suffix('.tmp.parquet');df.to_parquet(tmp,index=False);tmp.replace(p)
            return df
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(f'{code}: ' + ' | '.join(errors))


def histories(codes, day, adjusted=True, start='2024-01-01', workers=3):
    codes=sorted(set(codes));frames,errors={},{}
    if not codes:
        return frames,errors
    # Probe before enqueuing thousands of calls during an upstream outage.
    probe=list(dict.fromkeys([c for c in ['510300','159915','600519'] if c in codes] + [codes[0],codes[len(codes)//2],codes[-1]]))[:min(3,len(codes))]
    rest=[c for c in codes if c not in probe]
    def collect(batch):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            todo={ex.submit(history,c,day,adjusted,False,start):c for c in batch}
            for i,f in enumerate(as_completed(todo),1):
                c=todo[f]
                try:
                    frames[c]=f.result()
                except Exception as e:
                    errors[c]=str(e)
                if i%100==0:
                    print(f'data {len(frames)+len(errors)}/{len(codes)}, failures={len(errors)}',flush=True)
    collect(probe)
    if len(errors)==len(probe):
        for c in rest:
            errors[c]='provider probe failed; batch skipped without repeated downloads'
        print(f'provider probe failed; skipped {len(rest)} requests',flush=True)
    else:
        collect(rest)
    print(f'data complete: {len(frames)}/{len(codes)}, errors={len(errors)}',flush=True)
    return frames,errors


def _sina_spot():
    base='https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.'
    count=int(get_json(base+'getHQNodeStockCount',{'node':'hs_a'}))
    def page(i):
        result=get_json(base+'getHQNodeData',{'page':i,'num':100,'sort':'symbol','asc':1,'node':'hs_a','symbol':'','_s_r_a':'page'})
        if not isinstance(result,list):
            raise RuntimeError('Sina malformed snapshot page')
        return result
    with ThreadPoolExecutor(max_workers=3) as ex:
        parts=list(ex.map(page,range(1,(count+99)//100+1)))
    raw=pd.DataFrame([r for part in parts for r in part]).drop_duplicates('code')
    if len(raw)<count*.99:
        raise RuntimeError(f'Sina snapshot incomplete {len(raw)}/{count}')
    spot=raw.rename(columns={'code':'代码','name':'名称','trade':'最新价','volume':'成交量','amount':'成交额',
                             'turnoverratio':'换手率','mktcap':'总市值'})
    for c in ['最新价','成交量','成交额','换手率','总市值']:
        spot[c]=pd.to_numeric(spot[c],errors='coerce')
    spot['总市值']*=10000.  # Sina market cap is 10,000 CNY.
    spot['source']='Sina'
    return spot


def stock_spot(day):
    p=CACHE/f'spot_{day}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    if day!=asof_now():
        raise RuntimeError('cannot backdate live market capitalization snapshot')
    try:
        spot=_sina_spot()
    except Exception as e:
        print(f'Sina snapshot unavailable: {e}; trying Eastmoney once',flush=True)
        import akshare as ak
        spot=ak.stock_zh_a_spot_em()
        spot['source']='Eastmoney'
    spot['代码']=spot['代码'].astype(str).str.zfill(6)
    spot=spot[~spot['名称'].str.contains('ST|退',case=False,na=False)]
    spot=spot[spot['代码'].str[:2].isin(['60','00','30','68']) & (spot['最新价']>2)]
    spot=spot.drop_duplicates('代码').sort_values(['总市值','代码'],ascending=[False,True])
    if len(spot)<1600 or spot['总市值'].isna().any():
        raise RuntimeError('incomplete stock snapshot/capitalization')
    spot['code6']=spot['代码']
    spot.to_parquet(p,index=False)
    return spot


def _board_catalog(day):
    cache=CACHE/f'board_catalog_{day}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    rows=[]
    try:
        for kind,typ in [('ind',2),('con',3)]:
            page=1
            while True:
                d=get_json('https://17.push2.eastmoney.com/api/qt/clist/get',{
                    'pn':page,'pz':100,'np':1,'fltt':2,'fid':'f12','po':0,
                    'fs':f'm:90 t:{typ} f:!50','fields':'f12,f14'})['data']
                rows.extend(dict(code=x['f12'],name=x['f14'],kind=kind) for x in d['diff'])
                if page*100>=d['total']:
                    break
                page+=1
        catalog=dict(asof=day,source='Eastmoney industry/concept catalog',boards=rows)
        cache.write_text(json.dumps(catalog,ensure_ascii=False))
        return catalog
    except Exception as e:
        paths=sorted(CACHE.glob('board_catalog_*.json'),reverse=True)+[ROOT/'strategies/board_catalog.json']
        for p in paths:
            c=json.loads(p.read_text())
            age=(pd.Timestamp(day)-pd.Timestamp(c['asof'])).days
            if 0<=age<=7:
                print(f'Board name catalog fallback dated {c["asof"]}; membership still refreshed for {day}: {e}',flush=True)
                return c
        raise RuntimeError('No recent verified Eastmoney board name catalog') from e


def board_mapping(day):
    p=CACHE/f'board_mapping_{day}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    catalog=_board_catalog(day)
    valid={x['code']:x for x in catalog['boards']}
    url='https://datacenter-web.eastmoney.com/api/data/v1/get'
    base={'reportName':'RPT_F10_CORETHEME_BOARDTYPE','columns':'SECURITY_CODE,BOARD_NAME,NEW_BOARD_CODE',
          'pageSize':5000,'sortColumns':'SECURITY_CODE,NEW_BOARD_CODE','sortTypes':'1,1'}
    folder=CACHE/'board_pages'/day
    folder.mkdir(parents=True,exist_ok=True)
    def page(i):
        q=folder/f'{i}.json'
        if q.exists():
            return json.loads(q.read_text())
        j=get_json(url,{**base,'pageNumber':i})
        d=j.get('result')
        if not d or not d.get('data'):
            raise RuntimeError(f'board membership page {i} missing')
        q.write_text(json.dumps(d,ensure_ascii=False))
        return d
    first=page(1)
    with ThreadPoolExecutor(max_workers=3) as ex:
        pieces=[first]+list(ex.map(page,range(2,first['pages']+1)))
    rows=[r for part in pieces for r in part['data']]
    if len(rows)!=first['count']:
        raise RuntimeError(f'Incomplete board membership {len(rows)}/{first["count"]}')
    mapping=pd.DataFrame([dict(code=str(r['SECURITY_CODE']).zfill(6),board=valid[r['NEW_BOARD_CODE']]['name'],
                               kind=valid[r['NEW_BOARD_CODE']]['kind']) for r in rows if r['NEW_BOARD_CODE'] in valid])
    if mapping.empty:
        raise RuntimeError('No matched Eastmoney industry/concept membership')
    mapping=mapping.drop_duplicates(['code','board','kind'])
    mapping['catalog_asof']=catalog['asof']
    mapping.to_parquet(p,index=False)
    print(f'Board membership complete: {len(mapping)} rows, {mapping.code.nunique()} securities; catalog {catalog["asof"]}',flush=True)
    return mapping


def execution_bars(codes,day):
    frames,errors=histories(codes,day,adjusted=False,start=str((pd.Timestamp(day)-pd.Timedelta(days=40)).date()))
    bars={}
    for c,f in frames.items():
        row=f[f.date==pd.Timestamp(day)]
        if row.empty:
            continue
        row=row.iloc[-1]
        # Tencent raw kline has no exchange ex-rights reference. Detect corporate action
        # separately against adjusted series instead of using raw pct to infer it.
        past=f[f.date<pd.Timestamp(day)]
        prev=float(past.iloc[-1].close) if len(past) else float(row.close)
        if row['source']=='Eastmoney':
            prev=float(row.close)/(1+float(row.pct)/100)
        else:
            try:
                adj=history(c,day,adjusted=True,start=str((pd.Timestamp(day)-pd.Timedelta(days=40)).date()))
                before=adj[adj.date<pd.Timestamp(day)]
                today=adj[adj.date==pd.Timestamp(day)]
                if len(before) and len(today):
                    prev=float(before.iloc[-1].close)*float(row.close)/float(today.iloc[-1].close)
            except Exception as e:
                errors[c]=f'corporate/limit reference unavailable: {e}'
                continue
        bars[c]={k:float(row[k]) for k in ['open','close','high','low','volume','amount']}
        bars[c]['prev']=prev
        bars[c]['previous_raw_close']=float(past.iloc[-1].close) if len(past) else prev
    return bars,errors
