"""Prove that an unavailable ETF cannot yet satisfy the strategy's 120-session age rule."""
from concurrent.futures import ThreadPoolExecutor
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from . import data


def classify_profile(html, code, day, min_age=120):
    soup=BeautifulSoup(html,'html.parser')
    fields={}
    for th in soup.find_all('th'):
        td=th.find_next_sibling('td')
        if td:
            fields[th.get_text('',strip=True)]=td.get_text(' ',strip=True)
    if code not in fields.get('基金代码',''):
        return None
    full=fields.get('基金全称','')
    if '交易型开放式' not in full and 'ETF' not in full.upper():
        return None
    def date_of(text):
        m=re.search(r'(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})',text)
        return pd.Timestamp(year=int(m[1]),month=int(m[2]),day=int(m[3])) if m else None
    issue=date_of(fields.get('发行日期',''))
    established=date_of(fields.get('成立日期/规模',''))
    # Issuance is an earlier bound than formation/listing, so this is an upper
    # bound on possible trading age. Never infer ineligibility from a failed quote.
    known_dates=[d for d in [issue,established] if d is not None]
    earliest=min(known_dates) if known_dates else None
    if earliest is None or earliest<pd.Timestamp('2024-01-01'):
        return None
    age=0 if earliest>pd.Timestamp(day) else len(data.sessions(str(earliest.date()),day))
    if age>=min_age:
        return None
    return dict(code=code,name=fields.get('基金简称',full),verified_asof=day,
                issue_date=str(issue.date()) if issue is not None else None,
                establishment_date=str(established.date()) if established is not None else None,
                maximum_possible_trading_sessions=age,required_sessions=min_age,
                reason='发行/成立至今不足策略要求的交易日，尚不可能满足上市年龄过滤',
                source_url=f'https://fundf10.eastmoney.com/jbgk_{code}.html')


def classify_unavailable(codes, day, protected=(), min_age=120):
    folder=data.CACHE/'etf_eligibility'/day
    folder.mkdir(parents=True,exist_ok=True)
    def one(c):
        if c in protected:
            return c,None
        p=folder/f'{c}.json'
        if p.exists():
            return c,json.loads(p.read_text())
        url=f'https://fundf10.eastmoney.com/jbgk_{c}.html'
        for attempt in range(2):
            try:
                r=requests.get(url,timeout=(8,20),headers={'User-Agent':'Mozilla/5.0'})
                r.raise_for_status()
                r.encoding=r.apparent_encoding
                proof=classify_profile(r.text,c,day,min_age)
                if proof:
                    p.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
                return c,proof
            except Exception as e:
                if attempt==1:
                    print(f'ETF age metadata unavailable {c}: {e}',flush=True)
        return c,None
    with ThreadPoolExecutor(max_workers=4) as ex:
        results=dict(ex.map(one,sorted(set(codes))))
    return {c:p for c,p in results.items() if p is not None}
