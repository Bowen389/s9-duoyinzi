import argparse
import csv
import io
import json
import traceback
from pathlib import Path
from . import data
from .broker import STOCKS, ETFS, START, initial, settle, equity


def write_json(p, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    s=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(s,encoding='utf-8')
    tmp.replace(p)


def load(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def export_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def save_accounts(accounts):
    for name,a in accounts.items():
        p=data.ROOT/'paper'/name
        write_json(p/'account.json',a)
        export_csv(p/'trades.csv',a['trades'])
        export_csv(p/'nav.csv',a['nav'])
        export_csv(p/'events.csv',a['events'])


def run(day, family='all'):
    from strategies.champion_source import VERSIONS
    from .signals import stock_signals, etf_signals
    names=STOCKS+ETFS
    accounts={n:load(data.ROOT/'paper'/n/'account.json',initial(n)) for n in names}
    all_days=data.sessions('2026-11-05',str((__import__('pandas').Timestamp(day)+__import__('pandas').Timedelta(days=25)).date()))
    # Settle every missed session using only signals that were actually committed beforehand.
    errors={}
    selected=names if family=='all' else (STOCKS if family=='stock' else ETFS)
    for d in [x for x in all_days if START<=x<=day]:
        todo=[n for n in selected if (not accounts[n]['last_settled'] or accounts[n]['last_settled']<d) and n not in errors]
        needed=set()
        pending={}
        for n in todo:
            previous=all_days[all_days.index(d)-1]
            sig=load(data.ROOT/'signals'/previous/(n+'.json'))
            pending[n]=sig
            needed.update(x['code'] for x in accounts[n]['lots'])
            if sig and sig['status']=='ok':
                needed.update(x['code'] for x in sig.get('picks',[]) if n in STOCKS)
                needed.update(sig.get('weights',{}))
        bars, bar_errors=data.execution_bars(needed,d)
        if bar_errors:
            # Refuse to consume a session on an outage. Other accounts can still settle independently.
            errors['execution_'+d]=str(bar_errors)
        for n in todo:
            req=set(x['code'] for x in accounts[n]['lots'])
            sig=pending[n]
            if sig and sig['status']=='ok':
                req.update(x['code'] for x in sig.get('picks',[]) if n in STOCKS)
                req.update(sig.get('weights',{}))
            if req & set(bar_errors):
                errors[n]='execution data incomplete; settlement deferred'
                continue
            if not sig:
                accounts[n]['events'].append(dict(date=d,event='SIGNAL_MISSING',reason='no prior committed signal; no retrospective buys'))
            accounts[n]=settle(accounts[n],d,sig,bars,all_days,VERSIONS.get(n))
        save_accounts(accounts)
    results={}
    for group,fn in [('stock',lambda:stock_signals(day)),('etf',lambda:etf_signals(day,accounts))]:
        if family not in ('all',group):
            continue
        members=STOCKS if group=='stock' else ETFS
        try:
            stale=[n for n in members if day>=START and accounts[n]['last_settled']!=day]
            if stale:
                raise RuntimeError(f'unsettled accounts: {stale}')
            results.update(fn())
        except Exception as e:
            traceback.print_exc()
            errors[group]=str(e)
            for n in members:
                results[n]=dict(strategy=n,status='data_error',asof=day,exec_day=data.next_session(day),error=str(e),picks=[])
    for n,s in results.items():
        write_json(data.ROOT/'signals'/day/(n+'.json'),s)
    save_accounts(accounts)
    # Reports always reload all families to support safe incremental recovery.
    results={n:load(data.ROOT/'signals'/day/(n+'.json'),dict(status='pending',strategy=n,asof=day)) for n in names}
    report=dict(asof=day,generated_at=str(__import__('pandas').Timestamp.now(tz='Asia/Shanghai')),
                start=START,strategies=results,accounts={n:dict(cash=a['cash'],equity=equity(a),positions=a['lots'],last_settled=a['last_settled'],
                return_pct=(equity(a)/200000-1)*100,corporate_review=a.get('corporate_review',[])) for n,a in accounts.items()},errors=errors)
    write_json(data.ROOT/'reports'/f'{day}.json',report)
    write_json(data.ROOT/'reports'/'latest.json',report)
    lines=[f'# 盘后信号 · {day}',f'\n模拟开始日：{START}；8 个独立账户，每户初始 200,000 元。次一交易日按真实开盘价记账。',
           '\n|策略|状态/门控|次日计划|现金|总资产|收益|','|---|---|---|---:|---:|---:|']
    for n,s in results.items():
        a=accounts[n]
        if s['status']!='ok':
            status=s['status']; plan=s.get('error','等待扫描')
        else:
            status='进攻开' if s['gate_on'] else '进攻关'
            if n in STOCKS:
                plan='、'.join(x['code']+' '+x['name'] for x in s['picks']) if s['gate_on'] else '不开新仓；已有仓位按到期规则退出'
            elif s['rebalance']:
                plan='；'.join(f'{c} {w:.1%}' for c,w in s['weights'].items()) or '转现金'
            else:
                plan='维持当前持仓（空仓则继续空仓）'
        if a.get('corporate_review'):
            status += '；除权核验待处理'
        lines.append(f"|{n}|{status}|{plan.replace('|','/')}|{a['cash']:.2f}|{equity(a):.2f}|{(equity(a)/200000-1)*100:+.2f}%|")
    lines.append('\n## 每日 Top 候选（无论门控是否通过均展示）')
    for n,s in results.items():
        lines.append(f'\n### {n}')
        if s['status'] != 'ok':
            lines.append('\n当日排名暂缺：' + s.get('error', '等待扫描') + '。不使用旧日排名替代。')
            continue
        lines.append('\n' + ('门控通过；是否买入仍以账户操作计划为准。' if s['gate_on'] else '门控未通过：以下仅为观察排名，不构成进攻买入信号。'))
        picks = s.get('picks', [])
        if not picks:
            lines.append('\n当日没有通过个股/ETF基础筛选的候选。')
            continue
        lines += ['\n|排名|代码|名称|得分|参考收盘价|', '|---:|---|---|---:|---:|']
        for rank,x in enumerate(picks, 1):
            ref = f"{x['ref_close']:.4f}" if x.get('ref_close') is not None else '—'
            name = str(x.get('name', '')).replace('|', '/')
            lines.append(f"|{rank}|{x['code']}|{name}|{x['score']:.4f}|{ref}|")
    lines+=['\n参考排名、门控、目标权重与逐笔交易见对应 JSON 和 paper 目录。缺数据时暂停生成该策略买单，不回退旧日期。',
            '\n股票选股范围严格按上传包全 A 非 ST（含创业板、科创板）；没有替换为主板。模拟账户不继承历史回测持仓。']
    text='\n'.join(lines)+'\n'
    (data.ROOT/'reports'/f'{day}.md').write_text(text)
    (data.ROOT/'reports'/'latest.md').write_text(text)
    print(text,flush=True)
    return any(s['status']!='ok' for s in results.values())


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--date')
    ap.add_argument('--family',choices=['all','stock','etf'],default='all')
    args=ap.parse_args()
    raise SystemExit(2 if run(args.date or data.asof_now(),args.family) else 0)
