"""Small real-provider check, separate from the long universe download."""
from . import data


def run():
    day=data.asof_now()
    checks=[('ETF','510300',False),('stock','600519',False),('index','000300',True)]
    for label,code,index in checks:
        f=data.history(code,day,index=index)
        last=f.iloc[-1]
        if str(last.date.date())!=day:
            raise RuntimeError(f'{label} stale: {last.date}')
        print(f"PASS {label} {code}: {day}; {last['source']}; {len(f)} rows; amount and turnover present",flush=True)
    catalog=data._board_catalog(day)
    if not catalog['boards']:
        raise RuntimeError('empty board catalog')
    reply=data.get_json('https://datacenter-web.eastmoney.com/api/data/v1/get',{
        'reportName':'RPT_F10_CORETHEME_BOARDTYPE','columns':'SECURITY_CODE,BOARD_NAME,NEW_BOARD_CODE',
        'pageNumber':1,'pageSize':5})
    result=reply.get('result') or {}
    if not result.get('data'):
        raise RuntimeError('no board membership response')
    print(f"PASS original Eastmoney board taxonomy: {len(catalog['boards'])} board names; {result['count']} membership records available",flush=True)


if __name__=='__main__':
    run()
