"""Diagnostic provider probes; strategy scans enforce their own data requirements."""
from . import data


def run():
    day = data.asof_now()
    warnings = []

    def check(label, fn):
        try:
            detail = fn()
            print(f"PASS {label}: {detail}", flush=True)
        except Exception as exc:
            message = f"{label}: {type(exc).__name__}: {exc}"
            warnings.append(message)
            print(f"::warning::{message}", flush=True)

    def history_check(code, index):
        frame = data.history(code, day, index=index)
        last = frame.iloc[-1]
        if str(last.date.date()) != day:
            raise RuntimeError(f"stale: {last.date}; expected {day}")
        return f"{day}; {last['source']}; {len(frame)} rows"

    for label, code, index in [('ETF', '510300', False), ('stock', '600519', False),
                                ('index', '000300', True)]:
        check(f"{label} {code}", lambda c=code, i=index: history_check(c, i))

    def catalog_check():
        catalog = data._board_catalog(day)
        if not catalog['boards']:
            raise RuntimeError('empty board catalog')
        return f"{len(catalog['boards'])} board names"

    def membership_check():
        reply = data.get_json('https://datacenter-web.eastmoney.com/api/data/v1/get', {
            'reportName': 'RPT_F10_CORETHEME_BOARDTYPE',
            'columns': 'SECURITY_CODE,BOARD_NAME,NEW_BOARD_CODE',
            'pageNumber': 1, 'pageSize': 5})
        result = reply.get('result') or {}
        if not result.get('data'):
            raise RuntimeError('no board membership response')
        return f"{result.get('count', len(result['data']))} membership records available"

    check('board catalog', catalog_check)
    check('board membership', membership_check)
    print(f"Preflight complete: {len(warnings)} warnings; independent scans will validate actual inputs",
          flush=True)
    return warnings


if __name__ == '__main__':
    run()
