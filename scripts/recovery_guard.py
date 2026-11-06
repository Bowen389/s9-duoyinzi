"""Select only unfinished strategy families for post-close recovery."""
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

FAMILIES = {'stock': ('champion','balanced','runner','original','wild'),
            'etf': ('B0','K3','K5')}


def recovery_family(now, report, signals, accounts):
    now = now.astimezone(ZoneInfo('Asia/Shanghai'))
    if now.weekday() >= 5 or now.time() < dt.time(15, 10):
        return 'skip'
    day = now.date().isoformat()
    try:
        stamp = dt.datetime.fromisoformat(report.get('generated_at', ''))
        fresh = (stamp.tzinfo is not None and stamp.astimezone(now.tzinfo).date() == now.date()
                 and stamp.astimezone(now.tzinfo).time() >= dt.time(16, 30)
                 and report.get('asof') == day)
    except (ValueError, TypeError):
        fresh = False
    missing = []
    for family, names in FAMILIES.items():
        if not fresh or any(signals.get(n, {}).get('asof') != day
                            or signals.get(n, {}).get('status') != 'ok'
                            or accounts.get(n, {}).get('last_settled') != day for n in names):
            missing.append(family)
    return 'all' if len(missing) == 2 else missing[0] if missing else 'skip'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    now = dt.datetime.now(ZoneInfo('Asia/Shanghai'))
    day = now.date().isoformat()
    names = sum((list(v) for v in FAMILIES.values()), [])
    print(recovery_family(now, read(root/'reports'/f'{day}.json'),
          {n: read(root/'signals'/day/f'{n}.json') for n in names},
          {n: read(root/'paper'/n/'account.json') for n in names}))
