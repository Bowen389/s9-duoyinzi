"""Verify a repeat scan preserves all dated decisions and paper accounts."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
before = json.loads(Path('/tmp/speed-baseline.json').read_text())
after = json.loads((root/'reports/2026-11-06.json').read_text())
assert after['asof'] == before['asof'] == '2026-11-06'
assert not after['errors'], after['errors']
fields = ('status', 'asof', 'exec_day', 'gate_on', 'picks', 'weights',
          'rebalance', 'target_codes', 'reasons', 'config', 'params')
assert set(before['strategies']) == set(after['strategies'])
for name, old in before['strategies'].items():
    new = after['strategies'][name]
    for key in fields:
        assert old.get(key) == new.get(key), (name, key)
assert before['stock_hits'] == after['stock_hits']
assert before['accounts'] == after['accounts']
for path in (root/'paper').rglob('account.json'):
    old = Path('/tmp/speed-baseline-paper')/path.relative_to(root/'paper')
    assert json.loads(path.read_text()) == json.loads(old.read_text()), path
print('MATCH: all 8 strategies, rankings, gates, weights and paper accounts')
