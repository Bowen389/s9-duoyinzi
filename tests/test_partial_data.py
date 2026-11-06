import ast
import gc
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def isolated_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class PartialDataTests(unittest.TestCase):
    def test_stale_probe_does_not_stop_other_checks(self):
        calls = []
        def history(code, day, index=False):
            calls.append(code)
            return pd.DataFrame([dict(date=pd.Timestamp('2026-11-05' if code == '600519' else day),
                                      source='fixture')])
        data = types.SimpleNamespace(asof_now=lambda: '2026-11-06', history=history,
            _board_catalog=lambda day: {'boards': ['fixture']},
            get_json=lambda *a: {'result': {'data': [1], 'count': 1}})
        run = isolated_function('runner/preflight.py', 'run', {'data': data})
        with patch('builtins.print'):
            warnings = run()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(calls, ['510300', '600519', '000300'])

    def test_failed_initial_probes_do_not_skip_rest(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        seen = []
        def history(code, *args):
            seen.append(code)
            if code in {'000000', '000002', '000004'}:
                raise RuntimeError('symbol unavailable')
            return code
        fn = isolated_function('runner/data.py', 'histories',
            dict(ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed, history=history))
        with patch('builtins.print'):
            frames, errors = fn(['000000', '000001', '000002', '000003', '000004'], '2026-11-06')
        self.assertEqual(set(frames), {'000001', '000003'})
        self.assertEqual(len(seen), 5)

    def stock_fixture(self, bad_count):
        day = '2026-11-06'
        codes = [str(i).zfill(6) for i in range(100)]
        frames = {c: pd.DataFrame([dict(date=pd.Timestamp('2026-11-05' if i < bad_count else day),
                    code=c, close=10., source='fixture')]) for i,c in enumerate(codes)}
        spot = pd.DataFrame({'code6': codes, '成交量': [100]*100, '名称': codes})
        data = types.SimpleNamespace(stock_spot=lambda d: spot,
            histories=lambda *a: (frames, {}), board_mapping=lambda d: pd.DataFrame({'code': codes}),
            history=lambda *a, **k: pd.DataFrame({'date': [pd.Timestamp(day)], 'close': [1.]}),
            next_session=lambda d: '2026-11-09')
        seen = []
        def factors(panel, *args):
            seen.extend(panel.code)
            return panel.assign(score=1.)
        ch = types.SimpleNamespace(VERSIONS={'test': dict(windows=[], weights=[], top_n=1, gate={}, label='test')},
            daily_stats_of=lambda p: p, build_board_day=lambda *a: None, compute_factors=factors,
            build_trend=lambda *a: pd.Series([True], index=[pd.Timestamp(day)]))
        fn = isolated_function('runner/signals.py', 'stock_signals',
            dict(data=data, ch=ch, pd=pd, np=np, gc=gc,
                 causal_piecewise=None, causal_ecology=None))
        return fn, day, seen

    def test_individual_stale_stocks_are_excluded(self):
        fn, day, seen = self.stock_fixture(2)
        with patch('builtins.print'):
            result = fn(day)['test']
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['coverage']['loaded'], 98)
        self.assertNotIn('000000', seen)
        self.assertNotIn('000001', seen)
        self.assertEqual(len(result['coverage']['inactive_errors']), 2)

    def test_widespread_outage_still_blocks_ranking(self):
        fn, day, seen = self.stock_fixture(3)
        with patch('builtins.print'), self.assertRaisesRegex(RuntimeError, 'stock coverage'):
            fn(day)
        self.assertEqual(seen, [])


if __name__ == '__main__':
    unittest.main()
