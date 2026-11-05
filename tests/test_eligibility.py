from runner import eligibility as e


def profile(issue='2026年10月12日',established='',code='158016'):
    return f'''<table><tr><th>基金代码</th><td>{code}（前端）</td>
      <th>基金全称</th><td>测试交易型开放式指数证券投资基金</td></tr>
      <tr><th>基金简称</th><td>测试ETF</td><th>发行日期</th><td>{issue}</td></tr>
      <tr><th>成立日期/规模</th><td>{established}</td></tr></table>'''


def test_recent_issuance_proves_age_ineligible():
    p=e.classify_profile(profile(),'158016','2026-11-05')
    assert p and p['maximum_possible_trading_sessions']<120
    assert p['issue_date']=='2026-10-12'


def test_old_fund_or_unknown_metadata_cannot_be_excluded():
    assert e.classify_profile(profile(issue='2016年01月01日'),'158016','2026-11-05') is None
    assert e.classify_profile('<h1>timeout</h1>','158016','2026-11-05') is None
    assert e.classify_profile(profile(code='999999'),'158016','2026-11-05') is None


def test_older_establishment_overrides_recent_issue_for_conservatism():
    assert e.classify_profile(profile(established='2018年01月01日 / 1亿'),'158016','2026-11-05') is None


def test_protected_core_or_held_etf_never_excluded(monkeypatch,tmp_path):
    monkeypatch.setattr(e.data,'CACHE',tmp_path)
    def forbidden(*a,**kw):
        raise AssertionError('protected ETF must not be classified away')
    monkeypatch.setattr(e.requests,'get',forbidden)
    assert e.classify_unavailable(['158016'],'2026-11-05',protected={'158016'})=={}
