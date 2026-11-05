#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s9_live.py — S9 策略单文件自包含执行器 (最小可执行单元)
=========================================================
用法(仅依赖同级 data/ 目录):
    python3 s9_live.py                 # K5 档: 生成当日指令 → stdout 摘要 + orders_K5_<asof>.json
    python3 s9_live.py K3              # K3 档      (K3 / K5 / B0)
    python3 s9_live.py --fetch         # 先联网追加 klines 再出单
    python3 s9_live.py --verify        # 金标准自检 (B0@W2, 验证本文件内联引擎与权威源一致)
    python3 s9_live.py K5 --json       # 额外把完整指令 JSON 打到 stdout (供管道/程序消费)

依赖: python3 + numpy pandas scipy pyarrow (缺失会自动 pip 安装并重执行) + data/ 目录 (klines csv)。
语义: 与 S9_orig_backtest.py 逐行同源 (本文件由其生成, 生成日 2026-11-05)。
⚠ 双副本提示: 引擎若改, 用 --verify 复检本文件 (金标准 total=0.3332458213558225 / sharpe_rf0=0.5730812747686882)。
"""
import argparse, json, os, subprocess, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def ensure_env():
    """缺失依赖自愈: 装齐后 re-exec (避免同进程热装冲突)。"""
    missing = []
    for mod, pip in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("pyarrow", "pyarrow")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    if not missing:
        return
    if os.environ.get("S9_REEXEC"):
        print("[env] still missing after install:", missing, flush=True)
        return
    print("[env] install + re-exec:", missing, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=True)
    env = dict(os.environ, S9_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

ensure_env()


# -*- coding: utf-8 -*-
"""
S9 ETF 趋势轮动策略 —— 单文件可回测版
=================================================================
这份文件把整个 S9 策略（数据面板 + 信号 + 回测引擎 + 参数）压成一个.py，
除 numpy/pandas/scipy 外无任何本项目依赖，可以直接拿去跑、改、对照。

数据要求（放在同目录 ./data/ 下，或改 DATA 变量）：
  etf_daily.parquet   列: code,date,name,open,high,low,close,volume,amount
  benchmark.parquet   列: date,close           (沪深300)
  cash_extra.parquet  列: code,date,name,close  (至少含 511880 银华日利，作现金收益代理)

用法：
  python S9_单文件回测.py                     # 跑 S9，打印指标 + 存台账
  python S9_单文件回测.py --start 2024-01-01   # 只跑样本外
  python S9_单文件回测.py --cost 0.0005        # 改成本做压力测试
  python S9_单文件回测.py --params my.json     # 从 json 载入参数

输出：results_single/S9_净值.csv / S9_逐笔成交.csv / S9_指标.json

=================================================================
S9 规则速查（9 组参数）
-----------------------------------------------------------------
池子 : 静态核心池（2019-12-31 前累计成交额前150，实得134只，冻结）
        ∪ 动态池（近5日日均成交额>5000万的前100只）
        − 货币类 − 上市<120日 − 当日成交额<300万 − 同底层指数去重
趋势 : close > MA20 > MA60
打分 : 40日指数衰减(0.05)加权对数回归 → 斜率年化 exp(s*252)-1，clip(-100%,2000%)，再乘 R²
       → 取前 3 名等权
放量 : 当日成交额 > 前5日均额 × 2.0 → 不买 / 已持有则卖出
卖出 : 止损-10%(收盘确认) > 放量 > 趋势破位 > 掉出前三
       迟滞: 新标的得分需 > 持仓中最低分 × 1.15
       最短持仓5日、冷静期5日、权重偏离>2%才调平
择时 : 沪深300 > MA60  或  全池多头排列占比(5日平滑) ≥ 25% → 才持仓
       否则清仓，转入 国债ETF 511260 + 黄金ETF 518880 各半
执行 : T日收盘出信号，T+1开盘成交；成本单边 1bp；闲置现金按货币ETF计息
=================================================================
"""
import os, re, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


def _find_data():
    """数据目录：优先 ./data，其次 ../data（本文件放在子目录时）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "data"), os.path.join(os.path.dirname(here), "data")):
        if os.path.exists(os.path.join(c, "etf_daily.parquet")):
            return c
    return os.path.join(here, "data")


DATA = _find_data()

# ============ 1. 数据面板 ============


def load_panel():
    daily = pd.read_parquet(os.path.join(DATA, "etf_daily.parquet"))
    extra = pd.read_parquet(os.path.join(DATA, "cash_extra.parquet"))
    bm = pd.read_parquet(os.path.join(DATA, "benchmark.parquet"))

    daily = pd.concat([daily, extra], ignore_index=True)
    daily = daily.drop_duplicates(["code", "date"], keep="last")

    cal = pd.DatetimeIndex(sorted(bm["date"].unique()))
    fields = {}
    for f in ["close", "open", "high", "low", "amount", "volume"]:
        w = daily.pivot_table(index="date", columns="code", values=f, aggfunc="last")
        w = w.reindex(cal)
        fields[f] = w
    close = fields["close"]
    # 缺失收盘价：最多前向填充 5 个交易日（停牌/无成交），超出视为不可交易
    close = close.ffill(limit=5)
    for f in ["open", "high", "low"]:
        fields[f] = fields[f].ffill(limit=5)
    fields["close"] = close

    codes = close.columns.to_numpy()
    names = daily.drop_duplicates("code", keep="last").set_index("code")["name"].to_dict()

    cash = extra[extra.code == "511880"].set_index("date")["close"].reindex(cal).ffill()
    cash_ret = cash.pct_change().fillna(0.0).clip(-0.05, 0.05).to_numpy()

    bench = bm.set_index("date")["close"].reindex(cal).ffill()

    return dict(
        dates=cal,
        codes=codes,
        names=names,
        close=close.to_numpy(dtype=np.float64),
        open=fields["open"].to_numpy(dtype=np.float64),
        high=fields["high"].to_numpy(dtype=np.float64),
        low=fields["low"].to_numpy(dtype=np.float64),
        amount=fields["amount"].fillna(0).to_numpy(dtype=np.float64),
        bench=bench,
        cash_ret=cash_ret,
    )


def static_core_pool(panel, asof="2019-12-31", top_n=150, min_days=120, min_avg_amt=0.0):
    """静态核心池：在回测起始日之前，用成交额排名冻结一份名单（不使用未来信息）。"""
    dates = panel["dates"]
    mask = dates <= pd.Timestamp(asof)
    amt = panel["amount"][mask]
    close = panel["close"][mask]
    days = np.isfinite(close).sum(axis=0)
    tot = np.nansum(amt, axis=0)
    ok = (days >= min_days) & (tot / np.maximum(days, 1) >= min_avg_amt)
    order = np.argsort(-tot)
    picked = [panel["codes"][i] for i in order if ok[i]][:top_n]
    return picked

# ============ 2. 信号 ============
def _kernel(L, decay):
    j = np.arange(L, dtype=np.float64)          # 0 = 最旧, L-1 = 最新
    if decay > 0:
        w = np.exp(decay * (j - (L - 1)))       # 越新权重越大
    else:
        w = np.ones(L)
    w = w / w.sum()
    x = j - (w * j).sum()
    return w, x


def rolling_wls_momentum(close, L=25, decay=0.05, ann=252.0):
    """返回 (年化斜率, R²)，形状与 close 相同，前 L-1 行为 NaN。"""
    T, N = close.shape
    out_slope = np.full((T, N), np.nan)
    out_r2 = np.full((T, N), np.nan)
    if T < L:
        return out_slope, out_r2

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(close)
    valid = np.isfinite(z)
    z0 = np.where(valid, z, 0.0)

    w, x = _kernel(L, decay)
    W = w.sum()
    A2 = (w * x * x).sum()

    def conv(mat):
        # 沿时间轴与核做卷积，取 valid 部分，长度 T-L+1
        return fftconvolve(mat, w[::-1][:, None], mode="valid", axes=0)

    S_w = conv(z0)                     # Σ w z
    S_wx = fftconvolve(z0, (w * x)[::-1][:, None], mode="valid", axes=0)
    S_wz2 = conv(z0 * z0)
    cnt = conv(valid.astype(np.float64))

    S_w, S_wx, S_wz2 = S_w[: T - L + 1], S_wx[: T - L + 1], S_wz2[: T - L + 1]
    cnt = cnt[: T - L + 1]

    sst = S_wz2 - S_w ** 2 / W
    slope = S_wx / A2
    r2 = np.where(sst > 1e-12, (slope ** 2) * A2 / np.maximum(sst, 1e-12), 0.0)
    r2 = np.clip(r2, 0.0, 1.0)
    ok = cnt >= 0.999          # 核已归一化，窗口内无缺失时卷积值为 1
    slope = np.where(ok, slope, np.nan)
    r2 = np.where(ok, r2, np.nan)

    out_slope[L - 1:] = np.exp(np.clip(slope * ann, -5, 5)) - 1.0
    out_r2[L - 1:] = r2
    return out_slope, out_r2


def sma(x, n):
    """忽略 NaN 的简单移动平均（窗口内有效值足够即可）。"""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < n:
        return out
    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)
    k = np.ones(n)
    s = fftconvolve(x0, k[:, None], mode="valid", axes=0)[: T - n + 1]
    c = fftconvolve(valid.astype(float), k[:, None], mode="valid", axes=0)[: T - n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(c >= n * 0.9, s / np.maximum(c, 1), np.nan)
    out[n - 1:] = v
    return out


def shift(x, k):
    """向后平移 k 天（避免未来信息）。"""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:] = x
    return out

# ============ 3. 回测引擎 ============
DEFAULT = dict(
    # ---- 股票池 ----
    static_pool_size=150,
    static_min_amt=0.0,        # 静态池：上市以来日均成交额门槛
    dyn_min_amt=5e7,           # 动态池：5 日均成交额门槛
    dyn_top=100,
    dyn_win=5,
    static_asof="2019-12-31",
    # ---- 趋势 ----
    ma_fast=20,
    ma_slow=60,
    # ---- 动量 ----
    mom_lookback=25,
    mom_decay=0.05,
    r2_power=1.0,
    score_cap=20.0,
    # ---- 量能 ----
    use_vol_filter=True,
    vol_spike_mult=2.5,
    vol_win=5,
    # ---- 卖出 ----
    stop_loss=0.08,
    stop_mode="close",
    min_hold=0,
    cooldown=0,
    switch_margin=0.0,        # 新标的得分需超过当前持仓的幅度（迟滞）
    use_rank_exit=True,
    # ---- 择时 ----
    regime="none",            # none | breadth | index | and
    regime_thr=0.25,
    regime_win=5,
    regime_ma=60,
    risk_asset="cash",        # cash | bond | gold
    # ---- 执行 ----
    exec_mode="next_open",    # next_open | close
    cost_per_side=0.0001,   # 单边 1bp（ETF 佣金万 0.5~1，另按 0.5 的保守上限取 1bp）
    min_amt=3e6,
    min_age=120,              # 上市满 N 个交易日才可入选
    # ---- 组合 ----
    top_n=1,
    dedupe_theme=False,       # 同一底层指数只留一只（避免"假分散"）
    weight="equal",           # equal | score | invvol
    target_vol=0.0,           # 组合目标年化波动（0=不做波动率控制）
    vol_lookback=20,
    dd_brake=0.0,             # 组合回撤刹车阈值（0=关闭）
    dd_resume=0.10,           # 回撤修复到该水平才恢复
    rebalance_dow=None,       # None=每日; 4=每周五
    rebalance_buffer=0.02,
)

RISK_CODES = {"bond": "511260", "gold": "518880"}

_CO = ("华泰柏瑞|华泰紫金|华宝|华夏|易方达|富国|国泰|南方|嘉实|广发|工银瑞信|工银|天弘|汇添富|"
       "招商|华安|博时|银华|鹏华|国联安|永赢|平安|国联|兴业|中银|建信|民生|西部利得|景顺长城|"
       "景顺|长城|华富|前海开源|浙商|中欧|东方红|东方|国投瑞银|泰康|泰达宏利|泓德|创金合信|"
       "兴证全球|诺安|大成|万家|中加|浦银安盛|华润元大|新华|方正富邦|中金|申万菱信|国海富兰克林|"
       "农银|交银施罗德|海富通|长盛|宝盈|融通|摩根|安信|泓德|睿远|汇安|红土创新|华泰证券")

def theme_key(name):
    n = re.sub(r"ETF.*$|etf.*$|联接.*$", "", str(name)).strip()
    n = re.sub(_CO, "", n).strip()
    n = re.sub(r"[（(].*?[）)]", "", n).strip()
    return n or str(name)
DEF_SETS = {"bond": ["bond"], "gold": ["gold"], "bond_gold": ["bond", "gold"],
            "adaptive": ["cash", "bond", "gold"]}


class Backtest:
    def __init__(self, panel, start="2020-01-01", end="2026-10-21",
                 static_pool=None, **kwargs):
        self.p = {**DEFAULT, **kwargs}
        self.panel = panel
        self.dates = panel["dates"]
        self.mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        self.codes = panel["codes"]
        self.cidx = {c: i for i, c in enumerate(self.codes)}
        self.names = panel["names"]
        self._static = static_pool
        self._prepare()

    # ---------------- 预处理 ----------------
    def _prepare(self):
        P, par = self.panel, self.p
        close, amt = P["close"], P["amount"]
        self.close, self.open, self.high, self.low, self.amt = (
            close, P["open"], P["high"], P["low"], amt)
        self.ma_f = sma(close, par["ma_fast"])
        self.ma_s = sma(close, par["ma_slow"])
        self.trend_ok = (close > self.ma_f) & (self.ma_f > self.ma_s)

        mom, r2 = rolling_wls_momentum(close, L=par["mom_lookback"], decay=par["mom_decay"])
        self.mom, self.r2 = mom, r2
        self.score = np.clip(mom, -1, par["score_cap"]) * np.power(np.clip(r2, 0, 1), par["r2_power"])

        # 量能排雷
        prev = shift(amt, 1)
        w = par["vol_win"]
        base = np.full_like(amt, np.nan)
        if amt.shape[0] > w:
            with np.errstate(invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base[w:] = np.nanmean(np.stack([shift(prev, k) for k in range(w)]), axis=0)[w:]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.vol_ratio = np.where(base > 0, amt / base, np.nan)
        self.vol_spike = self.vol_ratio > par["vol_spike_mult"]

        # 上市时间
        valid = np.isfinite(close)
        first = np.where(valid.any(axis=0), valid.argmax(axis=0), 10 ** 9)
        age = np.arange(len(self.dates))[:, None] - first[None, :]
        self.age_ok = age >= par["min_age"]

        # 动态池
        amt5 = pd.DataFrame(amt).rolling(par["dyn_win"], min_periods=3).mean().to_numpy()
        self.amt5 = amt5
        self.dyn_ok = amt5 >= par["dyn_min_amt"]
        # 货币/现金类 ETF 不参与选股（只作为避险资产）
        cash_like = np.array([bool(re.search("货币|添益|日利|现金|保证金|理财", str(self.names.get(c, ""))))
                              for c in self.codes])
        self.equity_ok = ~cash_like
        self.theme = [theme_key(self.names.get(c, c)) for c in self.codes]
        self.theme = np.array(self.theme)
        self.tradable = (amt >= par["min_amt"]) & valid & (close > 0)

        if self._static is None:
            self._static = static_core_pool(self.panel, asof=par["static_asof"],
                                            top_n=par["static_pool_size"],
                                            min_avg_amt=par["static_min_amt"])
        self.static_idx = np.array([self.cidx[c] for c in self._static if c in self.cidx])

        self.regime = self._build_regime()

        # 风险资产（避险用）
        self.risk_idx = []
        if par["risk_asset"] in DEF_SETS:
            self.risk_idx = [self.cidx[RISK_CODES[k]] for k in DEF_SETS[par["risk_asset"]]
                             if k != "cash" and RISK_CODES[k] in self.cidx]
            self.risk_adaptive = par["risk_asset"] == "adaptive"
        else:
            self.risk_adaptive = False

    def _build_regime(self):
        par = self.p
        mode = par["regime"]
        if mode == "none":
            return np.ones(len(self.dates), dtype=bool)
        close = self.close
        n_trend = (np.isfinite(close) & self.trend_ok & self.age_ok).sum(axis=1)
        breadth = n_trend / np.maximum((np.isfinite(close) & self.age_ok).sum(axis=1), 1)
        b = pd.Series(breadth).rolling(par["regime_win"], min_periods=1).mean().to_numpy()
        bm = self.panel["bench"].to_numpy()
        ma = pd.Series(bm).rolling(par["regime_ma"]).mean().to_numpy()
        idx_on = bm > ma
        br_on = b >= par["regime_thr"]
        if mode == "breadth":
            on = br_on
        elif mode == "index":
            on = idx_on
        elif mode == "and":
            on = idx_on & br_on
        elif mode == "or":
            on = idx_on | br_on
        else:
            raise ValueError(mode)
        return pd.Series(on).fillna(False).to_numpy()

    def _defensive(self, t):
        """风险关闭时持有的避险资产。adaptive: 在 债券/黄金/现金 中按 60 日动量择优。"""
        par = self.p
        if not self.risk_idx:
            return []
        if not self.risk_adaptive:
            return list(self.risk_idx)
        L = par.get("def_lookback", 60)
        if t < L:
            return []
        best, best_r = None, 0.0            # 现金的收益视为 0
        for c in self.risk_idx:
            p0, p1 = self.close[t - L, c], self.close[t, c]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r = p1 / p0 - 1
                if r > best_r:
                    best, best_r = c, r
        return [best] if best is not None else []

    # ---------------- 候选 ----------------
    def _rank(self, t):
        par = self.p
        sc = self.score[t]
        ok = self.tradable[t] & self.trend_ok[t] & self.age_ok[t] & self.equity_ok & np.isfinite(sc)
        if par["use_vol_filter"]:
            ok &= ~np.nan_to_num(self.vol_spike[t], nan=0).astype(bool)
        pool = np.zeros(len(self.codes), dtype=bool)
        pool[self.static_idx] = True
        pool |= self.dyn_ok[t]
        idx = np.flatnonzero(ok & pool)
        if len(idx) == 0:
            return idx
        if par.get("random_seed") is not None:
            # 安慰剂检验：在通过所有筛选的候选里随机挑，用于衡量"排序"的真实贡献
            rs = np.random.default_rng(int(par["random_seed"]) * 100003 + t)
            perm = rs.permutation(len(idx))
            idx = idx[perm]
        else:
            idx = idx[np.argsort(-sc[idx])]
        if par["dedupe_theme"]:
            seen, keep = set(), []
            for c in idx:
                k = self.theme[c]
                if k in seen:
                    continue
                seen.add(k)
                keep.append(c)
            idx = np.array(keep, dtype=int)
        return idx

    def _vol_scaler(self, t, target):
        par = self.p
        if not par["target_vol"] or not target:
            return 1.0
        L = par["vol_lookback"]
        if t < L + 1:
            return par.get("_init_scaler", 1.0)
        sub = self.close[t - L: t + 1][:, target]
        r = np.diff(np.log(np.where(sub > 0, sub, np.nan)), axis=0)
        vol = np.nanstd(r) * np.sqrt(252)
        if not np.isfinite(vol) or vol <= 1e-6:
            return 1.0
        return float(np.clip(par["target_vol"] / vol, 0.0, 1.0))

    def _weights(self, t, target):
        par = self.p
        if not target:
            return np.array([])
        if par["weight"] == "equal" or len(target) == 1:
            w = np.ones(len(target))
        elif par["weight"] == "score":
            # NaN-safe: 未上市/预热期 score 为 NaN，按既有下限 1e-3 处理（否则归一化后污染全部权重）
            sc_ = np.where(np.isfinite(self.score[t, target]), self.score[t, target], 1e-3)
            s = np.maximum(sc_, 1e-3)
            w = s / s.sum()
        elif par["weight"] == "invvol":
            r = pd.DataFrame(self.close[:, target]).pct_change().rolling(20).std().iloc[t].to_numpy()
            r = np.where(np.isfinite(r) & (r > 0), r, np.nanmedian(r))
            w = 1 / r
        else:
            raise ValueError(par["weight"])
        w = w / w.sum()
        return w * self._vol_scaler(t, target)

    # ---------------- 执行 ----------------
    def _execute(self, t, px, units, cash, target, weights, entry_px, entry_t, trades,
                 reasons=None, risk_set=None):
        par = self.p
        val = cash + float(np.nansum(units * px))
        cost = par["cost_per_side"]
        tgt = {c: w for c, w in zip(target, weights)} if len(target) else {}
        # 卖出
        for c in list(np.flatnonzero(units > 0)):
            if c not in tgt:
                cash += units[c] * px[c] * (1 - cost)
                trades.append(dict(date=self.dates[t], code=self.codes[c],
                                   name=self.names.get(self.codes[c], ""), side="S",
                                   px=px[c], pnl=px[c] / entry_px[c] - 1,
                                   hold=t - entry_t[c], value=units[c] * px[c],
                                   shares=units[c],
                                   reason=(reasons or {}).get(c, "调仓")))
                units[c] = 0.0
                entry_px[c] = np.nan
                entry_t[c] = -1
        # 买入 / 调平：每一笔成交都留痕（含加仓、减仓）
        for c, w in tgt.items():
            if px[c] <= 0 or not np.isfinite(px[c]):
                continue
            want = w * val / px[c]
            delta = want - units[c]
            if abs(delta) * px[c] / max(val, 1e-9) < par["rebalance_buffer"] and units[c] > 0:
                continue
            if delta > 0:
                spend = min(delta * px[c] * (1 + cost), cash)
                if spend <= 0:
                    continue
                sh = spend / (px[c] * (1 + cost))
                if units[c] == 0:
                    entry_px[c] = px[c] * (1 + cost)
                    entry_t[c] = t
                    why = "避险买入" if (risk_set and c in risk_set) else "动量买入"
                else:
                    why = "加仓"
                trades.append(dict(date=self.dates[t], code=self.codes[c],
                                   name=self.names.get(self.codes[c], ""), side="B",
                                   px=px[c], pnl=np.nan, hold=np.nan,
                                   value=sh * px[c], shares=sh, reason=why))
                cash -= sh * px[c] * (1 + cost)
                units[c] += sh
            else:
                sh = min(-delta, units[c])
                if sh <= 0:
                    continue
                trades.append(dict(date=self.dates[t], code=self.codes[c],
                                   name=self.names.get(self.codes[c], ""), side="S",
                                   px=px[c], pnl=px[c] / entry_px[c] - 1,
                                   hold=t - entry_t[c], value=sh * px[c], shares=sh,
                                   reason=(reasons or {}).get(c, "减仓")))
                cash += sh * px[c] * (1 - cost)
                units[c] -= sh
        return units, cash, entry_px, entry_t, trades

    # ---------------- 主循环 ----------------
    def run(self):
        par, n = self.p, len(self.dates)
        equity = np.full(n, np.nan)
        units = np.zeros(len(self.codes))
        entry_px = np.full(len(self.codes), np.nan)
        entry_t = np.full(len(self.codes), -1)
        last_exit = np.full(len(self.codes), -10 ** 6)
        cash = 1.0
        trades = []
        pend = None
        exposure, nhold = [], []
        hold_path = []
        i0 = int(np.argmax(self.mask))
        # --end 修复：循环止于窗口内最后一天（原版 range(i0, n) 会忽略 end，跑满全样本）
        i1 = len(self.dates) - 1 - int(np.argmax(self.mask[::-1]))
        peak, brake_on, brake_scale = 1.0, False, 1.0

        for t in range(i0, i1 + 1):
            px_open = np.where(np.isfinite(self.open[t]), self.open[t], self.close[t])

            # --- 开盘执行 ---
            if pend is not None:
                tgt, wts, rsn, rsk = pend
                units, cash, entry_px, entry_t, trades = self._execute(
                    t, px_open, units, cash, tgt, wts, entry_px, entry_t, trades,
                    reasons=rsn, risk_set=rsk)
                pend = None

            held = np.flatnonzero(units > 0)
            nhold.append(len(held))
            hold_path.append(frozenset(self.codes[c] for c in held))
            exposure.append(bool(len(held)))
            equity[t] = cash + float(np.nansum(units * self.close[t]))
            if t == i1:
                break

            # 现金/货币基金收益（次日）
            cash *= 1 + self.panel["cash_ret"][t + 1]

            # --- 收盘决策 ---
            rank = self._rank(t)
            rank_pos = {c: i for i, c in enumerate(rank[: par["top_n"] * 3])}
            keep, why = [], {}
            for c in held:
                v = self.close[t, c]
                # 止损
                if par["stop_loss"] and np.isfinite(entry_px[c]):
                    trig = (v / entry_px[c] - 1 <= -par["stop_loss"]) if par["stop_mode"] == "close" \
                        else (self.low[t, c] / entry_px[c] - 1 <= -par["stop_loss"])
                    if trig:
                        last_exit[c] = t
                        why[c] = "止损"
                        continue
                # 放量出场
                if par["use_vol_filter"] and bool(np.nan_to_num(self.vol_spike[t, c])):
                    last_exit[c] = t
                    why[c] = "放量离场"
                    continue
                # 最小持仓期
                if t - entry_t[c] < par["min_hold"]:
                    keep.append(c)
                    continue
                # 趋势破了也走
                if not self.trend_ok[t, c]:
                    last_exit[c] = t
                    why[c] = "趋势破位"
                    continue
                keep.append(c)

            if not par["use_rank_exit"]:
                add = [c for c in rank if c not in keep][: max(par["top_n"] - len(keep), 0)]
                target = keep + add
            else:
                # 迟滞：新候选需明显更强才换
                inc_score = [self.score[t, c] for c in keep]
                base = min(inc_score) if inc_score else -np.inf
                add = []
                for c in rank:
                    if c in keep:
                        continue
                    if c in add:
                        continue
                    if base is not -np.inf and self.score[t, c] < base * (1 + par["switch_margin"]):
                        continue
                    if t - last_exit[c] <= par["cooldown"]:
                        continue
                    add.append(c)
                    if len(keep) + len(add) >= par["top_n"]:
                        break
                target = keep + add
                # 超出持仓数的（例如加了缓冲），保留得分最高者
                if len(target) > par["top_n"] and not keep:
                    target = target[: par["top_n"]]
                elif len(target) > par["top_n"]:
                    keep_sorted = sorted(keep, key=lambda c: -self.score[t, c])
                    target = keep_sorted + add
                    target = target[: par["top_n"]]

            # 组合回撤刹车（按比例减仓，可自动恢复）
            if par["dd_brake"]:
                cur = equity[t] if np.isfinite(equity[t]) else 0.0
                peak = max(peak, cur)
                dd = cur / peak - 1 if peak > 0 else 0.0
                if dd <= -2 * par["dd_brake"]:
                    brake_scale = 0.0
                elif dd <= -par["dd_brake"]:
                    brake_scale = 0.5
                elif dd >= -par["dd_resume"]:
                    brake_scale = 1.0
                else:
                    brake_scale = max(brake_scale, 0.5)

            # 择时开关
            if not self.regime[t]:
                target = []
                target = self._defensive(t)

            # 每周调仓模式：非调仓日只做风控卖出，不加仓
            if par["rebalance_dow"] is not None and self.dates[t].dayofweek != par["rebalance_dow"]:
                if set(target) != set(held):
                    # 仅允许减仓
                    target = [c for c in target if c in set(held)] or []
                    if target == [] and len(held) > 0:
                        target = list(held) if par["use_rank_exit"] else []
                else:
                    target = list(held)

            if set(target) != set(held):
                wts = self._weights(t, target) * (brake_scale if par["dd_brake"] else 1.0)
                reasons = {c: why.get(c, "排名掉队" if par["use_rank_exit"] else "调仓")
                           for c in held if c not in set(target)}
                risk_set = set(self.risk_idx) | set(target) if self.regime[t] is False else set(self.risk_idx)
                if par["exec_mode"] == "close":
                    px = self.close[t]
                    units, cash, entry_px, entry_t, trades = self._execute(
                        t, px, units, cash, target, wts, entry_px, entry_t, trades,
                        reasons=reasons, risk_set=risk_set)
                else:
                    pend = (target, wts, reasons, risk_set)

        eq = pd.Series(equity[i0:], index=self.dates[i0:]).dropna()
        tr = pd.DataFrame(trades)
        res = dict(equity=eq, trades=tr, dates=self.dates[i0:], hold_path=hold_path,
                   exposure=float(np.mean(exposure)), avg_hold=float(np.mean(nhold)))
        res.update(metrics(eq, self.panel["bench"], self.dates[i0:]))
        if len(tr):
            sells = tr[(tr.side == "S")].dropna(subset=["pnl"])
            res["n_trades"] = int(len(tr))
            res["win_rate"] = float((sells.pnl > 0).mean()) if len(sells) else np.nan
            res["avg_win"] = float(sells.pnl[sells.pnl > 0].mean()) if len(sells) else np.nan
            res["avg_loss"] = float(sells.pnl[sells.pnl <= 0].mean()) if len(sells) else np.nan
            res["payoff"] = abs(res["avg_win"] / res["avg_loss"]) if res["avg_loss"] else np.nan
            res["avg_hold_days"] = float(sells.hold.mean())
            res["turnover"] = float(tr.value.sum() / 2 / max(len(eq) / 252, 1e-9))
        self.res = res
        return res


# ---------------- 绩效 ----------------
def metrics(eq, bench=None, dates=None):
    if len(eq) < 20:
        return {}
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - 0.015) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    out = dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd,
               calmar=cagr / abs(mdd) if mdd < 0 else np.nan, dd_series=dd, years=yrs,
               win_day=float((r > 0).mean()), worst_day=float(r.min()),
               sortino=(r.mean() * 252 - 0.015) / (r[r < 0].std() * np.sqrt(252)) if (r < 0).any() else np.nan)
    m = eq.resample("ME").last().pct_change().dropna()
    out["monthly"], out["win_month"] = m, float((m > 0).mean())
    y = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]),
                   eq.resample("YE").last()]).pct_change().dropna()
    out["yearly"] = y
    # 近 4 个月（约 84 个交易日）
    if len(eq) > 84:
        out["ret_4m"] = eq.iloc[-1] / eq.iloc[-85] - 1
    if bench is not None:
        b = bench.reindex(eq.index).ffill()
        br = b.pct_change().reindex(r.index).fillna(0)
        out["bench_total"] = b.iloc[-1] / b.iloc[0] - 1
        out["bench_cagr"] = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
        out["bench_mdd"] = (b / b.cummax() - 1).min()
        out["bench_4m"] = b.iloc[-1] / b.iloc[-85] - 1 if len(b) > 84 else np.nan
        out["corr"] = r.corr(br)
        beta = np.cov(r, br)[0, 1] / np.var(br) if np.var(br) > 0 else np.nan
        out["beta"] = beta
        out["alpha"] = cagr - beta * out["bench_cagr"]
        out["bench_eq"] = b / b.iloc[0]
    return out


def summarize(res, name=""):
    keys = ["total", "cagr", "vol", "sharpe", "sortino", "mdd", "calmar", "win_rate",
            "payoff", "avg_hold_days", "n_trades", "turnover", "exposure", "corr", "alpha"]
    row = {"name": name}
    for k in keys:
        v = res.get(k, np.nan)
        row[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
    row["bench_cagr"] = float(res.get("bench_cagr", np.nan))
    row["ret_4m"] = float(res.get("ret_4m", np.nan))
    return row

# =================================================================
# S9 参数（就是这 11 个键把 S0 变成 S9）
# =================================================================
S9 = dict(switch_margin=0.15, min_hold=5, cooldown=5, mom_lookback=40, stop_loss=0.10,
          regime="or", risk_asset="bond_gold", vol_spike_mult=2.0, top_n=3,
          dedupe_theme=True)

# -*- coding: utf-8 -*-
"""省内存面板加载: 与 S9_orig_backtest.load_panel() 输出等价, 峰值内存 ~1/3。
- 不读 name 大列（另行以 code→name 小表提供）
- 逐字段 pivot 后立即转 numpy 并释放
用法: python3 s9_lean.py   # 打印校验和
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def load_panel_lean():
    cal = pd.DatetimeIndex(sorted(pd.read_parquet(os.path.join(DATA, "benchmark.parquet"))["date"].unique()))
    daily_cols = ["code", "date", "open", "close", "high", "low", "volume", "amount"]
    daily = pd.read_parquet(os.path.join(DATA, "etf_daily.parquet"), columns=daily_cols)
    extra = pd.read_parquet(os.path.join(DATA, "cash_extra.parquet"))
    bm = pd.read_parquet(os.path.join(DATA, "benchmark.parquet"))

    daily = pd.concat([daily, extra[daily_cols]], ignore_index=True)
    daily = daily.drop_duplicates(["code", "date"], keep="last")

    codes = np.array(sorted(daily["code"].unique()))
    cidx = {c: i for i, c in enumerate(codes)}
    T, N = len(cal), len(codes)
    tidx = pd.Series(np.arange(T), index=cal)

    rows = tidx.reindex(pd.DatetimeIndex(daily["date"])).to_numpy()
    cols = daily["code"].map(cidx).to_numpy()
    valid = np.isfinite(rows)

    fields = {}
    for f in ["close", "open", "high", "low", "amount", "volume"]:
        arr = np.full((T, N), np.nan, dtype=np.float64)
        arr[rows[valid], cols[valid]] = daily[f].to_numpy(dtype=np.float64)[valid]
        fields[f] = arr

    close = pd.DataFrame(fields["close"]).ffill(limit=5).to_numpy()
    for f in ["open", "high", "low"]:
        fields[f] = pd.DataFrame(fields[f]).ffill(limit=5).to_numpy()
    fields["close"] = close
    fields["amount"] = np.nan_to_num(fields["amount"], nan=0.0)

    nm = pd.read_parquet(os.path.join(DATA, "etf_daily.parquet"), columns=["code", "name"])
    names = nm.drop_duplicates("code", keep="last").set_index("code")["name"].to_dict()

    cash = extra[extra.code == "511880"].set_index("date")["close"].reindex(cal).ffill()
    cash_ret = cash.pct_change().fillna(0.0).clip(-0.05, 0.05).to_numpy()

    bench = bm.set_index("date")["close"].reindex(cal).ffill()

    return dict(dates=cal, codes=codes, names=names,
                close=fields["close"], open=fields["open"], high=fields["high"], low=fields["low"],
                amount=fields["amount"], bench=bench, cash_ret=cash_ret)

def checksum(P):
    import hashlib
    h = hashlib.sha1()
    h.update(str(P["dates"][-1]).encode())
    h.update(P["codes"].tobytes())
    for k in ["close", "open", "high", "low", "amount"]:
        h.update(np.nan_to_num(P[k], nan=-999.0).round(6).tobytes())
    h.update(np.asarray(P["bench"], dtype=np.float64).round(6).tobytes())
    h.update(np.asarray(P["cash_ret"], dtype=np.float64).round(10).tobytes())
    out = dict(dates_first=str(P["dates"][0]), dates_last=str(P["dates"][-1]), n_dates=len(P["dates"]),
               n_codes=len(P["codes"]), close_sum=float(np.nansum(P["close"])),
               amt_sum=float(np.nansum(P["amount"])), sha1=h.hexdigest())
    return out


# =================================================================
# 内嵌静态池 (150 只, asof 2019-12-31) — 与 static_pool_full.json 同步
# =================================================================
POOL = json.loads(r"""["511990", "511880", "510050", "159001", "510300", "511810", "159005", "159901", "159915", "510900", "511660", "518880", "511900", "159003", "510180", "511980", "511010", "510500", "511800", "511860", "511820", "159919", "511690", "159902", "159920", "510330", "510880", "159949", "510230", "512880", "511850", "511600", "159934", "159937", "511960", "511830", "518800", "511670", "511910", "512000", "511650", "511930", "510170", "512500", "510650", "513500", "513100", "510060", "159903", "510510", "511970", "512660", "510310", "510850", "510010", "510030", "512090", "512800", "510130", "511700", "159922", "512070", "513050", "512760", "510660", "512180", "159952", "159939", "159905", "512960", "510810", "512160", "510290", "512990", "510630", "510390", "512010", "510360", "510710", "159907", "510590", "512380", "159928", "510160", "510410", "510020", "159938", "511260", "511920", "513030", "511220", "512900", "159925", "510580", "159943", "512950", "512290", "159929", "512750", "512510", "511770", "510190", "159959", "512580", "159948", "512100", "512400", "510210", "510150", "512520", "512680", "512980", "512480", "513660", "159931", "159909", "512690", "512040", "511270", "512910", "513600", "159912", "512260", "159923", "512330", "159933", "159910", "510800", "159958", "512170", "512700", "512810", "510090", "512200", "512120", "512550", "510600", "159908", "512640", "512890", "512820", "159936", "159906", "159954", "510270", "159965", "159940", "159913", "159957", "159930"]""")

PROFILES = {
    "B0": {},                                  # = DEFAULT ∪ S9
    "K3": dict(top_n=2, regime_thr=0.20, dyn_min_amt=0, weight="score", stop_loss=0.0),
    "K5": dict(target_vol=0.15, top_n=2, regime_thr=0.20),
}
GOLDEN = dict(name="B0@W2 2016-01-04~2019-12-31", start="2016-01-04", end="2019-12-31",
              total=0.3332458213558225, sharpe_rf0=0.5730812747686882, tol=1e-6)


# =================================================================
# 数据层 (仅依赖 data/): 缺失或与 klines 不同步 → 本地自动重建
# =================================================================
KDIR = os.path.join(HERE, "data", "klines")

def rebuild_parquets_from_klines():
    """纯本地: klines csv → 三件套 (与 rebuild_etf_daily.py 逐行同口径: 含511880, >=2016, 排序)。"""
    import pandas as pd
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    rows, n_skip = [], 0
    for fn in sorted(os.listdir(KDIR)):
        if not fn.endswith(".csv"):
            continue
        base = fn[:-4]
        if base.startswith("_index"):
            continue
        try:
            df = pd.read_csv(os.path.join(KDIR, fn))
        except Exception:
            n_skip += 1; continue
        if df.empty or "date" not in df.columns:
            n_skip += 1; continue
        df["code"] = base
        rows.append(df)
    daily = pd.concat(rows, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["date"] >= "2016-01-01"]
    daily = daily[["code", "date", "name", "open", "close", "high", "low", "volume", "amount"]]
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    daily.to_parquet(os.path.join(HERE, "data", "etf_daily.parquet"), index=False)
    idx = pd.read_csv(os.path.join(KDIR, "_index_000300.csv"))
    idx["date"] = pd.to_datetime(idx["date"])
    bm = idx[["date", "close"]].sort_values("date")
    bm = bm[bm["date"] >= "2016-01-01"].reset_index(drop=True)
    bm.to_parquet(os.path.join(HERE, "data", "benchmark.parquet"), index=False)
    ce = pd.read_csv(os.path.join(KDIR, "511880.csv"))
    ce["date"] = pd.to_datetime(ce["date"])
    ce = ce[ce["date"] >= "2016-01-01"].sort_values("date").reset_index(drop=True)
    ce["code"] = "511880"
    ce = ce[["code", "date", "name", "open", "close", "high", "low", "volume", "amount"]]
    ce.to_parquet(os.path.join(HERE, "data", "cash_extra.parquet"), index=False)
    print(f"[data] rebuilt: etf {len(daily):,} rows / {daily.code.nunique()} codes (skip {n_skip}) | "
          f"bench {len(bm)} last {bm.date.max().date()} | cash {len(ce)}", flush=True)

def bootstrap_klines():
    """空包引导: 无 klines 但有 data/codes.txt → 生成表头空骨架(供 --fetch 全量填充)。"""
    ipath = os.path.join(KDIR, "_index_000300.csv")
    if os.path.exists(ipath):
        return
    cpath = os.path.join(HERE, "data", "codes.txt")
    if not os.path.exists(cpath):
        raise SystemExit(f"data/klines 缺失且无 data/codes.txt, 无法引导数据: {KDIR}")
    os.makedirs(KDIR, exist_ok=True)
    header = "date,open,close,high,low,volume,amount,name"
    n = 0
    with open(cpath, encoding="utf-8") as fh:
        for line in fh:
            code = line.strip()
            if not code:
                continue
            p = os.path.join(KDIR, code + ".csv")
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(header + "\n")
                n += 1
    print(f"[bootstrap] {n} skeleton csvs from codes.txt (待 --fetch 填充)", flush=True)

def ensure_data():
    """benchmark.parquet 末日 == _index klines 末日; 否则(或缺失)本地重建。
    klines 为空骨架(未 fetch)时绝不能重建(会毁掉随包 parquet)。"""
    import pandas as pd
    ipath = os.path.join(KDIR, "_index_000300.csv")
    if not os.path.exists(ipath):
        raise SystemExit(f"data/klines missing: {KDIR} (放数据, 或用含 data/codes.txt 的包引导后 --fetch)")
    klast = pd.read_csv(ipath, usecols=["date"])["date"].max()
    etf_p = os.path.join(HERE, "data", "etf_daily.parquet")
    if pd.isna(klast):
        if not os.path.exists(etf_p):
            raise SystemExit("数据未就绪: klines 为空骨架且无 etf_daily.parquet → 先跑 python3 s9_live.py --fetch")
        return
    try:
        blast = pd.read_parquet(os.path.join(HERE, "data", "benchmark.parquet"), columns=["date"])["date"].max()
        elast = pd.read_parquet(etf_p, columns=["date"])["date"].max()
    except Exception:
        blast = elast = None
    if str(pd.Timestamp(blast).date()) != str(pd.Timestamp(klast).date()) or \
       str(pd.Timestamp(elast).date()) != str(pd.Timestamp(klast).date()):
        # 重建放子进程: 其 ~1.5GB 峰值随进程退出释放, 不污染后续回测(否则同进程 OOM/137)
        me = os.path.abspath(__file__)
        r = subprocess.run([sys.executable, me, "--_rebuild-internal"])
        if r.returncode != 0:
            raise SystemExit(f"rebuild failed rc={r.returncode}")

def do_fetch():
    """联网追加 klines(每文件 beg=自身末日+1, 4 重试, 8 线程) → 本地重建三件套。"""
    import random, time, urllib.request, warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # 骨架 concat 触发的无害告警
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd
    FIELDS = "f51,f52,f53,f54,f55,f56,f57"
    END = "20501231"

    def secid(code):
        if code.startswith("_index"):
            return "1." + code.replace("_index_", "")
        return ("1." if code.startswith(("5", "6")) else "0.") + code

    def fetch_one(f):
        base = os.path.basename(f)[:-4]
        try:
            cur = pd.read_csv(f)
        except Exception:
            return base, "unreadable"
        if "date" not in cur.columns:
            return base, "empty"
        if cur.empty:
            beg = "19900101"          # 空骨架: 全量下载
        else:
            last = str(cur["date"].astype(str).max())[:10].replace("-", "")
            beg = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid(base)}"
               f"&klt=101&fqt=1&beg={beg}&end={END}&fields1=f1,f2,f3&fields2={FIELDS}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    d = json.load(r)
                data = d.get("data") or {}
                name = data.get("name") or ""
                rows = []
                for kl in data.get("klines") or []:
                    p = kl.split(",")
                    if len(p) >= 7:
                        rows.append([p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                                     float(p[5]), float(p[6]), name])
                if not rows:
                    return base, "nodata"
                have = set(cur["date"].astype(str).str[:10])
                new = [r for r in rows if r[0] not in have]
                if not new:
                    return base, "dup"
                nd = pd.DataFrame(new, columns=cur.columns)
                out = pd.concat([cur, nd], ignore_index=True).drop_duplicates("date", keep="last")
                out.to_csv(f, index=False)
                return base, f"added {len(new)}"
            except Exception as e:
                if attempt == 3:
                    return base, f"err:{e}"
                time.sleep(0.5 * (attempt + 1) + random.random())

    files = sorted([os.path.join(KDIR, x) for x in os.listdir(KDIR) if x.endswith(".csv")])
    stats, fails = {}, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            code, st = fut.result()
            key = "err" if st.startswith("err") else st.split()[0]
            stats[key] = stats.get(key, 0) + 1
            if st.startswith("err") or st == "unreadable":
                fails.append((code, st))
            if i % 300 == 0:
                print(f"  fetch {i}/{len(files)} ...", flush=True)
    print("[fetch] stats:", stats, flush=True)
    for code, _ in list(fails):  # 串行重试一轮
        _, st2 = fetch_one(os.path.join(KDIR, f"{code}.csv"))
        print("  retry", code, st2, flush=True)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--_rebuild-internal"])
    if r.returncode != 0:
        raise SystemExit(f"rebuild failed rc={r.returncode}")


# =================================================================
# 决策 → 指令 (与引擎 run() 同语义; stub 日 = 次一工作日开盘)
# =================================================================
def add_stub(P):
    import numpy as np, pandas as pd
    last = pd.Timestamp(P["dates"][-1])
    stub = last
    while True:
        stub = stub + pd.Timedelta(days=1)
        if stub.dayofweek < 5:
            break
    P["dates"] = P["dates"].append(pd.DatetimeIndex([stub]))
    for k in ["close", "open", "high", "low"]:
        P[k] = np.vstack([P[k], P[k][-1]])
    P["amount"] = np.vstack([P["amount"], np.zeros(P["amount"].shape[1])])
    P["bench"] = pd.concat([P["bench"], pd.Series([P["bench"].iloc[-1]], index=pd.DatetimeIndex([stub]))])
    P["cash_ret"] = np.append(P["cash_ret"], 0.0)
    return last, stub

def reconstruct_entry(trades):
    import pandas as pd
    tr = trades.copy(); tr["date"] = pd.to_datetime(tr["date"])
    ent = {}
    for _, r in tr.sort_values("date", kind="stable").iterrows():
        if r["side"] == "B":
            ent[r["code"]] = (r["px"], r["date"])
        else:
            ent.pop(r["code"], None)
    return ent

def close_reason(bt, t, c, entry_map):
    import numpy as np, pandas as pd
    p = bt.p
    v = bt.close[t, c]
    ep, etd = entry_map.get(str(bt.codes[c]), (np.nan, None))
    if p["stop_loss"] and np.isfinite(ep) and np.isfinite(v):
        trig = (v / ep - 1 <= -p["stop_loss"]) if p["stop_mode"] == "close" else (bt.low[t, c] / ep - 1 <= -p["stop_loss"])
        if trig:
            return "止损"
    if p["use_vol_filter"] and bool(np.nan_to_num(bt.vol_spike[t, c])):
        return "放量离场"
    if etd is not None:
        etd_i = int(np.searchsorted(bt.dates.values, np.datetime64(pd.Timestamp(etd))))
        if t - etd_i < p["min_hold"]:
            return "持有期不足"
    if not bt.trend_ok[t, c]:
        return "趋势破位"
    return "排名掉队"

def gate_info(bt, t):
    import numpy as np, pandas as pd
    close = bt.close
    n_trend = (np.isfinite(close) & bt.trend_ok & bt.age_ok).sum(axis=1)
    breadth = n_trend / np.maximum((np.isfinite(close) & bt.age_ok).sum(axis=1), 1)
    bwin = pd.Series(breadth).rolling(bt.p["regime_win"], min_periods=1).mean().to_numpy()
    ma = pd.Series(bt.panel["bench"]).rolling(bt.p["regime_ma"]).mean().to_numpy()
    bmv = float(np.asarray(bt.panel["bench"])[t])
    ci = bmv > ma[t]
    cb = bwin[t] >= bt.p["regime_thr"]          # 引擎 _build_regime 是 GE
    return dict(gate_on=bool(bt.regime[t]), mode=bt.p["regime"],
                index=round(bmv, 2), ma60=round(float(ma[t]), 2), index_pass=bool(ci),
                breadth=round(float(bwin[t]), 4), breadth_thr=bt.p["regime_thr"], breadth_pass=bool(cb))

def generate(profile="K5", asof=None, verify=False, out_json=True):
    """核心: 返回指令 dict (schema s9-daily-orders/v1)。CLI 与 import 共用。"""
    import numpy as np, pandas as pd
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    health = dict(ok=True, messages=[]); messages = health["messages"]

    P = load_panel_lean()
    last, stub = add_stub(P)
    asof = pd.Timestamp(asof) if asof else pd.Timestamp(P["dates"][-2])
    if pd.Timestamp(P["dates"][-2]) != asof:
        raise SystemExit(f"asof {asof.date()} != data last {pd.Timestamp(P['dates'][-2]).date()} (先 --fetch)")

    params = {**DEFAULT, **dict(S9), **PROFILES[profile]}
    if len(POOL) != 150:
        health["ok"] = False; messages.append(f"pool {len(POOL)} != 150")

    golden_res = None
    if verify:
        import gc
        base_only = {**DEFAULT, **dict(S9)}
        bt2 = Backtest(P, start=GOLDEN["start"], end=GOLDEN["end"], static_pool=POOL, **base_only)
        r2 = bt2.run()
        rr = r2["equity"].pct_change().dropna()
        sh0 = float(rr.mean() / rr.std() * np.sqrt(252))
        golden_res = dict(name=GOLDEN["name"], total=float(r2["total"]), expected=GOLDEN["total"],
                          sharpe_rf0=sh0, expected_sharpe=GOLDEN["sharpe_rf0"])
        golden_res["pass"] = bool(abs(golden_res["total"] - GOLDEN["total"]) < GOLDEN["tol"]
                                  and abs(sh0 - GOLDEN["sharpe_rf0"]) < 1e-4)
        if not golden_res["pass"]:
            health["ok"] = False; messages.append("golden check FAILED (engine copy drifted?)")
        del bt2, r2; gc.collect()

    bt = Backtest(P, start=str(P["dates"][0].date()), end=str(stub.date()), static_pool=POOL, **params)
    res = bt.run()
    t_asof, t_stub = len(P["dates"]) - 2, len(P["dates"]) - 1
    hold_now, hold_tgt = set(res["hold_path"][t_asof]), set(res["hold_path"][t_stub])
    tgt_idx = [bt.cidx[c] for c in sorted(hold_tgt)]
    wts = bt._weights(t_asof, tgt_idx) if tgt_idx else np.array([])
    wmap = {str(bt.codes[i]): float(w) for i, w in zip(tgt_idx, wts)}

    gate = gate_info(bt, t_asof)
    entry_map = reconstruct_entry(res["trades"])
    _trd = res["trades"].copy(); _trd["date"] = pd.to_datetime(_trd["date"])
    tradereason = {r["code"]: (r.get("reason") or "") for _, r in _trd[_trd["date"] == asof].iterrows()}

    orders = []
    for c in sorted(hold_tgt):
        if c in hold_now:
            action, reason = "HOLD", "在场"
        else:
            action = "BUY"
            reason = tradereason.get(c) or ("风险开闸·避险资产轮动" if not gate["gate_on"] else "开闸·打分入选")
        orders.append(dict(action=action, code=c, name=str(bt.names.get(c, "")),
                           target_weight_pct=round(wmap.get(c, 0.0) * 100, 2),
                           ref_price=round(float(bt.close[t_asof, bt.cidx[c]]), 4), reason=reason))
    sells = [dict(action="SELL", code=c, name=str(bt.names.get(c, "")),
                  ref_price=round(float(bt.close[t_asof, bt.cidx[c]]), 4),
                  reason=close_reason(bt, t_asof, bt.cidx[c], entry_map))
             for c in sorted(hold_now - hold_tgt)]

    rank = bt._rank(t_asof)
    static_set = set(bt.static_idx)
    attack = [dict(code=str(bt.codes[i]), name=str(bt.names.get(bt.codes[i], "")),
                   score=round(float(bt.score[t_asof, i]), 4),
                   in_static_pool=bool(i in static_set), in_dyn_pool=bool(bt.dyn_ok[t_asof, i]))
              for i in list(rank)[:3] if str(bt.codes[i]) not in hold_tgt]

    for c in sorted(hold_tgt | hold_now):
        if not np.isfinite(bt.close[t_asof, bt.cidx[c]]):
            health["ok"] = False; messages.append(f"close not finite: {c}")
    if golden_res is not None:
        health["golden"] = golden_res

    out = dict(schema="s9-daily-orders/v1",
               generated_at=dt.datetime.now().isoformat(timespec="seconds"),
               asof=str(asof.date()), exec_day=str(stub.date()),
               exec_price_note="开盘执行; 参考价=asof收盘(近似)",
               profile=profile, params={k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()},
               data=dict(etf_last=str(pd.Timestamp(P["dates"][-2]).date()),
                         pool_size=len(POOL), n_dates=len(P["dates"]), n_codes=len(P["codes"])),
               gate=gate, holdings_now=sorted(hold_now), target_next=sorted(hold_tgt),
               orders=orders, sells=sells,
               tradereason_asof=tradereason, attack_candidates_if_gate_opens=attack,
               equity_asof=round(float(res["equity"].iloc[-2]), 6),
               equity_stub=round(float(res["equity"].iloc[-1]), 6), health=health)
    if out_json:
        path = os.path.join(HERE, f"orders_{profile}_{asof.date()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
        out["_path"] = path
    return out

def main():
    ap = argparse.ArgumentParser(description="S9 单文件日度指令生成器")
    ap.add_argument("profile", nargs="?", choices=list(PROFILES), default="K5")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--fetch", action="store_true", help="联网更新 klines 后再出单")
    ap.add_argument("--verify", action="store_true", help="金标准 B0@W2 自检")
    ap.add_argument("--json", action="store_true", help="完整 JSON 打到 stdout")
    ap.add_argument("--_rebuild-internal", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._rebuild_internal:
        rebuild_parquets_from_klines()
        return
    bootstrap_klines()
    if a.fetch:
        do_fetch()
    ensure_data()

    out = generate(profile=a.profile, asof=a.asof, verify=a.verify, out_json=True)
    g = out["gate"]
    print(f"== S9 {a.profile} | asof {out['asof']} -> exec {out['exec_day']} 开盘 == ")
    print(f"门控: {'ON' if g['gate_on'] else 'OFF'} mode={g['mode']} | {g['index']:.0f} vs ma60 {g['ma60']:.0f} ({'v' if g['index_pass'] else 'x'})"
          f" | 广度{g['breadth']:.3f} {'>=' if g['breadth_pass'] else '<'} {g['breadth_thr']}")
    print("持仓:", ", ".join(out["holdings_now"]) or "(空)", " -> 目标:", ", ".join(out["target_next"]) or "(空)")
    for s in out["sells"]:
        print(f"  SELL {s['code']} {s['name']} [{s['reason']}] ref={s['ref_price']}")
    for o in out["orders"]:
        print(f"  {o['action']:4s} {o['code']} {o['name']} {o['target_weight_pct']:.1f}% ref={o['ref_price']} [{o['reason']}]")
    if out["attack_candidates_if_gate_opens"]:
        print("翻多候选:", ", ".join(f"{x['code']}({x['score']:.2f})" for x in out["attack_candidates_if_gate_opens"]))
    gv = out["health"].get("golden")
    if gv:
        print("金标准:", "PASS" if gv["pass"] else "FAIL", f"total={gv['total']:.6f} sharpe={gv['sharpe_rf0']:.6f}")
    print("健康:", "OK" if out["health"]["ok"] else "FAIL", "|", "; ".join(out["health"]["messages"]) or "-", flush=True)
    if a.json:
        o2 = dict(out); o2.pop("_path", None)
        print(json.dumps(o2, ensure_ascii=False, default=float))
    if not out["health"]["ok"]:
        sys.exit(2)

if __name__ == "__main__":
    main()
