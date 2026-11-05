# S9 策略执行手册 (STRATEGY_PLAYBOOK)

> 读者: 另一个没有本会话上下文的 agent / 执行者。
> 目标: 拿到本文件 + 同目录三个文件即可独立完成每日决策、下单与自检。
> **权威顺序: `S9_orig_backtest.py`(引擎源码, 语义唯一真源) > `strategy_config.json`(机器可读快照) > 本文件(说明)。**
> 版本: v1 / 数据截止 2026-11-05 / 引擎为原版回测(3 个 bug 修复, 见 `S9_orig_backtest.py` 顶部注释)。

---

## 0. 快速开始(每日三步)

```bash
cd /home/user
python3 daily_orders.py --profile K5 --fetch     # 1) 更新行情(klines→parquet)+ 生成指令
python3 daily_orders.py --profile K5 --verify    # 2) 金标准自检(建议每次先跑)
# → 输出 orders_K5_<asof>.json, stdout 打印门控/持仓/指令; exit 0=健康, 2=健康失败
python3 daily_orders.py --profile K3 --verify    # 同法生成 K3
```

- 不联网: 去掉 `--fetch`(需先保证数据已更新, 见 §5)。
- 指令语义: **asof 日收盘决策 → 次一工作日开盘执行**; 参考价=asof 收盘价(近似)。
- 输出 JSON 字段速查: `gate`(门控分解) `orders`(目标持仓 BUY/HOLD+目标权重) `sells`(卖出+原因) `holdings_now/target_next` `attack_candidates_if_gate_opens` `health`。

## 1. 文件地图

| 文件 | 作用 |
|---|---|
| **`daily_orders.py`** | 唯一日常入口: 面板加载(lean) + stub 日 + 门控/目标/权重/卖出原因 + 健康检查 + `--verify` 金标准 + JSON 输出。自愈(pyarrow 缺失自动装并 re-exec; etf_daily 缺失自动从 klines 重建) |
| **`strategy_config.json`** | 机器可读规格: 三档全量参数(effective)、决策管线 7 步、公式、universe 规则、数据契约、坑单 |
| **`STRATEGY_PLAYBOOK.md`** | 本文件 |
| `S9_orig_backtest.py` | 引擎(真源)。`dict(m.S9)` 只含 S9 增量键; `Backtest.__init__` 内部合并 `{**DEFAULT, **S9, **kwargs}` |
| `s9_lean.py` | 内存瘦身面板加载(与引擎 load_panel 校验和一致)。bench 必须是 pandas Series |
| `update_klines.py` | 联网追加 klines(`--beg yyyymmdd`, 4 重试, 8 线程) + 重建 benchmark(≥2016)/cash_extra |
| `rebuild_etf_daily.py` | 纯本地: klines → etf_daily.parquet(+顺带重建 benchmark/cash_extra, 与 etf 同截止日) |
| `get_signals.py` | 一次性脚本(11-05 实例): K3/K5 成交明细/持仓/门控分解, 含 stub 手工追加的参考写法 |
| `static_pool_full.json` | 静态池 150 只(asof 2019-12-31, 成交额降序, age≥120) |
| `data/klines/*.csv` | **数据之源**: 前复权日线, 1717 有效代码 + `_index_000300.csv`(基准) + `511880.csv`(现金计息) |
| `report.md` / `report_orig_engine.md` | 分析报告(基线 / 原版引擎优化); 选股依据与窗口对比 |
| `results_*.csv` | 63 配置结果(含金标准 B0 W2 行) |

## 2. 参数(必须按此合并!)

**合并规则 `{**DEFAULT, **S9, **profile_overrides}`**(引擎内自动; 手工传参同规则)。

- DEFAULT(节选, 全量见 config): `static_pool_size 150, dyn_min_amt 5e7, dyn_win 5, static_asof 2019-12-31, ma_fast 20, ma_slow 60, mom_decay 0.05, r2_power 1, score_cap 20, use_vol_filter true, vol_spike_mult 2.5, vol_win 5, stop_loss 0.08, stop_mode close, min_hold 0, cooldown 0, switch_margin 0, use_rank_exit true, regime none, regime_thr 0.25, regime_win 5, regime_ma 60, risk_asset cash, exec_mode next_open, cost_per_side 1e-4, min_amt 3e6, min_age 120, top_n 1, dedupe_theme false, weight equal, target_vol 0, vol_lookback 20, dd_brake 0, rebalance_dow null, rebalance_buffer 0.02`
- S9 覆盖: `switch_margin 0.15, min_hold 5, cooldown 5, mom_lookback 40, stop_loss 0.1, regime or, risk_asset bond_gold, vol_spike_mult 2.0, top_n 3, dedupe_theme true`
- **B0** = 上面全部(S9 基线)。
- **K3** = B0 + `{top_n 2, regime_thr 0.20, dyn_min_amt 0, weight score, stop_loss 0}`
- **K5** = B0 + `{target_vol 0.15, top_n 2, regime_thr 0.20}` ← 实盘首选(稳健: W1 Sharpe 1.749/DD −10.2%/换手 25×)
- 选型: 要收益+交易便宜→K3(W1 +2466%, 但换手 154×、DD −28%、20bp 下 W2 Sharpe 0.25); 要稳→K5; 中间→K5b(`target_vol .15, top_n 3, thr .20`, 见 results)。完整对比报告: 会话交付的 K3 vs K5(记忆)与 `results_*_orig.csv`。

## 3. 完整算法(逐日, 与引擎逐行对应)

```
输入: 面板 close/open/high/low/amount/cash_ret/bench(前复权, 交易日历), t=asof
输出: target(次日开盘执行的目标持仓与权重)

# ── 3.1 特征(引擎 _prepare) ─────────────────────────────
trend_ok[c]  = close>MA20[c] 且 MA20[c]>MA60[c]           # 未上市/热身期 NaN→False
age_ok[c]    = 上市 ≥120 交易日
tradable[c]  = amount≥3,000,000 元 且 valid 且 close>0
equity_ok[c] = 名称不匹配 正则 `货币|添益|日利|现金|保证金|理财`   # ⚠ source: 字面"现金"误伤"现金流ETF", 不要修
vol_spike[c] = amount[t-1] > 2.0× mean(amount[t-6:t-1])     # K3/K5 用 S9 的 2.0
mom,r2       = WLS(log close, L=40, 权重 exp(0.05·j) 归一, x 居中);
               mom = exp(clip(slope·252,−5,5))−1; r2 = clip(slope²·A2/sst,0,1); cnt<0.999L → NaN
score        = clip(mom,−1,20) × clip(r2,0,1)^1.0          # [引擎 L288]
pool(t)      = static_idx(150只 ∪) {amt5日均 ≥ dyn_min_amt}  # K5: 5e7 / K3: 0
# ── 3.2 门控(_build_regime) ────────────────────────────
breadth[t]   = count(isfinite(close)&trend_ok&age_ok) / max(count(isfinite(close)&age_ok),1)
cond_idx     = bench[t] > MA60(bench)[t]                    # 严格 GT
cond_bw      = mean(breadth,5d)[t] >= regime_thr            # ⚠ 引擎是 ≥ (GE)
mode=or: gate = cond_idx | cond_bw                          # (and/breadth/index/none 同理; none=恒开)
# ── 3.3 持仓逐只平仓决策(顺序=优先级) ─────────────────
for c in held, 按序:
  1) stop_loss>0 且 close 模式: close[t,c]/entry_px−1 ≤ −stop_loss → 卖「止损」   # K3 stop_loss=0 → 关闭
  2) use_vol_filter 且 vol_spike → 卖「放量离场」
  3) t−entry_t < min_hold(5) → 保留(不卖)
  4) not trend_ok → 卖「趋势破位」
  5) 否则保留(keep); 被替换掉的 → 「排名掉队」
# ── 3.4 加仓(_rank) ───────────────────────────────────
if gate OFF: target = _defensive = [511260, 518880]         # bond_gold 固定二只(risk_adaptive=False)
else:
  rank = pool 内按 score 降序, 且 tradable&age&equity&finite, dedupe_theme 每主题留首个
  keep 逐只过上面 1-5 → keep 集
  for c in rank: 跳过在 keep/add 内; 需 score[c] ≥ min(keep scores)×(1+0.15); t−last_exit[c] ≤5 → 跳过;
       收满 top_n 停                                            # K3/K5 top_n=2
# ── 3.5 交易触发 ───────────────────────────────────────
仅当 set(target) ≠ set(held) 才交易! (权重漂移不交易; rebalance_buffer 2% 只在 _execute 内过滤"成员已变"的漂移单)
# ── 3.6 权重 ───────────────────────────────────────────
K3: w ∝ max(score,1e-3) 归一      K5/B0: 等权
× vol_scaler = clip(0.15/vol20, 0, 1)   # 仅 K5(target_vol=0.15); vol 低时=1.0
# ── 3.7 执行 ───────────────────────────────────────────
T+1 开盘市价: 先卖后买; 成本单边 1bp; 价格非有限或≤0 → 跳过;
闲置现金按 cash_extra(511880) 日收益计息; 收盘记净值。
```

## 4. 何时算"要下单"

- `orders_*.json` 里 `action ∈ {BUY,HOLD}` 对应目标持仓; `sells` 非空 → 次日开盘卖出(原因逐只给)。
- **只有 `set(target) != set(holdings_now)` 才会产生成交**; 两边一样 → 明日无单(如 11-04/11-05: 门控关、防御二只不动 → 空仓计划)。
- 门控翻多时 `attack_candidates_if_gate_opens` = top 候选(含 `in_dyn_pool` 标志); 最终名单以当日重跑 `daily_orders.py` 输出为准(勿手改)。

## 5. 数据契约与健康检查

- 更新: `python3 daily_orders.py --profile K5 --fetch`(= `update_klines.py --beg <45天前>` + `rebuild_etf_daily.py`); 只本地重建: `python3 rebuild_etf_daily.py`。
- 不变量(工具内置, 失败 exit 2 且 `health.ok=false`):
  1. `etf_last == bench_last == cash_last == klines_last`;
  2. 静态池恰 150 只; 3. 决策输入价均有限; 4. (`--verify`) 金标准:
     **B0 窗口 2016-01-04~2019-12-31: total = 0.3332458213558225 (tol 1e-6), Sharpe_rf0 = 0.5730812747686882 (tol 1e-4)** — 不过=引擎/数据被动过, 停止下单并排查。
- benchmark/cash_extra 行数应 ≥2016 起(≈2632 行@11-05, 会随日历增长)。

## 6. 已知坑(全部踩过, 勿重复)

1. **环境每轮重置**: pyarrow 丢失→工具自愈重执行; 大 parquet 被快照剔除→工具自动从 klines 重建(无需网络)。若手跑脚本: 先 `pip install pyarrow`。
2. **OOM**: benchmark 必须 ≥2016 截断(全历史 5303 行 → 面板爆内存 → SIGKILL 137)。OOM 时先查 benchmark 行数。
3. **传参陷阱**: 只传 overrides 会落回 DEFAULT(regime="none" 等)。必须 `{**DEFAULT,**S9,**ov}`(引擎/工具已封装)。
4. 类型: `hold_path` 存 codes(str); 引擎内部 `target/held/rank` 是整数位; `_weights(t, int位列表)`; `res["trades"]` 是 DataFrame; `bench` 必须是 Series。
5. 网络: push2his 偶发 `_ssl handshake timed out`(4 重试可过); push2 clist / akshare spot 常 502 → 用官方 scale API。
6. 口径: 报告 Sharpe 1.8–2 是**周频×√52 / 近 2 年 / 组合级**口径; 工具与引擎全历史日频×√252(如 B0 W2 0.573)。两者都对, 不可横比。
7. 「现金」排除正则的误伤(现金流ETF)是 source 行为——保留, 不要"修复"(会改变金标准)。
8. `get_signals.py` 里的日期/断言是 11-05 的一次性实例; 每日请用 `daily_orders.py`(asof 自动取数据最新日)。

## 7. 状态快照(2026-11-05, 一次性; 以工具当日输出为准)

- 门控 **OFF**: 指数 4113 < ma60 4415 ✗; 广度 0.070 < 0.20 ✗ (mode=or; B0 阈值 0.25 同样 ✗)。
- 持仓(自 2026-09-14): `511260 十年国债ETF@135.885` + `518880 黄金ETF@8.420`; K5 权重 50/50, K3 权重 98.4/1.6(score)。
- 11-04/11-05 均无成交; 明日(11-06)计划同为 HOLD。
- 门控翻多 top3: K5 `515450(0.400) 515300(0.319) 512700(0.315)`; K3 `515450(0.400) 563510(0.388) 560570(0.379)`(差异源于 dyn_min_amt: K5 需 5 日均额≥5000 万, K3 无门槛)。
- 全历史净值: K3 41.42, K5 7.70。

## 8. 交付物关系

- 选股依据/窗口比较 → `report.md`(round-1), `report_orig_engine.md`(round-2 原版引擎优化, 63 配置)。
- 机器可读参数/管线/公式 → `strategy_config.json`(由引擎进程内导出, 改引擎后需同步再导)。
- 每日执行 → `daily_orders.py`(本手册 §0)。
- 本手册任何描述与 `S9_orig_backtest.py` 冲突 → 以源码为准, 并回改本手册。
