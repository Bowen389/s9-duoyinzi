# Decision block ported from uploaded S9 engine; consumes actual paper holdings.
import numpy as np

def decide(bt, t, held, entry_px, entry_t, last_exit):
    par = bt.p
    peak, brake_scale = 1., 1.
    if par["dd_brake"]:
        raise ValueError("drawdown brake requires paper-equity adapter")
    # --- 收盘决策 ---
    rank = bt._rank(t)
    rank_pos = {c: i for i, c in enumerate(rank[: par["top_n"] * 3])}
    keep, why = [], {}
    for c in held:
        v = bt.close[t, c]
        # 止损
        if par["stop_loss"] and np.isfinite(entry_px[c]):
            trig = (v / entry_px[c] - 1 <= -par["stop_loss"]) if par["stop_mode"] == "close" \
                else (bt.low[t, c] / entry_px[c] - 1 <= -par["stop_loss"])
            if trig:
                last_exit[c] = t
                why[c] = "止损"
                continue
        # 放量出场
        if par["use_vol_filter"] and bool(np.nan_to_num(bt.vol_spike[t, c])):
            last_exit[c] = t
            why[c] = "放量离场"
            continue
        # 最小持仓期
        if t - entry_t[c] < par["min_hold"]:
            keep.append(c)
            continue
        # 趋势破了也走
        if not bt.trend_ok[t, c]:
            last_exit[c] = t
            why[c] = "趋势破位"
            continue
        keep.append(c)

    if not par["use_rank_exit"]:
        add = [c for c in rank if c not in keep][: max(par["top_n"] - len(keep), 0)]
        target = keep + add
    else:
        # 迟滞：新候选需明显更强才换
        inc_score = [bt.score[t, c] for c in keep]
        base = min(inc_score) if inc_score else -np.inf
        add = []
        for c in rank:
            if c in keep:
                continue
            if c in add:
                continue
            if base is not -np.inf and bt.score[t, c] < base * (1 + par["switch_margin"]):
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
            keep_sorted = sorted(keep, key=lambda c: -bt.score[t, c])
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
    if not bt.regime[t]:
        target = []
        target = bt._defensive(t)

    # 每周调仓模式：非调仓日只做风控卖出，不加仓
    if par["rebalance_dow"] is not None and bt.dates[t].dayofweek != par["rebalance_dow"]:
        if set(target) != set(held):
            # 仅允许减仓
            target = [c for c in target if c in set(held)] or []
            if target == [] and len(held) > 0:
                target = list(held) if par["use_rank_exit"] else []
        else:
            target = list(held)


    changed = set(target) != set(held)
    weights = bt._weights(t, target) if changed else None
    return target, weights, why, last_exit
