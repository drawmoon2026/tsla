#!/usr/bin/env python
"""E8-A+S2 留出段逐笔交易一次性导出（供 intel/dashboard 全系统推演页读取）.

数据来源与口径（全部复用既有冻结机制，不引入任何新参数）：
- 交易流：models/e8a/holdout_ref.csv 的 in_trade 事件（= E8-A 62 笔留出交易的
  入场 bar 序号），逐笔用 E9 结算机制（src.common.execution.settle_bracket，
  tp0.5%/sl2%/timeout48/日内强平，费用 1bp/边 + 滑点 2bp）在 data/TSLA_5m_3y.csv
  上重放出 入场/出场时刻、价格、出场类型、单笔净收益；
- S2 拦截：holdout_ref.csv s2_off 列（E11 冻结口径：距 252 日滚动高点回撤 >20%，
  前一交易日收盘决定）；被拦截的 8 笔照常结算、单列 blocked=1（对照显示用）；
- 存档核对（不一致即报错退出）：
    全 62 笔  vs outputs/e8_pooled/frontier_shift.csv（n=62, WR 80.65%, avg +15.97bp）
    S2 后 54 笔 vs outputs/e11_bear_switch/switch_table.csv S2 行
    （n=54, WR 83.33%, avg +21.31bp, total +12.04%, MTM MDD -3.68%）
- 每日收盘 + S2 状态（留出窗口）另存 daily.csv 供页面画价格曲线与关闸底纹。

用法：  .venv/bin/python research/e8a_trades_export.py
输出：  outputs/e8a_replay/{trades.csv, daily.csv, summary.json}
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.e8a_crash_test import mdd, mtm_equity_path  # noqa: E402
from research.e9_frontier_search import COST, BarSet, run_combo, seg_stats  # noqa: E402
from src.common.data_io import ET, load_bars  # noqa: E402
from src.common.execution import entry_fill, settle_bracket  # noqa: E402

REF_CSV = ROOT / "models" / "e8a" / "holdout_ref.csv"
BARS_CSV = ROOT / "data" / "TSLA_5m_3y.csv"
OUT = ROOT / "outputs" / "e8a_replay"

TP, SL, TIMEOUT = 0.005, 0.020, 48          # models/e8a/meta.json geometry
HOLDOUT_START = date(2025, 10, 1)

# 存档口径（复算不一致 → 报错退出，绝不静默带偏页面）
EXP_ALL = {"n": 62, "wr": 0.8064516129032258, "avg_bp": 15.965470720673224,
           "total": 0.10214092741428882}          # frontier_shift / switch_table S5
EXP_S2 = {"n": 54, "wr": 0.8333333333333334, "avg_bp": 21.307620566097704,
          "total": 0.120365289615876, "mtm_mdd": -0.036789355598045304}  # S2 行


def detail_trade(bs: BarSet, epos: int) -> dict:
    """与 BarSet.eval_trade 完全同口径的逐笔明细（多给出场类型/价格/时刻）。"""
    end = min(epos + TIMEOUT, bs.n)
    day = bs.et_dates[epos]
    off = np.nonzero(bs.et_dates[epos:end] != day)[0]
    day_cut = bool(len(off))
    if day_cut:
        end = epos + int(off[0])
    window = bs.bars.iloc[epos:end]
    entry_px = entry_fill(bs.opens[epos], 1, COST)
    res = settle_bracket(window, 1, entry_px, entry_px * (1 + TP),
                         entry_px * (1 - SL), COST)
    exit_pos = epos + window.index.get_loc(res.exit_time)
    # close = 到 timeout / 日终都没触发挂单：区分两种以便页面如实展示
    exit_type = res.hit if res.hit in ("tp", "sl") else (
        "dayend" if (day_cut and exit_pos == end - 1) else "timeout")
    return {"entry_px": entry_px, "exit_px": float(res.exit_px),
            "exit_pos": exit_pos, "exit_t": res.exit_time, "ret": float(res.ret),
            "exit_type": exit_type}


def close_mdd(rets: np.ndarray) -> float:
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF_CSV)
    bars = load_bars(str(BARS_CSV))
    bs = BarSet(bars)

    # ---- 62 笔交易流：holdout_ref in_trade 事件（同 epos 事件取门控概率最大值）
    t_ref = (ref[ref["in_trade"]]
             .groupby("epos", as_index=False)
             .agg(prob=("prob", "max"), s2_off=("s2_off", "any")))
    t_ref = t_ref.sort_values("epos").reset_index(drop=True)

    # run_combo 交叉验证：62 个 epos 本身应是非重叠序列，重放必须逐位一致
    combo = run_combo(bs, [(int(p), 1) for p in t_ref["epos"]], TP, SL, TIMEOUT)
    assert len(combo) == len(t_ref), "in_trade epos 序列存在重叠——与 run_combo 不一致"

    rows, prev_exit = [], -1
    for i, r in t_ref.iterrows():
        epos = int(r["epos"])
        assert epos > prev_exit, "交易重叠"
        d = detail_trade(bs, epos)
        assert abs(d["ret"] - float(combo.loc[i, "ret"])) < 1e-12, "重放收益不一致"
        prev_exit = d["exit_pos"]
        ts_e = bars.index[epos].tz_convert(ET)
        ts_x = d["exit_t"].tz_convert(ET)
        rows.append({
            "seq": i + 1, "epos": epos,
            "entry_t_utc": bars.index[epos].isoformat(),
            "entry_et": ts_e.strftime("%Y-%m-%d %H:%M"),
            "et_day": str(ts_e.date()),
            "prob": round(float(r["prob"]), 6),
            "entry_open": float(bs.opens[epos]),
            "entry_fill": round(d["entry_px"], 4),
            "exit_et": ts_x.strftime("%Y-%m-%d %H:%M"),
            "exit_px": round(d["exit_px"], 4),
            "exit_type": d["exit_type"],
            "ret": d["ret"],
            "blocked": bool(r["s2_off"]),
        })
    tr = pd.DataFrame(rows)

    # ---- 存档核对：62 笔全流 / S2 过滤后 54 笔 -------------------------------
    def check(tag: str, got: dict, exp: dict, tol_bp: float = 0.05) -> None:
        ok = (got["n"] == exp["n"] and abs(got["wr"] - exp["wr"]) < 1e-9
              and abs(got["avg_bp"] - exp["avg_bp"]) < tol_bp
              and abs(got["total"] - exp["total"]) < 1e-6)
        line = (f"[{tag}] n={got['n']} WR={got['wr']:.2%} avg={got['avg_bp']:+.2f}bp "
                f"total={got['total']:+.2%}  vs 存档 n={exp['n']} WR={exp['wr']:.2%} "
                f"avg={exp['avg_bp']:+.2f}bp total={exp['total']:+.2%}")
        print(line, "一致" if ok else "不一致")
        if not ok:
            raise SystemExit(f"[{tag}] 与存档不一致——中止导出")

    def stats(sub: pd.DataFrame) -> dict:
        s = seg_stats(sub.rename(columns={"ret": "ret"}), "x")
        return {"n": s["x_n"], "wr": s["x_wr"], "avg_bp": s["x_avg_bp"],
                "total": s["x_total"], "mdd_closed": close_mdd(sub["ret"].to_numpy())}

    all_st = stats(tr)
    check("全 62 笔（S5 无开关）", all_st, EXP_ALL)
    kept = tr[~tr["blocked"]].reset_index(drop=True)
    s2_st = stats(kept)
    check("S2 过滤后 54 笔", s2_st, EXP_S2)

    # bar 级盯市 MDD（与 switch_table hold_mtm_mdd 同口径）
    mk = kept[["epos"]].copy()
    mtm_s2 = mdd(mtm_equity_path(bs, mk, TP, SL, TIMEOUT))
    mtm_all = mdd(mtm_equity_path(bs, tr[["epos"]].copy(), TP, SL, TIMEOUT))
    assert abs(mtm_s2 - EXP_S2["mtm_mdd"]) < 1e-9, "S2 盯市 MDD 与存档不一致"
    print(f"[MTM MDD] S2 后 {mtm_s2:+.2%}（存档一致）  全流 {mtm_all:+.2%}")

    # S2 流内累计权益（页面推进用；拦截笔不入权益）
    eqs = np.cumprod(1 + kept["ret"].to_numpy())
    tr["cum_eq_s2"] = np.nan
    tr.loc[~tr["blocked"], "cum_eq_s2"] = eqs
    tr["ret"] = tr["ret"].round(8)
    tr["cum_eq_s2"] = tr["cum_eq_s2"].round(8)

    # ---- 每日收盘 + S2 状态（E11 switch_states 同式：shift(1)，252 日回撤）----
    et_days = bars.index.tz_convert(ET).date
    dc = bars["Close"].groupby(et_days).last()
    s2_state = (dc / dc.rolling(252).max() - 1.0 < -0.20).shift(1).fillna(False)
    daily = pd.DataFrame({"et_day": [str(d) for d in dc.index],
                          "close": dc.round(4).to_numpy(),
                          "s2_off": s2_state.astype(bool).to_numpy()})
    daily = daily[daily["et_day"] >= str(HOLDOUT_START)].reset_index(drop=True)
    # 与 holdout_ref 的 s2_off 交叉验证（事件日粒度）
    ref_day = ref.groupby("et_day")["s2_off"].any()
    chk = daily.set_index("et_day")["s2_off"].reindex(ref_day.index)
    assert (chk.dropna() == ref_day[chk.notna()]).all(), "S2 日状态与 holdout_ref 不一致"

    # ---- 汇总 JSON -----------------------------------------------------------
    d0, d1 = daily["et_day"].iloc[0], daily["et_day"].iloc[-1]
    span_days = (date.fromisoformat(d1) - date.fromisoformat(d0)).days
    blocked = tr[tr["blocked"]]
    dist = kept["exit_type"].value_counts().to_dict()
    summary = {
        "source": ("holdout_ref.csv in_trade 事件 × E9 结算重放；"
                   "与 frontier_shift.csv / e11 switch_table.csv 存档核对一致"),
        "window": {"start": d0, "end": d1, "cal_days": span_days,
                   "trading_days": int(len(daily))},
        "all": {**all_st, "mtm_mdd": mtm_all},
        "s2": {**s2_st, "mtm_mdd": mtm_s2},
        "blocked": {"n": int(len(blocked)),
                    "total": float(np.prod(1 + blocked["ret"]) - 1),
                    "sum_ret": float(blocked["ret"].sum())},
        "exit_dist_s2": {k: int(v) for k, v in dist.items()},
        "ann_10mo": float((1 + s2_st["total"]) ** (12 / 10) - 1),
        "ann_actual": float((1 + s2_st["total"]) ** (365 / span_days) - 1),
        "geometry": {"tp": TP, "sl": SL, "timeout_bars": TIMEOUT,
                     "cost": "费 1bp/边 + 滑点 2bp（入场与止损出场）",
                     "day_flat": True},
    }

    tr.to_csv(OUT / "trades.csv", index=False)
    daily.to_csv(OUT / "daily.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"窗口 {d0} → {d1}（{span_days} 天 / {len(daily)} 交易日）")
    print(f"出场分布（S2 后）：{dist}；拦截 {len(blocked)} 笔 合计乘积 "
          f"{summary['blocked']['total']:+.2%}")
    print(f"年化：10个月口径 {summary['ann_10mo']:+.2%}  实际天数口径 "
          f"{summary['ann_actual']:+.2%}")
    print(f"written: {OUT}/trades.csv, daily.csv, summary.json")


if __name__ == "__main__":
    main()
