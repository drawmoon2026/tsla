"""What-if 沙盘预计算 — 全系统推演页「改参数即时重算」的数据层.

对推演交易流（outputs/replay_current/trades.csv：62 笔留出 + 前向新增，全部
gate 通过的入场事件）预结算参数网格：

    几何 {tp 0.5% / 0.75% / 1%} × {sl 1% / 1.5% / 2%} = 9 组合
    （冻结几何 tp0.5/sl2 必含并标记）

每事件记录 {prob, s2_off, 九组合各自的 (exit_pos, ret, exit_type, exit_et)}，
全部用 research/e8a_trades_export.detail_trade 同机制悲观结算
（src.common.execution.settle_bracket：费 1bp/边 + 滑点 2bp、TP 限价需严格
穿越、SL 停损市价含跳空、同 bar 双触发算 SL、timeout 48×5m、日内强平）。

页面 client 端从本文件产物即时重算：gate 阈值过滤 → 贪心非重叠流（用各组合
自己的 exit_pos，几何变了流也如实变）→ S2 开/关 → 笔数/胜率/总收益/年化/MDD。

诚实护栏（导出侧）：
- 冻结组合逐笔收益必须与 trades.csv 完全一致（1e-9），否则中止；
- 冻结基线（gate 0.4325 / tp0.5/sl2 / S2 开）在留出段的 n/WR/total 必须与
  research/e8a_trades_export.EXP_S2 存档一致，否则中止——保证沙盘基线行
  与总结卡数字同源同值。

用法：  .venv/bin/python -m intel.sandbox_export
输出：  outputs/replay_current/sandbox.json（约 65 事件 × 9 组合，几十 KB）
（intel/replay_refresh.py 每次续演后自动调用本导出，launchd 链无需改动。）
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "replay_current"
FALLBACK_DIR = ROOT / "outputs" / "e8a_replay"
ROLLING_CSV = ROOT / "data" / "TSLA_5m_rolling.csv"
SEED_CSV = ROOT / "data" / "TSLA_5m_3y.csv"
MODEL_META = ROOT / "models" / "e8a" / "meta.json"

TP_GRID = (0.005, 0.0075, 0.010)   # 行：tp 0.5% / 0.75% / 1%
SL_GRID = (0.010, 0.015, 0.020)    # 列：sl 1% / 1.5% / 2%
TIMEOUT = 48
FROZEN_TP, FROZEN_SL = 0.005, 0.020
GATE_UI = {"min": 0.40, "max": 0.60, "step": 0.0025, "default": 0.4325}


def detail_trade_geo(bs, epos: int, tp: float, sl: float) -> dict:
    """detail_trade 同口径、tp/sl 参数化（timeout/日内强平/成本全冻结）."""
    from research.e9_frontier_search import COST
    from src.common.execution import entry_fill, settle_bracket

    end = min(epos + TIMEOUT, bs.n)
    day = bs.et_dates[epos]
    off = np.nonzero(bs.et_dates[epos:end] != day)[0]
    day_cut = bool(len(off))
    if day_cut:
        end = epos + int(off[0])
    window = bs.bars.iloc[epos:end]
    entry_px = entry_fill(bs.opens[epos], 1, COST)
    res = settle_bracket(window, 1, entry_px, entry_px * (1 + tp),
                         entry_px * (1 - sl), COST)
    exit_pos = epos + window.index.get_loc(res.exit_time)
    exit_type = res.hit if res.hit in ("tp", "sl") else (
        "dayend" if (day_cut and exit_pos == end - 1) else "timeout")
    return {"exit_pos": int(exit_pos), "ret": float(res.ret),
            "exit_type": exit_type, "exit_t": res.exit_time}


def _greedy(events: list[dict], gate: float, ci: int,
            s2_on: bool) -> tuple[list[float], int]:
    """页面 JS 同一算法的 Python 参照实现：贪心非重叠 + S2 过滤.

    S2 开时被拦事件仍占流位（与冻结口径一致：照常结算、不入权益）。
    """
    busy, kept, n_blk = -1, [], 0
    for ev in events:
        if ev["p"] < gate or ev["epos"] <= busy:
            continue
        combo = ev["c"][ci]
        busy = combo[0]
        if s2_on and ev["s2"]:
            n_blk += 1
            continue
        kept.append(combo[1])
    return kept, n_blk


def _stats(rets: list[float], n_blk: int, cal_days: int) -> dict:
    if not rets:
        return {"n": 0, "n_blocked": n_blk, "wr": None, "total": 0.0,
                "ann": 0.0, "mdd": 0.0, "avg_bp": None}
    r = np.asarray(rets)
    eq = np.cumprod(1 + r)
    total = float(eq[-1] - 1)
    peak = np.maximum.accumulate(eq)
    return {"n": int(len(r)), "n_blocked": n_blk,
            "wr": float((r > 0).mean()), "total": total,
            "ann": float((1 + total) ** (365 / max(1, cal_days)) - 1),
            "mdd": float((eq / peak - 1).min()),
            "avg_bp": float(r.mean() * 1e4)}


def export_sandbox(bars=None, bs=None) -> Path:
    """产出 outputs/replay_current/sandbox.json；返回路径。

    bars/bs 可由 replay_refresh 传入复用（省一次加载）；独立运行时自己加载
    滚动数据（缺失退回冻结源）。交易流读 outputs/replay_current/trades.csv
    （缺失退回 outputs/e8a_replay/trades.csv，纯留出段）。
    """
    from research.e8a_trades_export import EXP_S2
    from research.e9_frontier_search import BarSet
    from src.common.data_io import ET, load_bars

    # ---- 交易流（= 全部 gate 通过、入了非重叠流的入场事件） -----------------
    src_dir = OUT_DIR if (OUT_DIR / "trades.csv").exists() else FALLBACK_DIR
    tr = pd.read_csv(src_dir / "trades.csv")
    if "segment" not in tr.columns:
        tr["segment"] = "holdout"
    assert tr["epos"].is_monotonic_increasing, "trades.csv epos 非升序"

    # ---- 窗口（年化/回撤口径用；沙盘窗口 = 留出段起 → 数据截至日） ----------
    daily = pd.read_csv(src_dir / "daily.csv")
    if "seg" in daily.columns:
        daily = daily[daily["seg"].isin(["holdout", "forward"])]
    d0, d1 = str(daily["et_day"].iloc[0]), str(daily["et_day"].iloc[-1])
    cal_days = (date.fromisoformat(d1) - date.fromisoformat(d0)).days

    # ---- bars（滚动优先；与交易流 epos 对齐防线） ----------------------------
    if bars is None or bs is None:
        src_bars = ROLLING_CSV if ROLLING_CSV.exists() else SEED_CSV
        print(f"[sandbox] 加载 {src_bars.name} ...")
        bars = load_bars(str(src_bars))
        bs = BarSet(bars)
    chk = bars.index.get_indexer(
        pd.DatetimeIndex(pd.to_datetime(tr["entry_t_utc"])))
    assert (chk == tr["epos"].to_numpy()).all(), "bars 与交易流 epos 错位——中止"

    gate_frozen = float(json.loads(MODEL_META.read_text(encoding="utf-8"))
                        ["gate"]["threshold"])
    combos = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]
    frozen_ci = combos.index((FROZEN_TP, FROZEN_SL))

    # ---- 逐事件 × 九组合预结算（冻结组合与 trades.csv 逐笔核对） ------------
    events = []
    for _, row in tr.iterrows():
        epos = int(row["epos"])
        cs = []
        for ci, (tp, sl) in enumerate(combos):
            d = detail_trade_geo(bs, epos, tp, sl)
            if ci == frozen_ci:
                # trades.csv ret 存 8 位小数：容差按其舍入半径（5e-9）放
                assert abs(d["ret"] - float(row["ret"])) < 6e-9, \
                    f"冻结组合结算与 trades.csv 漂移（epos={epos}）——中止"
            cs.append([d["exit_pos"], round(d["ret"], 8), d["exit_type"],
                       d["exit_t"].tz_convert(ET).strftime("%Y-%m-%d %H:%M")])
        events.append({"epos": epos, "p": float(row["prob"]),
                       "s2": int(bool(row["blocked"])),
                       "seg": str(row["segment"]),
                       "d": str(row["et_day"]),
                       "e": str(row["entry_et"])[-5:], "c": cs})

    # ---- 冻结基线：与存档/总结卡同源核对（漂移即中止，绝不静默） ------------
    base_rets, base_blk = _greedy(events, GATE_UI["default"], frozen_ci, True)
    baseline = _stats(base_rets, base_blk, cal_days)
    hold_rets = [ev["c"][frozen_ci][1] for ev in events
                 if ev["seg"] == "holdout" and not ev["s2"]]
    hs = _stats(hold_rets, 0, cal_days)
    ok = (hs["n"] == EXP_S2["n"] and abs(hs["wr"] - EXP_S2["wr"]) < 1e-9
          and abs(hs["total"] - EXP_S2["total"]) < 1e-6)
    print(f"[sandbox] 冻结基线（留出段）n={hs['n']} WR={hs['wr']:.2%} "
          f"total={hs['total']:+.2%} " + ("与存档一致" if ok else "与存档不一致"))
    if not ok:
        raise SystemExit("[sandbox] 冻结基线与 EXP_S2 存档漂移——中止导出")

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(src_dir.relative_to(ROOT)),
        "gate_frozen": round(gate_frozen, 6),
        "gate_ui": GATE_UI,
        "tp_grid": list(TP_GRID), "sl_grid": list(SL_GRID),
        "frozen_ci": frozen_ci, "timeout_bars": TIMEOUT,
        "window": {"start": d0, "end": d1, "cal_days": cal_days},
        "n_events": len(events),
        "baseline": {k: (round(v, 8) if isinstance(v, float) else v)
                     for k, v in baseline.items()},
        "events": events,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "sandbox.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    print(f"[sandbox] {len(events)} 事件 × {len(combos)} 组合 · 窗口 {d0} → {d1}"
          f"（{cal_days} 天）· 基线 {baseline['n']} 笔 "
          f"total {baseline['total']:+.2%} ann {baseline['ann']:+.2%} "
          f"mdd {baseline['mdd']:.2%}")
    print(f"written: {path}  ({path.stat().st_size:,} bytes)")
    return path


def main() -> None:
    export_sandbox()


if __name__ == "__main__":
    main()
