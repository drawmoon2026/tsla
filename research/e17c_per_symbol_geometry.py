#!/usr/bin/env python
"""E17-C — 池逐标的几何重标定（原 E16，用户经 E17「三十计划」指令批准）.

E15 剩余病灶的对症实验：冻结几何 tp0.5%/sl2%（TSLA 调出）在池标的上单笔期望为负
（WR ~72% 但 -3.3bp，几何幻觉）。本实验允许每标的换几何，且允许"这只股票我做不了"。

协议（防挖掘装订，全套预登记）：
1. 几何候选集受限（非自由网格）：每标的从五种几何
     {tp0.5/sl1, tp0.5/sl2(冻结), tp0.75/sl1.5, tp1/sl2, tp1/sl1}
   中选一；timeout 48 bar / 日内强平 / 悲观结算 / 成本模型全部冻结不动。
   选择依据只用训练段（et_day < 2025-09-30）：每标的训练段事件经 E15 分位 gate
   过滤、按 epos 去重后，五几何各自悲观结算的单笔期望（独立结算均值），
   取期望最高者；五几何期望全非正的标的剔除出组合。
2. 留出段（2025-10-01 → 数据尾）：入选标的用训练段选定几何 + E15 分位 gate +
   独立 S2，组合模拟复用 research/e14_pool_portfolio 引擎（引用不复制），
   N ∈ {4, 8} 两档预登记，$100k 账本等额分片。
3. 冻结自校验与 E14/E15 相同：TSLA 逐事件 prob 对 holdout_ref；TSLA 冻结几何
   单标的 62 笔 / S2 后 54 笔 +12.04% 存档复现（锚用冻结几何，与 TSLA 本实验
   选中几何无关）。
4. 诚实边界：训练段选几何 = 样本内选择（26 标的 × 5 几何 = 130 次比较，如实
   记账）；池标的训练段 prob 为 in-sample（模型见过这些事件）；留出段单次检验
   为方向性证据；训练段期望与留出段实际的秩相关（选择有效性）一并报告。
   多重比较 +130（几何选择）+2（N 两档）= +132。

用法:  .venv/bin/python research/e17c_per_symbol_geometry.py
输出:  outputs/e17c_geometry/{trades.csv, equity.csv, per_symbol.csv, summary.txt}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import research.e8_pooled_gbdt as e8  # noqa: E402
import research.e14_pool_portfolio as e14  # noqa: E402  组合引擎（引用不复制）
from research.e11_bear_switch import daily_closes, switch_states  # noqa: E402
from research.e9_frontier_search import COST, BarSet  # noqa: E402
from research.ml_common import build_dataset, load_events  # noqa: E402
from src.common.data_io import ET, load_bars  # noqa: E402
from src.common.execution import entry_fill, settle_bracket  # noqa: E402

OUT = ROOT / "outputs" / "e17c_geometry"

KEEP_FRAC = 0.10          # E15 分位 gate：继承冻结设计 top~10% 规则
MIN_TRAIN_EVENTS = 50     # E15 兜底规则原样沿用
TIMEOUT = int(e14.META["geometry"]["timeout_bars"])   # 48，冻结不动

# 几何候选集（预登记受限五选一；tp0.5/sl2.0 = 冻结几何）
GEOMS = [
    ("tp0.5/sl1.0", 0.005, 0.010),
    ("tp0.5/sl2.0", 0.005, 0.020),
    ("tp0.75/sl1.5", 0.0075, 0.015),
    ("tp1.0/sl2.0", 0.010, 0.020),
    ("tp1.0/sl1.0", 0.010, 0.010),
]
FROZEN_GEOM = "tp0.5/sl2.0"

E15_PER_SYM = ROOT / "outputs" / "e15_percentile_gate" / "per_symbol.csv"

# E15 存档锚点（outputs/e15_percentile_gate/summary.txt，2026-08-01）
E15_REF = {
    4: {"total": -0.1111, "ann": -0.1356, "mtm_mdd": -0.1189, "n": 1362,
        "tsla_n": 35},
    8: {"total": -0.0827, "ann": -0.1013, "mtm_mdd": -0.0856, "n": 1583,
        "tsla_n": 51},
}
# E14 存档（绝对阈值 gate，最早对照）
E14_TOTALS = {4: -0.2170, 8: -0.1890}

ANN_TARGET = 0.30   # E17 北极星
ANN_PASS = 0.08     # 达标线（已验证最优 ~14.6% 的下限参照）


def detail_trade_geom(bs: BarSet, epos: int, tp: float, sl: float) -> dict | None:
    """e14.detail_trade 的参数化版本（tp/sl 可变，timeout/日内强平/成本冻结）。"""
    end = min(epos + TIMEOUT, bs.n)
    day = bs.et_dates[epos]
    off = np.nonzero(bs.et_dates[epos:end] != day)[0]
    day_cut = bool(len(off))
    if day_cut:
        end = epos + int(off[0])
    window = bs.bars.iloc[epos:end]
    if window.empty:
        return None
    entry_px = entry_fill(bs.opens[epos], 1, COST)
    res = settle_bracket(window, 1, entry_px, entry_px * (1 + tp),
                         entry_px * (1 - sl), COST)
    exit_pos = epos + window.index.get_loc(res.exit_time)
    chk = bs.eval_trade(epos, 1, tp, sl, TIMEOUT)
    assert chk is not None and abs(chk[0] - res.ret) < 1e-12 and chk[1] == exit_pos, \
        "detail_trade_geom 与 BarSet.eval_trade 不一致"
    exit_type = res.hit if res.hit in ("tp", "sl") else (
        "dayend" if (day_cut and exit_pos == end - 1) else "timeout")
    return {"entry_px": entry_px, "ret": float(res.ret), "exit_pos": exit_pos,
            "exit_t": bs.idx[exit_pos], "exit_type": exit_type}


def geom_rets(bs: BarSet, eposes: np.ndarray, tp: float, sl: float) -> np.ndarray:
    """一组候选 epos 在给定几何下的独立悲观结算收益（不可结算的剔除）。"""
    out = []
    for e in eposes:
        r = bs.eval_trade(int(e), 1, tp, sl, TIMEOUT)
        if r is not None:
            out.append(r[0])
    return np.asarray(out, dtype=float)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 每标的冻结管线：事件 -> 特征（与 E14/E15 同源）----------------
    frames, bsets, s2_states = [], {}, {}
    for sym in e14.SYMS:
        if sym == e8.TARGET:
            bars_csv, events_csv = e8.TSLA_BARS, e8.TSLA_EVENTS
        else:
            bars_csv = e8.POOL_DIR / f"{sym}_5m_3y.csv"
            events_csv = e8.EVENTS_POOL_DIR / sym / "v_events_5m_unique.csv"
        bars = load_bars(str(bars_csv))
        ds, _ = build_dataset(load_events(Path(events_csv)), bars)
        ds["symbol"] = sym
        frames.append(ds)
        bsets[sym] = BarSet(bars)
        s2_states[sym] = switch_states(daily_closes(bars))["S2"]  # 冻结 S2 规则
        print(f"[{sym}] events={len(ds)}  bars={len(bars)}")

    all_ds = e8.add_normalized_features(pd.concat(frames, ignore_index=True))
    feat_ok = all_ds[e8.POOLED_FEATURES[:-1]].notna().all(axis=1)
    all_ds = all_ds[feat_ok].reset_index(drop=True)

    assert list(all_ds["symbol"].cat.categories) == e14.META["symbol_categories"], \
        "symbol 类别表与 meta.json 不一致"
    tsla_med = float(all_ds.loc[all_ds["symbol"] == "TSLA", "rv20_med_train"].iloc[0])
    assert abs(tsla_med - e14.META["tsla_rv20_med_train"]) < 1e-12, \
        "TSLA rv20 训练段中位数与 meta.json 不一致"

    # ---------- 冻结模型推断：训练段（定阈值+选几何）+ 评估段（交易用）--------
    model = joblib.load(ROOT / "models" / "e8a" / "model.joblib")
    all_ds["prob"] = model.predict_proba(all_ds[e8.POOLED_FEATURES])[:, 1]
    train = all_ds[all_ds["et_day"] < e8.TRAIN_END]
    ev = all_ds[all_ds["et_day"] >= e14.EVAL_START].reset_index(drop=True)

    # 逐事件 prob 对照 holdout_ref.csv（冻结自校验，与 E14/E15 相同）
    ref = pd.read_csv(ROOT / "models" / "e8a" / "holdout_ref.csv")
    ref["entry_t"] = pd.to_datetime(ref["entry_t"], utc=True)
    tsla_ev = ev[ev["symbol"] == "TSLA"]
    assert len(tsla_ev) == len(ref), "TSLA 留出事件数与 holdout_ref 不一致"
    a = ref.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    b = tsla_ev.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    assert np.abs(a - b).max() < 1e-9, "冻结模型 prob 与 holdout_ref 不一致"
    print(f"[check] TSLA {len(tsla_ev)} 事件 prob 与 holdout_ref 完全一致")

    # ---------- E15 分位 gate 阈值（逻辑原样沿用，非新选择）-------------------
    pool_train_probs = train.loc[train["symbol"] != "TSLA", "prob"].to_numpy()
    fallback_thr = float(np.quantile(pool_train_probs, 1.0 - KEEP_FRAC))
    thr_map, fallback_syms = {}, []
    for sym in e14.SYMS:
        p_tr = train.loc[train["symbol"] == sym, "prob"].to_numpy()
        if sym == "TSLA":
            thr_map[sym] = e14.THR
        elif len(p_tr) < MIN_TRAIN_EVENTS:
            thr_map[sym] = fallback_thr
            fallback_syms.append(sym)
        else:
            thr_map[sym] = float(np.quantile(p_tr, 1.0 - KEEP_FRAC))
    print(f"[thr] 池兜底标的: {fallback_syms if fallback_syms else '无'}")

    # ---------- E17-C 核心：训练段五几何期望 -> 逐标的选几何 / 剔除 ------------
    sel_rows = {}
    for sym in e14.SYMS:
        sub = train[train["symbol"] == sym].copy()
        bs = bsets[sym]
        sub["epos"] = bs.idx.get_indexer(pd.DatetimeIndex(sub["entry_t"]))
        assert (sub["epos"] >= 0).all()
        gate = sub[sub["prob"] >= thr_map[sym]]
        cand_epos = np.sort(gate["epos"].unique())          # epos 去重（同 E15 cand）
        row = {"n_train_events": len(sub), "n_train_cand": len(cand_epos)}
        means = {}
        for gname, tp, sl in GEOMS:
            rr = geom_rets(bs, cand_epos, tp, sl)
            means[gname] = float(rr.mean()) if len(rr) else np.nan
            row[f"train_bp[{gname}]"] = means[gname] * 1e4 if len(rr) else np.nan
        best = max(means, key=lambda g: (means[g] if np.isfinite(means[g]) else -np.inf))
        row["best_geom"] = best
        row["train_bp_best"] = means[best] * 1e4
        row["included"] = bool(np.isfinite(means[best]) and means[best] > 0)
        sel_rows[sym] = row
        tag = "入选" if row["included"] else "剔除"
        print(f"[select] {sym:<6s} {tag}  best={best} "
              f"{means[best] * 1e4:+.2f}bp  (train_cand={len(cand_epos)})")
    sel_df = pd.DataFrame(sel_rows).T
    included = [s for s in e14.SYMS if sel_rows[s]["included"]]
    excluded = [s for s in e14.SYMS if not sel_rows[s]["included"]]
    geom_of = {s: sel_rows[s]["best_geom"] for s in e14.SYMS}
    geom_px = {g: (tp, sl) for g, tp, sl in GEOMS}

    # ---------- 留出段：E15 gate + epos + S2（结构同 E15）---------------------
    per_sym_rows, cand_frames = [], []
    for sym in e14.SYMS:
        sub = ev[ev["symbol"] == sym].copy()
        bs = bsets[sym]
        sub["epos"] = bs.idx.get_indexer(pd.DatetimeIndex(sub["entry_t"]))
        assert (sub["epos"] >= 0).all()
        gate = sub[sub["prob"] >= thr_map[sym]]
        cand = (gate.groupby("epos", as_index=False)
                .agg(prob=("prob", "max"), entry_t=("entry_t", "first"),
                     et_day=("et_day", "first")))
        cand["sym"] = sym
        cand["s2_off"] = cand["et_day"].map(s2_states[sym]).fillna(False).astype(bool)
        cand_frames.append(cand)
        per_sym_rows.append({
            "symbol": sym,
            "scope": "时间外+标的外" if sym == "TSLA" else "时间外·标的内",
            "thr": thr_map[sym],
            "n_events_eval": len(sub), "n_gate_pass": len(gate),
            "retention": len(gate) / len(sub) if len(sub) else np.nan,
            "n_cand": len(cand), "n_s2_blocked": int(cand["s2_off"].sum()),
        })
    cands_all = pd.concat(cand_frames, ignore_index=True)
    per_sym = pd.DataFrame(per_sym_rows).set_index("symbol").join(sel_df)

    # ---------- TSLA 冻结几何存档复现（锚与本实验选中几何无关）----------------
    tc = cands_all[cands_all["sym"] == "TSLA"]
    frozen_details = {}
    for r in tc.itertuples():
        d = e14.detail_trade(bsets["TSLA"], int(r.epos))
        assert d is not None
        frozen_details[("TSLA", int(r.epos))] = d
    rep = e14.seq_replay(tc, frozen_details)
    wr, avg = float((rep["ret"] > 0).mean()), float(rep["ret"].mean() * 1e4)
    assert len(rep) == e14.EXP_TSLA["n"] and abs(wr - e14.EXP_TSLA["wr"]) < 1e-9 \
        and abs(avg - e14.EXP_TSLA["avg_bp"]) < 0.05, "TSLA 62 笔存档复现失败"
    keep = rep[~rep["et_day"].map(s2_states["TSLA"]).fillna(False).astype(bool)]
    tot_s2 = float(np.prod(1 + keep["ret"]) - 1)
    assert len(keep) == e14.EXP_TSLA["n_s2"] \
        and abs(tot_s2 - e14.EXP_TSLA["total_s2"]) < 1e-6, "TSLA S2 后 54 笔复现失败"
    print(f"[check] TSLA 冻结几何复现: 62 笔 WR {wr:.2%} avg {avg:+.2f}bp；"
          f"S2 后 54 笔 total {tot_s2:+.2%} —— 与存档一致")

    # ---------- 入选标的：选中几何逐候选结算明细 -------------------------------
    details, drop_unsettle = {}, 0
    inc_cands = cands_all[cands_all["sym"].isin(included)].reset_index(drop=True)
    keep_mask = np.ones(len(inc_cands), dtype=bool)
    for i, r in enumerate(inc_cands.itertuples()):
        tp, sl = geom_px[geom_of[r.sym]]
        d = detail_trade_geom(bsets[r.sym], int(r.epos), tp, sl)
        if d is None:                       # 极端尾部不可结算，剔除并计数
            keep_mask[i] = False
            drop_unsettle += 1
            continue
        details[(r.sym, int(r.epos))] = d
    inc_cands = inc_cands[keep_mask].reset_index(drop=True)
    if drop_unsettle:
        print(f"[warn] 留出段不可结算候选剔除 {drop_unsettle} 个")

    # ---------- 选择有效性：训练段期望 vs 留出段实际（独立结算口径）------------
    hold_bp = {}
    for sym in e14.SYMS:
        cc = cands_all[(cands_all["sym"] == sym) & (~cands_all["s2_off"])]
        tp, sl = geom_px[geom_of[sym]]
        rr = geom_rets(bsets[sym], cc["epos"].to_numpy(), tp, sl)
        hold_bp[sym] = float(rr.mean() * 1e4) if len(rr) else np.nan
    per_sym["hold_bp_best_geom"] = pd.Series(hold_bp)
    v_all = per_sym[["train_bp_best", "hold_bp_best_geom"]].dropna().astype(float)
    rho_all, p_all = spearmanr(v_all["train_bp_best"], v_all["hold_bp_best_geom"])
    v_inc = v_all.loc[[s for s in included if s in v_all.index]]
    rho_inc, p_inc = spearmanr(v_inc["train_bp_best"], v_inc["hold_bp_best_geom"])

    # ---------- 组合模拟（e14.simulate / e14.mtm_paths，两档 N）---------------
    live = inc_cands[~inc_cands["s2_off"]].reset_index(drop=True)
    times = [bs.idx[bs.et_dates >= e14.EVAL_START] for bs in bsets.values()]
    timeline = pd.DatetimeIndex(sorted(set().union(*[set(t) for t in times])))
    tl_days = np.asarray(timeline.tz_convert(ET).date)
    cal_days = (tl_days[-1] - tl_days[0]).days
    months = cal_days / 30.4375

    runs = {}
    eq_daily = pd.DataFrame(index=pd.Index(sorted(set(tl_days)), name="et_day"))
    day_last = pd.Series(np.arange(len(timeline)), index=tl_days).groupby(level=0).last()
    for ncap in e14.N_CAPS:
        res = e14.simulate(live, details, ncap)
        tr = res["trades"]
        eq_path, cnt_path, posval = e14.mtm_paths(tr, bsets, timeline)
        assert abs(eq_path[-1] - res["final_eq"]) < 1e-6
        assert abs(eq_path[-1] - (e14.CAPITAL + tr["pnl"].sum())) < 1e-6
        runs[ncap] = {
            **res,
            "total": eq_path[-1] / e14.CAPITAL - 1.0,
            "ann": (eq_path[-1] / e14.CAPITAL) ** (365 / cal_days) - 1.0,
            "mtm_mdd": e14.mdd(eq_path),
            "avg_conc": float(cnt_path.mean()), "max_conc": int(cnt_path.max()),
            "util": float((posval / eq_path).mean()),
            "per_month": len(tr) / months,
        }
        eq_daily[f"eq_n{ncap}"] = eq_path[day_last.to_numpy()]
        print(f"[N={ncap}] trades={len(tr)}  total={runs[ncap]['total']:+.2%}  "
              f"ann={runs[ncap]['ann']:+.2%}  mtm_mdd={runs[ncap]['mtm_mdd']:+.2%}  "
              f"skip_cap={res['skip_cap']}")

    # ---------- 对照：池 26 等权 B&H / TSLA B&H（同段，不计成本）--------------
    dcs = {}
    for sym, bs in bsets.items():
        mask = bs.et_dates >= e14.EVAL_START
        first = int(np.nonzero(mask)[0][0])
        dc = pd.Series(bs.closes[mask],
                       index=np.asarray(bs.et_dates)[mask]).groupby(level=0).last()
        dcs[sym] = dc / bs.opens[first]
    bh = pd.DataFrame(dcs).reindex(eq_daily.index).ffill()
    eq_daily["bh_pool_26"] = bh.mean(axis=1) * e14.CAPITAL
    eq_daily["bh_tsla"] = bh["TSLA"] * e14.CAPITAL
    bh_pool_total = eq_daily["bh_pool_26"].iloc[-1] / e14.CAPITAL - 1
    bh_tsla_total = eq_daily["bh_tsla"].iloc[-1] / e14.CAPITAL - 1
    bh_pool_mdd = e14.mdd(eq_daily["bh_pool_26"].to_numpy())
    bh_tsla_mdd = e14.mdd(eq_daily["bh_tsla"].to_numpy())

    # ---------- 逐标的贡献表（含 E15 对照列）---------------------------------
    for ncap in e14.N_CAPS:
        tr = runs[ncap]["trades"]
        g = tr.groupby("symbol")
        per_sym[f"n{ncap}_trades"] = g.size().reindex(per_sym.index).fillna(0).astype(int)
        per_sym[f"n{ncap}_wr"] = g.apply(lambda x: float((x["ret"] > 0).mean()),
                                         include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_avg_bp"] = g.apply(lambda x: float(x["ret"].mean() * 1e4),
                                             include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_pnl"] = g["pnl"].sum().reindex(per_sym.index).fillna(0.0)
    e15_ps = pd.read_csv(E15_PER_SYM).set_index("symbol")
    per_sym["e15_n4_pnl"] = e15_ps["n4_pnl"]
    per_sym["e15_n8_pnl"] = e15_ps["n8_pnl"]
    per_sym["d_n4_pnl"] = per_sym["n4_pnl"] - per_sym["e15_n4_pnl"]

    # ---------- 输出 ----------------------------------------------------------
    trades_all = pd.concat([runs[n]["trades"] for n in e14.N_CAPS], ignore_index=True)
    trades_all["geom"] = trades_all["symbol"].map(geom_of)
    trades_all["entry_et"] = pd.DatetimeIndex(trades_all["entry_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    trades_all["exit_et"] = pd.DatetimeIndex(trades_all["exit_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    cols = ["n_cap", "symbol", "geom", "entry_et", "exit_et", "et_day", "prob",
            "alloc", "entry_px", "exit_type", "ret", "pnl", "eq_at_entry",
            "epos", "exit_pos"]
    trades_all[cols].to_csv(OUT / "trades.csv", index=False, float_format="%.6f")
    eq_daily.round(2).to_csv(OUT / "equity.csv")
    per_sym.reset_index().to_csv(OUT / "per_symbol.csv", index=False,
                                 float_format="%.6f")

    # ---------- summary（中文判读）-------------------------------------------
    d0, d1 = str(tl_days[0]), str(tl_days[-1])
    r4, r8 = runs[4], runs[8]
    n_sig = len(live)

    L = []
    L.append("E17-C — 池逐标的几何重标定（原 E16；冻结 E8-A 模型 + E15 分位 gate + 独立 S2）")
    L.append(f"窗口 {d0} → {d1}（{cal_days} 天 / {len(eq_daily)} 交易日 / {months:.1f} 个月）；"
             f"账本 ${e14.CAPITAL:,.0f}，等额分片，N∈{{4,8}} 两档预登记 —— "
             "协议与 E15 逐项一致，唯一改动 = 逐标的几何（五选一，训练段定）+ 允许剔除标的")
    L.append("几何候选集（预登记受限）: " + "、".join(g for g, _, _ in GEOMS)
             + f"；timeout {TIMEOUT}bar / 日内强平 / 悲观结算 / 成本(1bp费+2bp滑点)冻结不动")
    L.append("选择规则：每标的训练段（<2025-09-30）事件经 E15 分位 gate 过滤、epos 去重后，"
             "五几何独立悲观结算的单笔期望取最高；期望全非正 → 剔除出组合")
    L.append("冻结自校验：TSLA 逐事件 prob 与 holdout_ref 一致；TSLA 冻结几何单标的 62 笔 / "
             "S2 后 54 笔 +12.04% 与存档一致 —— 管线仅几何选择层改动确认")
    L.append("")
    L.append(f"== 训练段选几何结果：入选 {len(included)}/26，剔除 {len(excluded)} ==")
    L.append(f"剔除清单（训练段五几何期望全非正）: "
             + ("、".join(f"{s}(best {geom_of[s]} "
                          f"{sel_rows[s]['train_bp_best']:+.1f}bp)" for s in excluded)
                if excluded else "无"))
    L.append("")
    L.append("== 逐标的五几何训练段期望（bp）与选中几何 ==")
    hdr = f"{'symbol':<7s}{'训练cand':>8s}" + "".join(f"{g:>14s}" for g, _, _ in GEOMS) \
        + f"{'选中':>14s}{'留出实际bp':>10s}{'入选':>5s}"
    L.append(hdr)
    for sym in e14.SYMS:
        r = per_sym.loc[sym]
        cells = "".join(f"{r[f'train_bp[{g}]']:>14.2f}" for g, _, _ in GEOMS)
        hb = r["hold_bp_best_geom"]
        L.append(f"{sym:<8s}{int(r['n_train_cand']):>7d}{cells}"
                 f"{geom_of[sym]:>14s}"
                 f"{(hb if np.isfinite(hb) else float('nan')):>10.2f}"
                 f"{'是' if r['included'] else '否':>4s}")
    n_frozen_kept = sum(1 for s in included if geom_of[s] == FROZEN_GEOM)
    L.append(f"入选 {len(included)} 标的中保留冻结几何 {FROZEN_GEOM} 的: {n_frozen_kept} 个；"
             "其余训练段更偏好 " + "、".join(sorted({geom_of[s] for s in included
                                                     if geom_of[s] != FROZEN_GEOM})))
    L.append("")
    L.append("== 组合指标（两档并发，vs E15 存档）==")
    L.append(f"{'指标':<22s}" + "".join(f"N={n:<12d}" for n in e14.N_CAPS)
             + "".join(f"E15·N={n:<8d}" for n in e14.N_CAPS))

    def row(name, fn, e15fn):
        L.append(f"{name:<24s}" + "".join(f"{fn(runs[n]):<13s}" for n in e14.N_CAPS)
                 + "".join(f"{e15fn(E15_REF[n]):<13s}" for n in e14.N_CAPS))
    row("总收益", lambda r: f"{r['total']:+.2%}", lambda e: f"{e['total']:+.2%}")
    row("年化（实际天数）", lambda r: f"{r['ann']:+.2%}", lambda e: f"{e['ann']:+.2%}")
    row("组合 MTM 最大回撤", lambda r: f"{r['mtm_mdd']:+.2%}", lambda e: f"{e['mtm_mdd']:+.2%}")
    row("总笔数", lambda r: f"{len(r['trades'])}", lambda e: f"{e['n']}")
    row("笔/月", lambda r: f"{r['per_month']:.1f}", lambda e: f"{e['n'] / months:.1f}")
    row("WR", lambda r: f"{(r['trades']['ret'] > 0).mean():.1%}", lambda e: "-")
    row("单笔均值 bp", lambda r: f"{r['trades']['ret'].mean() * 1e4:+.2f}", lambda e: "-")
    row("平均/最大并发", lambda r: f"{r['avg_conc']:.2f} / {r['max_conc']}", lambda e: "-")
    row("平均资金利用率", lambda r: f"{r['util']:.1%}", lambda e: "-")
    row("容量跳过信号数", lambda r: f"{r['skip_cap']}", lambda e: "-")
    row("同标的占用跳过", lambda r: f"{r['skip_busy']}", lambda e: "-")
    L.append("")
    L.append(f"S2 后组合候选信号 {n_sig} 个（入选 {len(included)} 标的；E15 全 26 标的为 "
             f"2049 个）；{n_sig / months:.1f} 信号/月")
    L.append("")
    L.append("== 两种口径分开报告（组合内贡献拆分）==")
    for n in e14.N_CAPS:
        tr = runs[n]["trades"]
        t_t = e14.blk_stats(tr[tr["symbol"] == "TSLA"])
        t_p = e14.blk_stats(tr[tr["symbol"] != "TSLA"])
        tsla_note = (f"{t_t['n']} 笔, WR {t_t['wr']:.1%}, {t_t['avg_bp']:+.2f}bp, "
                     f"PnL ${t_t['pnl']:+,.0f}" if t_t["n"] else
                     "0 笔（TSLA 被训练段选几何剔除）")
        L.append(f"  N={n}  TSLA（时间外+标的外）: {tsla_note}"
                 f"（E15 同档 {E15_REF[n]['tsla_n']} 笔）")
        L.append(f"        池（时间外·标的内）: {t_p['n']} 笔, WR {t_p['wr']:.1%}, "
                 f"{t_p['avg_bp']:+.2f}bp, PnL ${t_p['pnl']:+,.0f}")
    L.append("")
    L.append("== 对照 ==")
    L.append("  TSLA 单标的存档（E8-A+S2, 10 个月）: +12.04%（年化 ~+15.2%），"
             "MTM MDD -3.68%")
    L.append(f"  E15 组合（分位 gate，冻结几何）: N4 {E15_REF[4]['total']:+.2%} / "
             f"N8 {E15_REF[8]['total']:+.2%}；E14（绝对阈值）: "
             f"{E14_TOTALS[4]:+.2%} / {E14_TOTALS[8]:+.2%}")
    L.append(f"  池 26 等权 B&H 同段: {bh_pool_total:+.2%}，日线 MDD {bh_pool_mdd:+.2%}")
    L.append(f"  TSLA B&H 同段: {bh_tsla_total:+.2%}，日线 MDD {bh_tsla_mdd:+.2%}")
    L.append("")
    L.append("== 逐标的贡献（N=4 档，PnL 降序，含 E15 对照）==")
    ps = per_sym[per_sym["included"] == True].sort_values("n4_pnl", ascending=False)  # noqa: E712
    L.append(f"{'symbol':<7s}{'几何':<14s}{'训练bp':>8s}{'留出bp':>8s}{'N4笔':>6s}"
             f"{'N4WR':>8s}{'N4bp':>9s}{'N4PnL$':>10s}{'E15N4$':>10s}{'ΔN4$':>10s}"
             f"{'N8PnL$':>10s}")
    for sym, r in ps.iterrows():
        L.append(f"{sym:<8s}{geom_of[sym]:<14s}{r['train_bp_best']:>8.1f}"
                 f"{r['hold_bp_best_geom']:>8.1f}{r['n4_trades']:>6d}"
                 f"{(r['n4_wr'] if np.isfinite(r['n4_wr']) else 0):>8.1%}"
                 f"{(r['n4_avg_bp'] if np.isfinite(r['n4_avg_bp']) else 0):>9.1f}"
                 f"{r['n4_pnl']:>10,.0f}{r['e15_n4_pnl']:>10,.0f}"
                 f"{r['d_n4_pnl']:>10,.0f}{r['n8_pnl']:>10,.0f}")
    L.append("")
    L.append("== 选择有效性（训练段期望排序 vs 留出段实际，Spearman）==")
    L.append(f"  全 26 标的（各自最优几何）: rho={rho_all:+.3f} (p={p_all:.3f}, "
             f"n={len(v_all)})")
    L.append(f"  仅入选 {len(v_inc)} 标的: rho={rho_inc:+.3f} (p={p_inc:.3f})")
    L.append("  判读：rho 显著为正 = 训练段期望排序在留出段保序（选择有依据）；"
             "接近 0 或为负 = 训练段选择在留出段无预测力（选择≈噪声拟合）。")
    L.append("")

    # ---------- 判读 ----------------------------------------------------------
    pos_hold = int((v_inc["hold_bp_best_geom"] > 0).sum())
    best4 = max(runs.values(), key=lambda r: r["ann"])
    L.append("== 判读 ==")
    L.append(f"1. 剔除机制：训练段五几何期望全非正剔除 {len(excluded)}/26"
             f"（{('、'.join(excluded)) if excluded else '无'}）——"
             + ("『这只股票我做不了』的表达能力被使用。" if excluded else
                "所有标的训练段都能找到正期望几何（注意 in-sample 乐观偏差）。"))
    L.append(f"2. 组合结果：N4 {r4['total']:+.2%}（年化 {r4['ann']:+.2%}）/ "
             f"N8 {r8['total']:+.2%}（年化 {r8['ann']:+.2%}），MTM MDD "
             f"{r4['mtm_mdd']:+.2%} / {r8['mtm_mdd']:+.2%}；"
             f"vs E15 修复 {r4['total'] - E15_REF[4]['total']:+.1%} / "
             f"{r8['total'] - E15_REF[8]['total']:+.1%}。"
             + ("两档均转正。" if min(r4["total"], r8["total"]) > 0 else
                ("一档转正一档未转。" if max(r4["total"], r8["total"]) > 0 else
                 "两档仍为负——逐标的几何重标定也救不回组合。")))
    L.append(f"3. 对池 B&H：B&H 同段 {bh_pool_total:+.2%}（MDD {bh_pool_mdd:+.2%}），"
             f"组合 N4 {'超额' if r4['total'] > bh_pool_total else '落后'} "
             f"{abs(r4['total'] - bh_pool_total):.2%}、N8 "
             f"{'超额' if r8['total'] > bh_pool_total else '落后'} "
             f"{abs(r8['total'] - bh_pool_total):.2%}。")
    L.append(f"4. 入选标的留出段验证：{pos_hold}/{len(v_inc)} 个入选标的留出段"
             f"（选中几何、独立结算、S2 后）期望为正；秩相关 rho={rho_inc:+.3f}"
             f"（p={p_inc:.3f}）——"
             + ("训练段选择在留出段有保序证据。" if (np.isfinite(p_inc) and p_inc < 0.05
                                                    and rho_inc > 0) else
                "训练段期望排序在留出段不保序，几何选择的样本外有效性未获支持。"))
    L.append(f"5. E17 双对照：最好档年化 {best4['ann']:+.2%} —— "
             f"30% 北极星{'达标' if best4['ann'] >= ANN_TARGET else '未达'}；"
             f"8% 达标线{'过线' if best4['ann'] >= ANN_PASS else '未过'}"
             f"（已验证最优仍为 TSLA 单标的 E8-A+S2 年化 ~+15.2%）。")
    L.append("6. 诚实边界：几何选择虽受限五选一、只用训练段，但仍是样本内选择——"
             "26 标的 × 5 几何 = 130 次比较如实记账；池标的训练段 prob 为 in-sample"
             "（模型见过这些事件），训练段期望因此系统性偏乐观，剔除决策与几何排序都"
             "受此污染；训练段期望用独立结算（候选窗口可重叠，样本相关导致有效 n 偏小）；"
             "留出段为单次检验，方向性证据级别，不升格达标候选；E15 分位 gate 与冻结"
             "模型/成本/timeout 的出身折扣全部继承。B&H 对照不计成本（对基准从宽）。"
             "多重比较 +132（130 几何选择 + N∈{4,8} 两档），计数器累计 ~3525。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
