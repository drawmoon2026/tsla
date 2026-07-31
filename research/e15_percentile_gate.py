#!/usr/bin/env python
"""E15 — 池 gate 按标的分位重标定（docs/strategy-lab.md E15，预登记协议）.

E14 病因（gate 绝对阈值 0.4325 跨标的失效：高波动标的 prob 分布右移、保留率爆炸）
的对症修正——单一预登记假设，非网格：

1. 每标的 gate 阈值 = 该标的 **训练段（et_day < 2025-09-30）** 事件概率分布的
   top10% 保留分位（np.quantile(probs, 0.90)，与冻结设计"保留率匹配"同规则、
   防前视）；TSLA 沿用冻结值 0.43250235336744486 不动。
   训练段事件 < 50 个的标的如实标注，用池 25 训练段合并分布兜底（记录名单）。
2. 其余协议与 E14 逐项一致：窗口 2025-10-01 → 数据尾；$100k 账本等额分片；
   N ∈ {4, 8} 两档预登记；同时刻信号 prob 降序取额度；冻结几何
   tp0.5%/sl2%/to48bar/日内强平悲观结算；每标的独立 S2（shift(1)）。
   组合模拟机制全部引用 research/e14_pool_portfolio.py（引用不复制）。
3. 冻结自校验与 E14 相同：TSLA 逐事件 prob 对 holdout_ref、单标的 62 笔 /
   S2 后 54 笔 +12.04% 存档复现（TSLA gate 未动，锚必须原样通过）。
4. 诚实边界：top10% 分位继承自冻结设计（E9 gate0.70 保留率匹配出身），非本次
   新选择；池标的阈值取自训练段 in-sample prob（模型见过这些事件）；"分位规则
   跨标的"整体仍是留出段单次检验——判读按方向性证据级别，不升格达标候选。
   多重比较 +2（N 两档）。

用法:  .venv/bin/python research/e15_percentile_gate.py
输出:  outputs/e15_percentile_gate/{trades.csv, equity.csv, per_symbol.csv, summary.txt}
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

import research.e8_pooled_gbdt as e8  # noqa: E402
import research.e14_pool_portfolio as e14  # noqa: E402  组合引擎（引用不复制）
from research.e11_bear_switch import daily_closes, switch_states  # noqa: E402
from research.e9_frontier_search import BarSet  # noqa: E402
from research.ml_common import build_dataset, load_events  # noqa: E402
from src.common.data_io import ET, load_bars  # noqa: E402

OUT = ROOT / "outputs" / "e15_percentile_gate"

KEEP_FRAC = 0.10          # 保留率目标：继承冻结设计 top~10% 规则，非新选择
MIN_TRAIN_EVENTS = 50     # 训练段事件低于此数走池合并分布兜底
E14_PER_SYM = ROOT / "outputs" / "e14_pool_portfolio" / "per_symbol.csv"

# E14 存档锚点（outputs/e14_pool_portfolio/summary.txt，2026-08-01）
E14_REF = {
    4: {"total": -0.2170, "ann": -0.2612, "mtm_mdd": -0.2535, "n": 2530,
        "tsla_n": 24},
    8: {"total": -0.1890, "ann": -0.2283, "mtm_mdd": -0.2059, "n": 3464,
        "tsla_n": 46},
}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 每标的冻结管线：事件 -> 特征（与 E14 同源）--------------------
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

    # ---------- 冻结模型推断：训练段（定阈值用）+ 评估段（交易用）------------
    model = joblib.load(ROOT / "models" / "e8a" / "model.joblib")
    all_ds["prob"] = model.predict_proba(all_ds[e8.POOLED_FEATURES])[:, 1]
    train = all_ds[all_ds["et_day"] < e8.TRAIN_END]
    ev = all_ds[all_ds["et_day"] >= e14.EVAL_START].reset_index(drop=True)

    # 逐事件 prob 对照 holdout_ref.csv（冻结自校验，与 E14 相同）
    ref = pd.read_csv(ROOT / "models" / "e8a" / "holdout_ref.csv")
    ref["entry_t"] = pd.to_datetime(ref["entry_t"], utc=True)
    tsla_ev = ev[ev["symbol"] == "TSLA"]
    assert len(tsla_ev) == len(ref), "TSLA 留出事件数与 holdout_ref 不一致"
    a = ref.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    b = tsla_ev.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    assert np.abs(a - b).max() < 1e-9, "冻结模型 prob 与 holdout_ref 不一致"
    print(f"[check] TSLA {len(tsla_ev)} 事件 prob 与 holdout_ref 完全一致")

    # ---------- E15 核心：每标的训练段 top10% 保留分位阈值 --------------------
    pool_train_probs = train.loc[train["symbol"] != "TSLA", "prob"].to_numpy()
    fallback_thr = float(np.quantile(pool_train_probs, 1.0 - KEEP_FRAC))
    thr_map, fallback_syms = {}, []
    thr_rows = []
    for sym in e14.SYMS:
        p_tr = train.loc[train["symbol"] == sym, "prob"].to_numpy()
        if sym == "TSLA":
            thr, src = e14.THR, "冻结值"
        elif len(p_tr) < MIN_TRAIN_EVENTS:
            thr, src = fallback_thr, "池兜底"
            fallback_syms.append(sym)
        else:
            thr, src = float(np.quantile(p_tr, 1.0 - KEEP_FRAC)), "自身训练段"
        thr_map[sym] = thr
        thr_rows.append({
            "symbol": sym, "thr": thr, "thr_source": src,
            "n_train_events": len(p_tr),
            "train_retention": float((p_tr >= thr).mean()) if len(p_tr) else np.nan,
        })
    thr_df = pd.DataFrame(thr_rows).set_index("symbol")
    print(f"[thr] 池兜底标的: {fallback_syms if fallback_syms else '无'}  "
          f"(池合并分位 {fallback_thr:.4f})")

    # ---------- gate（每标的分位阈值）+ epos + S2（结构同 E14）----------------
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
        s2 = s2_states[sym]
        s2_eval = s2[np.asarray([d >= e14.EVAL_START for d in s2.index])]
        per_sym_rows.append({
            "symbol": sym,
            "scope": "时间外+标的外" if sym == "TSLA" else "时间外·标的内",
            "n_events_eval": len(sub), "n_gate_pass": len(gate),
            "retention": len(gate) / len(sub) if len(sub) else np.nan,
            "n_cand": len(cand), "n_s2_blocked": int(cand["s2_off"].sum()),
            "s2_off_day_share": float(s2_eval.mean()) if len(s2_eval) else np.nan,
        })
    cands_all = pd.concat(cand_frames, ignore_index=True)
    per_sym = pd.DataFrame(per_sym_rows).set_index("symbol").join(thr_df)

    # ---------- 逐候选冻结结算明细（e14.detail_trade，冻结几何）---------------
    details = {}
    for r in cands_all.itertuples():
        d = e14.detail_trade(bsets[r.sym], int(r.epos))
        assert d is not None
        details[(r.sym, int(r.epos))] = d

    # ---------- TSLA 单标的存档复现（TSLA gate 未动，锚必须原样通过）----------
    tc = cands_all[cands_all["sym"] == "TSLA"]
    rep = e14.seq_replay(tc, details)
    wr, avg = float((rep["ret"] > 0).mean()), float(rep["ret"].mean() * 1e4)
    assert len(rep) == e14.EXP_TSLA["n"] and abs(wr - e14.EXP_TSLA["wr"]) < 1e-9 \
        and abs(avg - e14.EXP_TSLA["avg_bp"]) < 0.05, "TSLA 62 笔存档复现失败"
    keep = rep[~rep["et_day"].map(s2_states["TSLA"]).fillna(False).astype(bool)]
    tot_s2 = float(np.prod(1 + keep["ret"]) - 1)
    assert len(keep) == e14.EXP_TSLA["n_s2"] \
        and abs(tot_s2 - e14.EXP_TSLA["total_s2"]) < 1e-6, "TSLA S2 后 54 笔复现失败"
    print(f"[check] TSLA 单标的复现: 62 笔 WR {wr:.2%} avg {avg:+.2f}bp；"
          f"S2 后 54 笔 total {tot_s2:+.2%} —— 与存档一致")

    # ---------- 组合模拟（e14.simulate / e14.mtm_paths，两档 N）---------------
    live = cands_all[~cands_all["s2_off"]].reset_index(drop=True)
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

    # ---------- 对照：池等权 B&H / TSLA B&H（同段，不计成本，同 E14）----------
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

    # ---------- 逐标的贡献表（含 E14 对照列）---------------------------------
    for ncap in e14.N_CAPS:
        tr = runs[ncap]["trades"]
        g = tr.groupby("symbol")
        per_sym[f"n{ncap}_trades"] = g.size().reindex(per_sym.index).fillna(0).astype(int)
        per_sym[f"n{ncap}_wr"] = g.apply(lambda x: float((x["ret"] > 0).mean()),
                                         include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_avg_bp"] = g.apply(lambda x: float(x["ret"].mean() * 1e4),
                                             include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_pnl"] = g["pnl"].sum().reindex(per_sym.index).fillna(0.0)
    e14_ps = pd.read_csv(E14_PER_SYM).set_index("symbol")
    per_sym["e14_retention"] = e14_ps["retention"]
    per_sym["e14_n4_pnl"] = e14_ps["n4_pnl"]
    per_sym["e14_n8_pnl"] = e14_ps["n8_pnl"]
    per_sym["d_n4_pnl"] = per_sym["n4_pnl"] - per_sym["e14_n4_pnl"]

    # ---------- 输出 ----------------------------------------------------------
    trades_all = pd.concat([runs[n]["trades"] for n in e14.N_CAPS], ignore_index=True)
    trades_all["entry_et"] = pd.DatetimeIndex(trades_all["entry_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    trades_all["exit_et"] = pd.DatetimeIndex(trades_all["exit_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    cols = ["n_cap", "symbol", "entry_et", "exit_et", "et_day", "prob", "alloc",
            "entry_px", "exit_type", "ret", "pnl", "eq_at_entry", "epos", "exit_pos"]
    trades_all[cols].to_csv(OUT / "trades.csv", index=False, float_format="%.6f")
    eq_daily.round(2).to_csv(OUT / "equity.csv")
    per_sym.reset_index().to_csv(OUT / "per_symbol.csv", index=False,
                                 float_format="%.6f")

    # ---------- summary（中文判读）-------------------------------------------
    d0, d1 = str(tl_days[0]), str(tl_days[-1])
    n_sig = len(live)
    tsla_pm = e14.EXP_TSLA["n_s2"] / months
    r4, r8 = runs[4], runs[8]

    L = []
    L.append("E15 — 池 gate 按标的分位重标定（冻结 E8-A+S2，26 标的，预登记协议）")
    L.append(f"窗口 {d0} → {d1}（{cal_days} 天 / {len(eq_daily)} 交易日 / {months:.1f} 个月）；"
             f"账本 ${e14.CAPITAL:,.0f}，等额分片，N∈{{4,8}} 两档预登记 —— 协议与 E14 逐项一致，"
             "唯一改动 = gate 逻辑")
    L.append(f"gate 规则：每标的阈值 = 该标的训练段（et_day < {e8.TRAIN_END}）事件 prob 的 "
             f"top{KEEP_FRAC:.0%} 保留分位（quantile {1 - KEEP_FRAC:.2f}，与冻结设计"
             "『保留率匹配』同规则、防前视）；TSLA 沿用冻结值 "
             f"{e14.THR:.6f} 不动；训练段事件 <{MIN_TRAIN_EVENTS} 走池 25 合并分布兜底"
             f"（本次兜底标的: {('、'.join(fallback_syms)) if fallback_syms else '无'}）")
    L.append("冻结件其余不动：几何 tp0.5%/sl2.0%/to48bar/日内强平，悲观结算；"
             "每标的独立 S2（252 日高回撤>20%，shift(1)）")
    L.append("冻结自校验：TSLA 逐事件 prob 与 holdout_ref 一致；单标的重放 62 笔 / S2 后 "
             "54 笔 +12.04% 与存档一致 —— 管线仅 gate 改动确认")
    L.append("")
    L.append("== 逐标的新阈值与保留率（判读锚 a）==")
    L.append(f"{'symbol':<7s}{'阈值':>9s}{'来源':>7s}{'训练n':>7s}{'训练保留':>9s}"
             f"{'留出保留':>9s}{'E14留出':>9s}")
    for sym, r in per_sym.iterrows():
        L.append(f"{sym:<8s}{r['thr']:>9.4f}{r['thr_source']:>6s}"
                 f"{r['n_train_events']:>7d}{r['train_retention']:>9.1%}"
                 f"{r['retention']:>9.1%}{r['e14_retention']:>9.1%}")
    pool_ret = per_sym.drop(index="TSLA")["retention"]
    L.append(f"池 25 留出段保留率：中位 {pool_ret.median():.1%}，范围 "
             f"[{pool_ret.min():.1%}, {pool_ret.max():.1%}]"
             f"（E14 为中位 {per_sym.drop(index='TSLA')['e14_retention'].median():.1%}、"
             f"最高 {per_sym.drop(index='TSLA')['e14_retention'].max():.1%}）；"
             "训练段按构造 = 10%，留出段偏离 10% 的部分即该标的 prob 分布的时移。")
    L.append("")
    L.append("== 组合指标（两档并发，vs E14 存档）==")
    L.append(f"{'指标':<22s}" + "".join(f"N={n:<12d}" for n in e14.N_CAPS)
             + "".join(f"E14·N={n:<8d}" for n in e14.N_CAPS))

    def row(name, fn, e14fn):
        L.append(f"{name:<24s}" + "".join(f"{fn(runs[n]):<13s}" for n in e14.N_CAPS)
                 + "".join(f"{e14fn(E14_REF[n]):<13s}" for n in e14.N_CAPS))
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
    L.append(f"S2 后组合候选信号 {n_sig} 个（S2 拦截 {int(cands_all['s2_off'].sum())} 个），"
             f"即 {n_sig / months:.1f} 信号/月（E14 为 530.6/月——分位 gate 把信号流"
             f"压回 {n_sig / months / 530.6:.0%}）；TSLA 单标的存档 {tsla_pm:.1f} 笔/月")
    L.append("")
    L.append("== 两种口径分开报告（组合内贡献拆分）==")
    for n in e14.N_CAPS:
        tr = runs[n]["trades"]
        t_t = e14.blk_stats(tr[tr["symbol"] == "TSLA"])
        t_p = e14.blk_stats(tr[tr["symbol"] != "TSLA"])
        L.append(f"  N={n}  TSLA（时间外+标的外）: {t_t['n']} 笔, WR {t_t['wr']:.1%}, "
                 f"{t_t['avg_bp']:+.2f}bp, PnL ${t_t['pnl']:+,.0f}"
                 f"（E14 同档 {E14_REF[n]['tsla_n']} 笔；单标的独跑 54 笔）")
        L.append(f"        池 25（时间外·标的内）: {t_p['n']} 笔, WR {t_p['wr']:.1%}, "
                 f"{t_p['avg_bp']:+.2f}bp, PnL ${t_p['pnl']:+,.0f}")
    L.append("  提示：池标的口径弱于 TSLA 口径——标的在训练分布内，只有时间是外推的。")
    L.append("")
    L.append("== 对照 ==")
    L.append("  TSLA 单标的存档（E8-A+S2, 10 个月）: +12.04%（年化 ~+15.2%），"
             "MTM MDD -3.68%，5.4 笔/月")
    L.append(f"  E14 组合（绝对阈值 gate）: N4 {E14_REF[4]['total']:+.2%} / "
             f"N8 {E14_REF[8]['total']:+.2%}，MTM MDD {E14_REF[4]['mtm_mdd']:+.2%} / "
             f"{E14_REF[8]['mtm_mdd']:+.2%}")
    L.append(f"  池 26 等权 B&H 同段: {bh_pool_total:+.2%}，日线 MDD {bh_pool_mdd:+.2%}")
    L.append(f"  TSLA B&H 同段: {bh_tsla_total:+.2%}，日线 MDD {bh_tsla_mdd:+.2%}")
    L.append("")
    L.append("== 逐标的贡献（N=4 档，PnL 降序，含 E14 对照；判读锚 d）==")
    ps = per_sym.sort_values("n4_pnl", ascending=False)
    L.append(f"{'symbol':<7s}{'口径':<10s}{'留出保留':>9s}{'N4笔':>6s}{'N4WR':>8s}"
             f"{'N4bp':>9s}{'N4PnL$':>10s}{'E14N4$':>10s}{'ΔN4$':>10s}"
             f"{'N8PnL$':>10s}{'E14N8$':>10s}")
    for sym, r in ps.iterrows():
        L.append(f"{sym:<8s}{r['scope']:<8s}{r['retention']:>9.1%}"
                 f"{r['n4_trades']:>6d}"
                 f"{(r['n4_wr'] if np.isfinite(r['n4_wr']) else 0):>8.1%}"
                 f"{(r['n4_avg_bp'] if np.isfinite(r['n4_avg_bp']) else 0):>9.1f}"
                 f"{r['n4_pnl']:>10,.0f}{r['e14_n4_pnl']:>10,.0f}"
                 f"{r['d_n4_pnl']:>10,.0f}{r['n8_pnl']:>10,.0f}"
                 f"{r['e14_n8_pnl']:>10,.0f}")
    L.append("")

    # ---------- 判读（预登记锚逐条）-------------------------------------------
    hi_vol = ["COIN", "INTC", "NVDA", "MU", "AMD", "PLTR", "AVGO", "QCOM"]
    hv = per_sym.loc[[s for s in hi_vol if s in per_sym.index]]
    n_fixed = int((hv["n4_pnl"] > 0).sum())
    n_bleed_less = int((hv["d_n4_pnl"] > 0).sum())
    tsla4 = int(per_sym.loc["TSLA", "n4_trades"])
    tsla8 = int(per_sym.loc["TSLA", "n8_trades"])
    pool4 = e14.blk_stats(r4["trades"][r4["trades"]["symbol"] != "TSLA"])
    L.append("== 判读（预登记锚逐条）==")
    L.append(f"a. 保留率是否回到 ~10%：训练段按构造全部 10.0%；留出段中位 "
             f"{pool_ret.median():.1%}、范围 [{pool_ret.min():.1%}, {pool_ret.max():.1%}]"
             f"（E14 最高曾到 {per_sym.drop(index='TSLA')['e14_retention'].max():.0%}）。"
             + ("gate 的 top-decile 筛选功能已恢复。" if pool_ret.max() < 0.30 else
                f"多数标的回到 ~10%，但 {'、'.join(pool_ret[pool_ret >= 0.30].index)} "
                "留出段仍 >=30%——其 prob 分布在留出段整体右移，分位阈值只能修跨标的差异、"
                "修不了时移。"))
    L.append(f"b. 组合是否转正：N4 {r4['total']:+.2%} / N8 {r8['total']:+.2%}"
             f"（E14 {E14_REF[4]['total']:+.2%} / {E14_REF[8]['total']:+.2%}，修复 "
             f"{r4['total'] - E14_REF[4]['total']:+.1%} / "
             f"{r8['total'] - E14_REF[8]['total']:+.1%}）；MTM MDD "
             f"{r4['mtm_mdd']:+.2%} / {r8['mtm_mdd']:+.2%}"
             f"（E14 {E14_REF[4]['mtm_mdd']:+.2%} / {E14_REF[8]['mtm_mdd']:+.2%}）。"
             + ("两档均转正。" if min(r4["total"], r8["total"]) > 0 else
                ("一档转正一档未转。" if max(r4["total"], r8["total"]) > 0 else
                 "两档仍为负——病因修正方向正确但不足以救回组合。")))
    L.append(f"c. 对池 B&H：B&H 同段 {bh_pool_total:+.2%}（MDD {bh_pool_mdd:+.2%}），"
             f"组合 N4 {'超额' if r4['total'] > bh_pool_total else '落后'} "
             f"{abs(r4['total'] - bh_pool_total):.2%}、N8 "
             f"{'超额' if r8['total'] > bh_pool_total else '落后'} "
             f"{abs(r8['total'] - bh_pool_total):.2%}"
             + ("——未接近 B&H。" if max(r4["total"], r8["total"]) < bh_pool_total - 0.10
                else "。"))
    L.append(f"d. TSLA 排挤：组合内 TSLA 成交 N4 {tsla4} 笔 / N8 {tsla8} 笔"
             f"（E14 为 {E14_REF[4]['tsla_n']} / {E14_REF[8]['tsla_n']}；单标的独跑 54 笔）。"
             + ("排挤明显缓解。" if tsla4 >= 2 * E14_REF[4]["tsla_n"] else
                ("有缓解但仍被挤占。" if tsla4 > E14_REF[4]["tsla_n"] else "排挤未缓解。")))
    L.append(f"e. 高波动标的止血（COIN/INTC/NVDA 等 8 只）：N4 档 {n_fixed}/8 转正、"
             f"{n_bleed_less}/8 较 E14 止血；三只点名标的 "
             + "、".join(f"{s} ${per_sym.loc[s, 'n4_pnl']:+,.0f}"
                          f"（E14 ${per_sym.loc[s, 'e14_n4_pnl']:+,.0f}）"
                          for s in ["COIN", "INTC", "NVDA"]) + "。")
    L.append(f"f. 两口径：TSLA（时间外+标的外）N8 档 "
             f"{e14.blk_stats(r8['trades'][r8['trades']['symbol'] == 'TSLA'])['avg_bp']:+.1f}bp；"
             f"池 25（时间外·标的内）N4 合计 WR {pool4['wr']:.1%}、{pool4['avg_bp']:+.2f}bp"
             + ("——期望转正。" if pool4["avg_bp"] > 0 else
                "——期望仍为负（几何幻觉：WR 高但 tp0.5%<<sl2%）。")
             + "池口径证据天然弱于 TSLA 口径，不可合并宣称。")
    L.append("g. 诚实边界：top10% 保留分位继承自冻结设计（E9 gate0.70 保留率匹配出身），"
             "非本次新选择；池标的阈值取自训练段 in-sample prob 分布（模型见过这些事件，"
             "分位数因此可能偏乐观地偏移）；『分位规则跨标的』整体仍是留出段单次检验，"
             "结果按方向性证据级别判读，不升格达标候选；gate/几何的 TSLA 事后网格出身"
             "与 E14 相同，多重比较折扣继承。B&H 对照不计成本（对基准从宽）。"
             "多重比较 +2（N∈{4,8} 两档）。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
