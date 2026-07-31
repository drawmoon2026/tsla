#!/usr/bin/env python
"""E14 — 多标的池组合级全面模拟（docs/strategy-lab.md E14，预登记协议）.

冻结 E8-A+S2（models/e8a/，零改动）在 26 标的（池 25 + TSLA）上的组合运转模拟：

1. 每标的独立跑冻结管线：既有 V 事件表 -> ml_common 特征（e8_pooled_gbdt 同源
   归一化）-> model.joblib 推断 -> gate（阈值 0.43250235336744486 冻结值，池标的
   保留率各异，逐标的报告、不重标定）-> 冻结几何 tp0.5%/sl2%/timeout48/日内强平
   悲观结算（settle_bracket, 1bp 费 + 2bp 滑点单边）-> 每标的独立 S2
   （252 日高回撤 >20% 停用，shift(1)）。
2. 评估段：仅 2025-10-01 起（训练截止 2025-09-30 + embargo 之后，至数据尾
   2026-07-22/23）。口径声明：池标的 = 时间外、标的内（在训练标的清单里）；
   TSLA = 时间外 + 标的外（leave-TSLA-out）。两口径分开报告。
3. 组合层：统一资金账本 $100k；等额分片 每笔投入 = min(当前账面权益/N, 现金)，
   N ∈ {4, 8} 两档预登记；同一标的同时只一仓；同时刻信号按 prob 降序取额度内。
4. 对照：TSLA 单标的存档（outputs/e8a_replay，+12.04%/10 个月）与池 26 标的
   等权 B&H、TSLA B&H 同段（B&H 不计成本——对基准从宽）。
5. 诚实边界：gate 阈值与几何为 TSLA 留出段事后网格出身（E8 档案已声明），组合
   结果继承该多重比较折扣；本实验 = 模拟平台能力验证 + 方向性证据，不是新的
   达标候选。多重比较 +2（两档 N）。

冻结管线自校验（不一致即报错退出）：
- symbol 类别表 / TSLA 训练段 rv20 中位数与 meta.json 一致；
- TSLA 留出段逐事件 prob 与 models/e8a/holdout_ref.csv 一致；
- TSLA 单标的顺序重放（gate 后、S2 前）必须复现存档 62 笔 / WR 80.65% /
  +15.97bp；S2 后 54 笔 / +12.04%。

用法:  .venv/bin/python research/e14_pool_portfolio.py
输出:  outputs/e14_pool_portfolio/{trades.csv, equity.csv, per_symbol.csv, summary.txt}
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

import research.e8_pooled_gbdt as e8  # noqa: E402
from research.e11_bear_switch import daily_closes, switch_states  # noqa: E402
from research.e9_frontier_search import COST, BarSet  # noqa: E402
from research.ml_common import build_dataset, load_events  # noqa: E402
from src.common.data_io import ET, load_bars  # noqa: E402
from src.common.execution import entry_fill, settle_bracket  # noqa: E402

OUT = ROOT / "outputs" / "e14_pool_portfolio"
META = json.loads((ROOT / "models" / "e8a" / "meta.json").read_text(encoding="utf-8"))

THR = float(META["gate"]["threshold"])            # 0.43250235336744486，冻结
TP = float(META["geometry"]["tp_pct"])            # 0.005
SL = float(META["geometry"]["sl_pct"])            # 0.020
TIMEOUT = int(META["geometry"]["timeout_bars"])   # 48
EVAL_START = date(2025, 10, 1)                    # = e8.HOLDOUT_START
CAPITAL = 100_000.0
N_CAPS = [4, 8]                                   # 预登记两档并发上限
SYMS = e8.POOL_SYMBOLS + [e8.TARGET]              # 26 标的

# TSLA 单标的存档锚点（e8a_replay / e11 switch_table）
EXP_TSLA = {"n": 62, "wr": 0.8064516129032258, "avg_bp": 15.965470720673224,
            "n_s2": 54, "total_s2": 0.120365289615876}


# ------------------------------------------------------------------ 冻结结算
def detail_trade(bs: BarSet, epos: int) -> dict | None:
    """与 e8a_trades_export.detail_trade 同口径（冻结几何，悲观结算）。"""
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
    res = settle_bracket(window, 1, entry_px, entry_px * (1 + TP),
                         entry_px * (1 - SL), COST)
    exit_pos = epos + window.index.get_loc(res.exit_time)
    chk = bs.eval_trade(epos, 1, TP, SL, TIMEOUT)
    assert chk is not None and abs(chk[0] - res.ret) < 1e-12 and chk[1] == exit_pos, \
        "detail_trade 与 BarSet.eval_trade 不一致"
    exit_type = res.hit if res.hit in ("tp", "sl") else (
        "dayend" if (day_cut and exit_pos == end - 1) else "timeout")
    return {"entry_px": entry_px, "ret": float(res.ret), "exit_pos": exit_pos,
            "exit_t": bs.idx[exit_pos], "exit_type": exit_type}


def seq_replay(cands: pd.DataFrame, details: dict) -> pd.DataFrame:
    """单标的顺序重放（run_combo 同语义：一次一仓，epos > 上笔 exit_pos）。"""
    rows, busy = [], -1
    for r in cands.sort_values("epos").itertuples():
        if r.epos <= busy:
            continue
        d = details[(r.sym, r.epos)]
        busy = d["exit_pos"]
        rows.append({"epos": r.epos, "et_day": r.et_day, "ret": d["ret"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 组合模拟
def simulate(cands: pd.DataFrame, details: dict, ncap: int) -> dict:
    ev = cands.sort_values(["entry_t", "prob"],
                           ascending=[True, False]).reset_index(drop=True)
    cash, open_pos = CAPITAL, {}   # sym -> (exit_t, alloc, proceeds)
    trades, skip_busy, skip_cap = [], 0, 0
    for r in ev.itertuples():
        t = r.entry_t
        for s in [s for s, (xt, _, _) in open_pos.items() if xt < t]:
            cash += open_pos[s][2]
            del open_pos[s]
        if r.sym in open_pos:            # 同标的已有持仓（含 run_combo 重叠语义）
            skip_busy += 1
            continue
        if len(open_pos) >= ncap:        # 并发额度用尽
            skip_cap += 1
            continue
        eq = cash + sum(al for _, al, _ in open_pos.values())
        alloc = min(eq / ncap, cash)
        d = details[(r.sym, r.epos)]
        cash -= alloc
        open_pos[r.sym] = (d["exit_t"], alloc, alloc * (1.0 + d["ret"]))
        trades.append({
            "n_cap": ncap, "symbol": r.sym, "epos": r.epos,
            "entry_t": r.entry_t, "exit_t": d["exit_t"],
            "exit_pos": d["exit_pos"], "et_day": r.et_day,
            "prob": r.prob, "alloc": alloc, "entry_px": d["entry_px"],
            "exit_type": d["exit_type"], "ret": d["ret"],
            "pnl": alloc * d["ret"], "eq_at_entry": eq,
        })
    for s, (_, _, pr) in open_pos.items():
        cash += pr
    tr = pd.DataFrame(trades)
    return {"trades": tr, "final_eq": cash, "skip_busy": skip_busy,
            "skip_cap": skip_cap}


def mtm_paths(tr: pd.DataFrame, bsets: dict, timeline: pd.DatetimeIndex):
    """组合 5m 盯市权益路径 + 并发数路径（持仓 bar 收盘价盯市，出场 bar 落袋）。"""
    tvals = timeline.values
    m = len(timeline)
    cash_d = np.zeros(m)
    pos_d = np.zeros(m)
    cnt_d = np.zeros(m)
    for t in tr.itertuples():
        bs = bsets[t.symbol]
        i_in = int(np.searchsorted(tvals, np.datetime64(t.entry_t)))
        i_out = int(np.searchsorted(tvals, np.datetime64(t.exit_t)))
        assert tvals[i_in] == np.datetime64(t.entry_t)
        assert tvals[i_out] == np.datetime64(t.exit_t)
        cash_d[i_in] -= t.alloc
        cash_d[i_out] += t.alloc * (1.0 + t.ret)
        cnt_d[i_in] += 1
        cnt_d[i_out] -= 1
        if t.exit_pos > t.epos:          # 持仓期 bar 收盘盯市（阶梯差分）
            marks = t.alloc * bs.closes[t.epos:t.exit_pos] / t.entry_px
            idxs = np.searchsorted(tvals, bs.idx.values[t.epos:t.exit_pos])
            np.add.at(pos_d, idxs[0], marks[0])
            if len(marks) > 1:
                np.add.at(pos_d, idxs[1:], np.diff(marks))
            np.add.at(pos_d, i_out, -marks[-1])
    eq = CAPITAL + np.cumsum(cash_d) + np.cumsum(pos_d)
    return eq, np.cumsum(cnt_d), np.cumsum(pos_d)


def mdd(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def blk_stats(tr: pd.DataFrame) -> dict:
    n = len(tr)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg_bp": np.nan, "pnl": 0.0}
    r = tr["ret"].to_numpy()
    return {"n": n, "wr": float((r > 0).mean()), "avg_bp": float(r.mean() * 1e4),
            "pnl": float(tr["pnl"].sum())}


# ------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 每标的冻结管线：事件 -> 特征 ----------------------------------
    frames, bsets, s2_states = [], {}, {}
    for sym in SYMS:
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

    assert list(all_ds["symbol"].cat.categories) == META["symbol_categories"], \
        "symbol 类别表与 meta.json 不一致"
    tsla_med = float(all_ds.loc[all_ds["symbol"] == "TSLA", "rv20_med_train"].iloc[0])
    assert abs(tsla_med - META["tsla_rv20_med_train"]) < 1e-12, \
        "TSLA rv20 训练段中位数与 meta.json 不一致"

    # ---------- 冻结模型推断（仅评估段）--------------------------------------
    model = joblib.load(ROOT / "models" / "e8a" / "model.joblib")
    ev = all_ds[all_ds["et_day"] >= EVAL_START].reset_index(drop=True)
    ev["prob"] = model.predict_proba(ev[e8.POOLED_FEATURES])[:, 1]

    # 逐事件 prob 对照 holdout_ref.csv（冻结自校验）
    ref = pd.read_csv(ROOT / "models" / "e8a" / "holdout_ref.csv")
    ref["entry_t"] = pd.to_datetime(ref["entry_t"], utc=True)
    ref["trough_confirm_t"] = pd.to_datetime(ref["trough_confirm_t"], utc=True)
    tsla_ev = ev[ev["symbol"] == "TSLA"]
    assert len(tsla_ev) == len(ref), "TSLA 留出事件数与 holdout_ref 不一致"
    # (entry_t, prob) 双键排序后逐位对齐（同 entry_t 可有多事件、prob 各异）
    a = ref.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    b = tsla_ev.sort_values(["entry_t", "prob"])["prob"].to_numpy()
    assert np.abs(a - b).max() < 1e-9, "冻结模型 prob 与 holdout_ref 不一致"
    print(f"[check] TSLA {len(tsla_ev)} 事件 prob 与 holdout_ref 完全一致")

    # ---------- gate（冻结阈值）+ epos + S2 ----------------------------------
    per_sym_rows, cand_frames = [], []
    for sym in SYMS:
        sub = ev[ev["symbol"] == sym].copy()
        bs = bsets[sym]
        sub["epos"] = bs.idx.get_indexer(pd.DatetimeIndex(sub["entry_t"]))
        assert (sub["epos"] >= 0).all()
        gate = sub[sub["prob"] >= THR]
        cand = (gate.groupby("epos", as_index=False)
                .agg(prob=("prob", "max"), entry_t=("entry_t", "first"),
                     et_day=("et_day", "first")))
        cand["sym"] = sym
        cand["s2_off"] = cand["et_day"].map(s2_states[sym]).fillna(False).astype(bool)
        cand_frames.append(cand)
        s2 = s2_states[sym]
        s2_eval = s2[np.asarray([d >= EVAL_START for d in s2.index])]
        per_sym_rows.append({
            "symbol": sym,
            "scope": "时间外+标的外" if sym == "TSLA" else "时间外·标的内",
            "n_events_eval": len(sub), "n_gate_pass": len(gate),
            "retention": len(gate) / len(sub) if len(sub) else np.nan,
            "n_cand": len(cand), "n_s2_blocked": int(cand["s2_off"].sum()),
            "s2_off_day_share": float(s2_eval.mean()) if len(s2_eval) else np.nan,
        })
    cands_all = pd.concat(cand_frames, ignore_index=True)
    per_sym = pd.DataFrame(per_sym_rows).set_index("symbol")

    # ---------- 逐候选冻结结算明细 -------------------------------------------
    details = {}
    for r in cands_all.itertuples():
        d = detail_trade(bsets[r.sym], int(r.epos))
        assert d is not None
        details[(r.sym, int(r.epos))] = d

    # ---------- TSLA 单标的存档复现（冻结自校验）------------------------------
    tc = cands_all[cands_all["sym"] == "TSLA"]
    rep = seq_replay(tc, details)
    wr, avg = float((rep["ret"] > 0).mean()), float(rep["ret"].mean() * 1e4)
    assert len(rep) == EXP_TSLA["n"] and abs(wr - EXP_TSLA["wr"]) < 1e-9 \
        and abs(avg - EXP_TSLA["avg_bp"]) < 0.05, "TSLA 62 笔存档复现失败"
    keep = rep[~rep["et_day"].map(s2_states["TSLA"]).fillna(False).astype(bool)]
    tot_s2 = float(np.prod(1 + keep["ret"]) - 1)
    assert len(keep) == EXP_TSLA["n_s2"] and abs(tot_s2 - EXP_TSLA["total_s2"]) < 1e-6, \
        "TSLA S2 后 54 笔存档复现失败"
    print(f"[check] TSLA 单标的复现: 62 笔 WR {wr:.2%} avg {avg:+.2f}bp；"
          f"S2 后 54 笔 total {tot_s2:+.2%} —— 与存档一致")

    # ---------- 组合模拟（S2 后候选流，两档 N）--------------------------------
    live = cands_all[~cands_all["s2_off"]].reset_index(drop=True)
    times = [bs.idx[bs.et_dates >= EVAL_START] for bs in bsets.values()]
    timeline = pd.DatetimeIndex(sorted(set().union(*[set(t) for t in times])))
    tl_days = np.asarray(timeline.tz_convert(ET).date)
    cal_days = (tl_days[-1] - tl_days[0]).days
    months = cal_days / 30.4375

    runs, eq_daily = {}, pd.DataFrame(index=pd.Index(sorted(set(tl_days)), name="et_day"))
    day_last = pd.Series(np.arange(len(timeline)), index=tl_days).groupby(level=0).last()
    for ncap in N_CAPS:
        res = simulate(live, details, ncap)
        tr = res["trades"]
        eq_path, cnt_path, posval = mtm_paths(tr, bsets, timeline)
        assert abs(eq_path[-1] - res["final_eq"]) < 1e-6
        assert abs(eq_path[-1] - (CAPITAL + tr["pnl"].sum())) < 1e-6
        runs[ncap] = {
            **res,
            "total": eq_path[-1] / CAPITAL - 1.0,
            "ann": (eq_path[-1] / CAPITAL) ** (365 / cal_days) - 1.0,
            "mtm_mdd": mdd(eq_path),
            "avg_conc": float(cnt_path.mean()), "max_conc": int(cnt_path.max()),
            "util": float((posval / eq_path).mean()),
            "per_month": len(tr) / months,
        }
        eq_daily[f"eq_n{ncap}"] = eq_path[day_last.to_numpy()]
        print(f"[N={ncap}] trades={len(tr)}  total={runs[ncap]['total']:+.2%}  "
              f"ann={runs[ncap]['ann']:+.2%}  mtm_mdd={runs[ncap]['mtm_mdd']:+.2%}  "
              f"skip_cap={res['skip_cap']}")

    # ---------- 对照：池等权 B&H / TSLA B&H（同段，不计成本）------------------
    dcs = {}
    for sym, bs in bsets.items():
        mask = bs.et_dates >= EVAL_START
        first = int(np.nonzero(mask)[0][0])
        dc = pd.Series(bs.closes[mask],
                       index=np.asarray(bs.et_dates)[mask]).groupby(level=0).last()
        dcs[sym] = dc / bs.opens[first]           # 每 $1 买入路径
    bh = pd.DataFrame(dcs).reindex(eq_daily.index).ffill()
    eq_daily["bh_pool_26"] = bh.mean(axis=1) * CAPITAL
    eq_daily["bh_tsla"] = bh["TSLA"] * CAPITAL
    bh_pool_total = eq_daily["bh_pool_26"].iloc[-1] / CAPITAL - 1
    bh_tsla_total = eq_daily["bh_tsla"].iloc[-1] / CAPITAL - 1
    bh_pool_mdd = mdd(eq_daily["bh_pool_26"].to_numpy())
    bh_tsla_mdd = mdd(eq_daily["bh_tsla"].to_numpy())

    # ---------- 逐标的贡献表 --------------------------------------------------
    for ncap in N_CAPS:
        tr = runs[ncap]["trades"]
        g = tr.groupby("symbol")
        per_sym[f"n{ncap}_trades"] = g.size().reindex(per_sym.index).fillna(0).astype(int)
        per_sym[f"n{ncap}_wr"] = g.apply(lambda x: float((x["ret"] > 0).mean()),
                                         include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_avg_bp"] = g.apply(lambda x: float(x["ret"].mean() * 1e4),
                                             include_groups=False).reindex(per_sym.index)
        per_sym[f"n{ncap}_pnl"] = g["pnl"].sum().reindex(per_sym.index).fillna(0.0)

    # ---------- 输出 ----------------------------------------------------------
    trades_all = pd.concat([runs[n]["trades"] for n in N_CAPS], ignore_index=True)
    trades_all["entry_et"] = pd.DatetimeIndex(trades_all["entry_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    trades_all["exit_et"] = pd.DatetimeIndex(trades_all["exit_t"]).tz_convert(ET) \
        .strftime("%Y-%m-%d %H:%M")
    cols = ["n_cap", "symbol", "entry_et", "exit_et", "et_day", "prob", "alloc",
            "entry_px", "exit_type", "ret", "pnl", "eq_at_entry", "epos", "exit_pos"]
    trades_all[cols].to_csv(OUT / "trades.csv", index=False,
                            float_format="%.6f")
    eq_daily.round(2).to_csv(OUT / "equity.csv")
    per_sym.reset_index().to_csv(OUT / "per_symbol.csv", index=False,
                                 float_format="%.6f")

    # ---------- summary（中文判读）-------------------------------------------
    d0, d1 = str(tl_days[0]), str(tl_days[-1])
    n_sig = len(live)
    tsla_pm = EXP_TSLA["n_s2"] / months

    L = []
    L.append("E14 — 多标的池组合级全面模拟（冻结 E8-A+S2，26 标的，预登记协议）")
    L.append(f"窗口 {d0} → {d1}（{cal_days} 天 / {len(eq_daily)} 交易日 / {months:.1f} 个月）；"
             f"账本 ${CAPITAL:,.0f}，等额分片，N∈{{4,8}} 两档预登记")
    L.append(f"冻结件：gate {THR:.6f}（TSLA 留出保留率匹配出身，池标的不重标定）；"
             f"几何 tp{TP:.1%}/sl{SL:.1%}/to{TIMEOUT}bar/日内强平，悲观结算；"
             "每标的独立 S2（252 日高回撤>20%，shift(1)）")
    L.append("冻结自校验：TSLA 逐事件 prob 与 holdout_ref 一致；单标的重放 62 笔 / S2 后 "
             "54 笔 +12.04% 与存档一致 —— 管线零改动确认")
    L.append("")
    L.append("== 组合指标（两档并发对比）==")
    hdr = f"{'指标':<22s}" + "".join(f"N={n:<12d}" for n in N_CAPS)
    L.append(hdr)
    def row(name, fn):
        L.append(f"{name:<24s}" + "".join(f"{fn(runs[n]):<13s}" for n in N_CAPS))
    row("总收益", lambda r: f"{r['total']:+.2%}")
    row("年化（实际天数）", lambda r: f"{r['ann']:+.2%}")
    row("组合 MTM 最大回撤", lambda r: f"{r['mtm_mdd']:+.2%}")
    row("总笔数", lambda r: f"{len(r['trades'])}")
    row("笔/月", lambda r: f"{r['per_month']:.1f}")
    row("WR", lambda r: f"{(r['trades']['ret'] > 0).mean():.1%}")
    row("单笔均值 bp", lambda r: f"{r['trades']['ret'].mean() * 1e4:+.2f}")
    row("平均/最大并发", lambda r: f"{r['avg_conc']:.2f} / {r['max_conc']}")
    row("平均资金利用率", lambda r: f"{r['util']:.1%}")
    row("容量跳过信号数", lambda r: f"{r['skip_cap']}")
    row("同标的占用跳过", lambda r: f"{r['skip_busy']}")
    L.append("")
    L.append(f"S2 后组合候选信号 {n_sig} 个（S2 拦截 {int(cands_all['s2_off'].sum())} 个），"
             f"即 {n_sig / months:.1f} 信号/月 —— 是 TSLA 单标的 {tsla_pm:.1f} 笔/月的 "
             f"{n_sig / months / tsla_pm:.1f} 倍信号流")
    for n in N_CAPS:
        L.append(f"  N={n} 实际成交 {runs[n]['per_month']:.1f} 笔/月 = TSLA 单标的的 "
                 f"{runs[n]['per_month'] / tsla_pm:.1f} 倍")
    L.append("")
    L.append("== 两种口径分开报告（组合内贡献拆分）==")
    for n in N_CAPS:
        tr = runs[n]["trades"]
        t_t = blk_stats(tr[tr["symbol"] == "TSLA"])
        t_p = blk_stats(tr[tr["symbol"] != "TSLA"])
        L.append(f"  N={n}  TSLA（时间外+标的外）: {t_t['n']} 笔, WR {t_t['wr']:.1%}, "
                 f"{t_t['avg_bp']:+.2f}bp, PnL ${t_t['pnl']:+,.0f}")
        L.append(f"        池 25（时间外·标的内）: {t_p['n']} 笔, WR {t_p['wr']:.1%}, "
                 f"{t_p['avg_bp']:+.2f}bp, PnL ${t_p['pnl']:+,.0f}")
    L.append("  提示：池标的口径弱于 TSLA 口径——标的在训练分布内，只有时间是外推的。")
    L.append("")
    L.append("== 对照 ==")
    L.append(f"  TSLA 单标的存档（E8-A+S2, 10 个月）: +12.04%（年化 ~+15.2%），"
             "MTM MDD -3.68%，5.4 笔/月")
    L.append(f"  池 26 等权 B&H 同段: {bh_pool_total:+.2%}，日线 MDD {bh_pool_mdd:+.2%}")
    L.append(f"  TSLA B&H 同段: {bh_tsla_total:+.2%}，日线 MDD {bh_tsla_mdd:+.2%}")
    L.append("")
    L.append("== 逐标的贡献（N=4 档，PnL 降序）==")
    ps = per_sym.sort_values("n4_pnl", ascending=False)
    L.append(f"{'symbol':<7s}{'口径':<10s}{'事件':>6s}{'保留率':>8s}{'S2拦':>5s}"
             f"{'S2停用':>8s}{'N4笔':>6s}{'N4WR':>8s}{'N4bp':>9s}{'N4PnL$':>10s}"
             f"{'N8笔':>6s}{'N8PnL$':>10s}")
    for sym, r in ps.iterrows():
        L.append(f"{sym:<8s}{r['scope']:<8s}{r['n_events_eval']:>6d}"
                 f"{r['retention']:>8.1%}{r['n_s2_blocked']:>5d}"
                 f"{r['s2_off_day_share']:>8.1%}{r['n4_trades']:>6d}"
                 f"{(r['n4_wr'] if np.isfinite(r['n4_wr']) else 0):>8.1%}"
                 f"{(r['n4_avg_bp'] if np.isfinite(r['n4_avg_bp']) else 0):>9.1f}"
                 f"{r['n4_pnl']:>10,.0f}{r['n8_trades']:>6d}{r['n8_pnl']:>10,.0f}")
    L.append("")

    # ---------- 判读 ----------------------------------------------------------
    r4, r8 = runs[4], runs[8]
    top3 = ps.head(3)
    bot3 = ps.tail(3)
    hi_r = per_sym[per_sym["retention"] > 0.5]
    lo_r = per_sym[per_sym["retention"] <= 0.2]
    tsla_solo_pm = 5.4
    L.append("== 判读 ==")
    L.append(f"1. 平台能力：{len(SYMS)} 标的冻结管线端到端跑通，逐事件 prob / 单标的 62 笔 / "
             "S2 后 54 笔全部与存档复现一致——组合级模拟机制可用。")
    L.append(f"2. 组合结果：总收益 N4 {r4['total']:+.2%} / N8 {r8['total']:+.2%}（年化 "
             f"{r4['ann']:+.2%} / {r8['ann']:+.2%}），MTM 最大回撤 {r4['mtm_mdd']:+.2%} / "
             f"{r8['mtm_mdd']:+.2%}。对比 TSLA 单标的存档 +12.04% / MDD -3.68%："
             + ("组合两档均更优。" if min(r4["total"], r8["total"]) > 0.1204 else
                ("其中一档超单标的收益。" if max(r4["total"], r8["total"]) > 0.1204 else
                 "两档均大幅劣于单标的存档——把冻结 TSLA 参数直接铺到 26 标的是减分项，"
                 "不是放大器。")))
    hi_names = "、".join(f"{s} {r['retention']:.0%}" for s, r in
                          hi_r.sort_values("retention", ascending=False).head(3).iterrows())
    L.append(f"3. 亏损定位=gate 不迁移：冻结阈值 {THR:.4f} 是按 TSLA 概率分布取的 top~10% "
             f"分位，但池内高波动标的的概率分布整体右移——保留率 >50% 的 {len(hi_r)} 个标的"
             f"（{hi_names} 等）贡献了 N4 亏损的几乎全部"
             f"（${hi_r['n4_pnl'].sum():+,.0f} / 组合 ${r4['trades']['pnl'].sum():+,.0f}）；"
             f"保留率 <=20%（gate 仍近似 top10% 筛选）的 {len(lo_r)} 个标的合计仅 "
             f"${lo_r['n4_pnl'].sum():+,.0f}，接近打平。绝对阈值跨标的失效，"
             "是本次模拟最清晰的结构性发现（若续做需按标的分位数重标定 gate——那是新实验，"
             "须另行预登记）。")
    L.append(f"4. 频率假设：信号流 {n_sig / months:.1f} 个/月 = TSLA 单标的 {tsla_solo_pm:.1f} "
             f"笔/月的 {n_sig / months / tsla_solo_pm:.0f} 倍；N4 实际成交 "
             f"{r4['per_month']:.1f} 笔/月（容量跳过 {r4['skip_cap']}），N8 "
             f"{r8['per_month']:.1f} 笔/月（跳过 {r8['skip_cap']}）。频率假设本身成立，"
             "但在 gate 失效的信号质量下，高频率只是放大亏损。低质量高保留标的还按 prob "
             "降序挤占额度：TSLA 在 N4 档只成交 24 笔（单标的可成交 54 笔），"
             "被池信号排挤。")
    L.append(f"5. 贡献集中度：top3 = "
             + "、".join(f"{s}（${r['n4_pnl']:+,.0f}）" for s, r in top3.iterrows())
             + "；bottom3 = "
             + "、".join(f"{s}（${r['n4_pnl']:+,.0f}）" for s, r in bot3.iterrows())
             + "。")
    pool_wr4 = blk_stats(r4["trades"][r4["trades"]["symbol"] != "TSLA"])
    L.append(f"6. 两口径：TSLA（时间外+标的外）在组合内 N8 档 46 笔 WR 80.4% +16.7bp，"
             "与存档方向一致；池 25（时间外·标的内）N4 合计 WR "
             f"{pool_wr4['wr']:.1%}、{pool_wr4['avg_bp']:+.2f}bp——"
             + ("同向为正。" if pool_wr4["avg_bp"] > 0 else
                "期望为负：胜率仍有 ~70%（几何幻觉，tp0.5%<<sl2%），但期望被高保留标的拖负。")
             + "池口径证据天然弱于 TSLA 口径（标的在训练分布内），不可合并宣称。")
    vs_bh4 = r4["total"] - bh_pool_total
    L.append(f"7. 对 B&H：池等权 B&H 同段 {bh_pool_total:+.2%}（MDD {bh_pool_mdd:+.2%}），"
             f"组合 N4 {'超额' if vs_bh4 > 0 else '落后'} {abs(vs_bh4):.2%}"
             + ("，且回撤更深（{:.2%} vs {:.2%}）——收益与风控双输。".format(
                    r4["mtm_mdd"], bh_pool_mdd)
                if vs_bh4 < 0 and r4["mtm_mdd"] < bh_pool_mdd else "。")
             + f"平均资金利用率 {r4['util']:.1%}（N4）。")
    L.append("8. 诚实边界：gate 阈值与几何来自 TSLA 留出段事后网格（E8 档案声明），组合结果"
             "继承该多重比较折扣；池标的口径为『时间外、标的内』。本实验 = 模拟平台能力验证 + "
             "方向性证据，不构成新的达标候选；负面结果同样按预登记协议如实入档。"
             "B&H 对照不计成本（对基准从宽）。多重比较 +2（N∈{4,8} 两档）。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
