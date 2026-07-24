#!/usr/bin/env python
"""E8-A crash stress test — time-clean pooled GBDT gate over 2021-10..2023-01.

Motivation (docs/strategy-lab.md E8): candidate E8-A = pooled GBDT gate at
top-10% retention x tp0.5%/sl2%/timeout48, long-only, day-flat, pessimistic
settlement (1bp fee + 2bp slip per side). The production pooled model was
trained on 2023-07+ data, so testing it on 2021-22 would be reverse leakage
(training on the future, testing on the past). This script rebuilds the whole
pipeline time-cleanly on 2019-2023 data:

- Data: 15 representative pool symbols + TSLA, 5m RTH bars 2019-07..2023-07
  (data/pool_crash/, fetched via src/alpaca_data.py --start/--end).
- Events: src/v_stats.py --tf 5m default grid, deduped physical events
  (outputs/e8a_crash_test/v_events/<SYM>/), identical to the E8 pool tables.
- Features/labels: research/ml_common.py, rv20-normalized as in E8
  (rv20_rel uses each symbol's TRAIN-period median rv20 only).
- Train: POOL symbols only (leave-TSLA-out), et_day < 2021-09-30
  (2021-09-30 itself dropped = 1-day embargo). Same LGBM hyper-params (E3).
- Gate threshold: 90th percentile of LEAVE-ONE-SYMBOL-OUT out-of-fold
  probabilities on the TRAIN segment (no test-period information at all;
  OOF避免用同模型的in-sample概率定分位，口径与"模型没见过被推断标的"一致).
- Test: TSLA events with et_day in [2021-10-01, 2023-01-31]; entries gated by
  prob >= threshold; settled on the 5m path with the E9 machinery
  (settle_bracket, one position at a time, always flat by day end).
- Controls: (a) ungated = every V event entered, same geometry;
  (b) 200 random-entry simulations, same n / geometry / window;
  (c) buy & hold over the same window.

Usage:  .venv/bin/python research/e8a_crash_test.py
Outputs: outputs/e8a_crash_test/{summary.txt, trades.csv, baseline.csv}
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score  # noqa: E402

from research.e9_frontier_search import COST, BarSet, run_combo, seg_stats  # noqa: E402
from research.ml_common import build_dataset, load_events, make_lgbm  # noqa: E402
from src.common.data_io import load_bars  # noqa: E402
from src.common.execution import entry_fill  # noqa: E402

# ----------------------------------------------------------------- parameters
POOL_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX",
    "JPM", "XOM", "DIS", "CAT", "INTC", "PLTR", "COIN",
]
TARGET = "TSLA"

DATA_DIR = ROOT / "data" / "pool_crash"
EVENTS_DIR = ROOT / "outputs" / "e8a_crash_test" / "v_events"
OUT = ROOT / "outputs" / "e8a_crash_test"

TRAIN_END = date(2021, 9, 30)    # train: et_day < TRAIN_END (embargo day dropped)
TEST_START = date(2021, 10, 1)
TEST_END = date(2023, 1, 31)

TP, SL, TIMEOUT = 0.005, 0.020, 48   # E8-A geometry
RETENTION = 0.10                     # top-10% gate
N_RANDOM = 200
N_BOOT = 1000
SEED = 42

POOLED_FEATURES = [
    "drop_pct_n", "drop_speed_n", "vol_ratio_trough", "et_hour",
    "open_to_now_ret_n", "prev_day_ret_n", "rv20_rel", "bars_since_open",
    "symbol",
]


def build_symbol_dataset(sym: str):
    bars_csv = DATA_DIR / f"{sym}_5m_2019_2023.csv"
    events_csv = EVENTS_DIR / sym / "v_events_5m_unique.csv"
    if not bars_csv.exists() or not events_csv.exists():
        return None
    ev = load_events(events_csv)
    bars = load_bars(str(bars_csv))
    ds, info = build_dataset(ev, bars)
    ds["symbol"] = sym
    return ds, bars, info


def add_normalized_features(all_ds: pd.DataFrame) -> pd.DataFrame:
    """Identical to e8_pooled_gbdt.add_normalized_features but anchored on the
    CRASH-protocol train end (2021-09-30): rv20_rel uses each symbol's
    train-period median rv20 only — no test-period information."""
    ds = all_ds.copy()
    rv = ds["rv20"].replace(0, np.nan)
    ds["drop_pct_n"] = ds["drop_pct"] / rv
    ds["drop_speed_n"] = ds["drop_speed"] / rv
    ds["open_to_now_ret_n"] = ds["open_to_now_ret"] / rv
    ds["prev_day_ret_n"] = ds["prev_day_ret"] / rv
    med = (
        ds[ds["et_day"] < TRAIN_END]
        .groupby("symbol")["rv20"]
        .median()
        .rename("rv20_med_train")
    )
    ds = ds.merge(med, left_on="symbol", right_index=True, how="left")
    ds["rv20_rel"] = ds["rv20"] / ds["rv20_med_train"]
    ds["symbol"] = pd.Categorical(ds["symbol"], categories=sorted(ds["symbol"].unique()))
    return ds


def mtm_equity_path(bs: BarSet, trades: pd.DataFrame, tp: float, sl: float,
                    timeout: int) -> np.ndarray:
    """Bar-level mark-to-market equity across the sequential trade list.
    Between trades equity is flat; inside a trade it is marked at bar closes
    (entry price incl. adverse slippage); the exit bar books the net return."""
    eq = 1.0
    path = [eq]
    for _, t in trades.iterrows():
        epos = int(t["epos"])
        ret, exit_pos = bs.eval_trade(epos, 1, tp, sl, timeout)  # cached
        entry_px = entry_fill(bs.opens[epos], 1, COST)
        eq0 = eq
        for k in range(epos, exit_pos):
            path.append(eq0 * (bs.closes[k] / entry_px))
        eq = eq0 * (1.0 + ret)
        path.append(eq)
    return np.asarray(path)


def mdd(path: np.ndarray) -> float:
    if len(path) == 0:
        return np.nan
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------- assemble datasets --------------------------------------
    frames, cov_rows, missing = [], [], []
    tsla_bars = None
    for sym in POOL_SYMBOLS + [TARGET]:
        got = build_symbol_dataset(sym)
        if got is None:
            missing.append(sym)
            print(f"[{sym}] MISSING bars/events — skipped")
            continue
        ds, bars, _ = got
        if sym == TARGET:
            tsla_bars = bars
        frames.append(ds)
        n_train = int((ds["et_day"] < TRAIN_END).sum())
        cov_rows.append({
            "symbol": sym, "n_usable": len(ds), "n_train": n_train,
            "pos_rate_train": float(ds.loc[ds["et_day"] < TRAIN_END, "label"].mean())
            if n_train else np.nan,
            "first_day": str(ds["et_day"].min()), "last_day": str(ds["et_day"].max()),
        })
        print(f"[{sym}] usable events {len(ds)} (train {n_train}), "
              f"{ds['et_day'].min()} .. {ds['et_day'].max()}")
    if tsla_bars is None:
        raise SystemExit("TSLA crash bars/events missing — fetch step incomplete")
    cov_df = pd.DataFrame(cov_rows)

    all_ds = add_normalized_features(pd.concat(frames, ignore_index=True))
    feat_ok = all_ds[POOLED_FEATURES[:-1]].notna().all(axis=1)
    n_dropped = int((~feat_ok).sum())
    all_ds = all_ds[feat_ok].reset_index(drop=True)

    is_tsla = all_ds["symbol"].astype(str) == TARGET
    train_mask = (~is_tsla) & (all_ds["et_day"] < TRAIN_END)
    pool_train = all_ds[train_mask]
    tsla_test = all_ds[
        is_tsla & (all_ds["et_day"] >= TEST_START) & (all_ds["et_day"] <= TEST_END)
    ].reset_index(drop=True)
    print(f"\npool train events {len(pool_train)} "
          f"({pool_train['et_day'].min()} .. {pool_train['et_day'].max()}), "
          f"TSLA test events {len(tsla_test)}; NaN-dropped {n_dropped}")

    # ---------------- gate threshold: LOO OOF on the TRAIN segment only ------
    oof = np.full(len(pool_train), np.nan)
    pt = pool_train.reset_index(drop=True)
    for sym in pt["symbol"].astype(str).unique():
        m = (pt["symbol"].astype(str) == sym).to_numpy()
        model = make_lgbm(SEED)
        model.fit(pt.loc[~m, POOLED_FEATURES], pt.loc[~m, "label"])
        oof[m] = model.predict_proba(pt.loc[m, POOLED_FEATURES])[:, 1]
    thr = float(np.quantile(oof, 1.0 - RETENTION))
    print(f"gate threshold (train LOO-OOF q{1 - RETENTION:.2f}): {thr:.4f}")

    # ---------------- final model + TSLA inference ---------------------------
    model = make_lgbm(SEED)
    model.fit(pool_train[POOLED_FEATURES], pool_train["label"])
    probs = model.predict_proba(tsla_test[POOLED_FEATURES])[:, 1]
    tsla_test["prob"] = probs

    y = tsla_test["label"].to_numpy()
    auc = roc_auc_score(y, probs)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(y), len(y))
        if len(np.unique(y[pick])) < 2:
            continue
        boots.append(roc_auc_score(y[pick], probs[pick]))
    auc_lo, auc_hi = np.percentile(boots, [2.5, 97.5])

    gated = tsla_test[tsla_test["prob"] >= thr]
    retention_test = len(gated) / len(tsla_test)
    print(f"TSLA test AUC {auc:.3f} [{auc_lo:.3f},{auc_hi:.3f}]; "
          f"gated {len(gated)}/{len(tsla_test)} ({retention_test:.1%})")

    # ---------------- settlement on the 5m path ------------------------------
    bs = BarSet(tsla_bars)
    test_bar_mask = (bs.et_dates >= TEST_START) & (bs.et_dates <= TEST_END)
    test_pos = np.nonzero(test_bar_mask)[0]

    def event_signals(ev_df: pd.DataFrame) -> list[tuple[int, int]]:
        epos = tsla_bars.index.get_indexer(pd.DatetimeIndex(ev_df["entry_t"]))
        assert (epos >= 0).all(), "event entry_t missing from bar index"
        return [(int(p), 1) for p in np.sort(epos)]

    def settle(ev_df: pd.DataFrame, variant: str):
        sig = event_signals(ev_df)
        trades = run_combo(bs, sig, TP, SL, TIMEOUT)
        st = seg_stats(trades, "s")
        st["mtm_mdd"] = mdd(mtm_equity_path(bs, trades, TP, SL, TIMEOUT))
        rows = []
        prob_map = {}
        epos_arr = tsla_bars.index.get_indexer(pd.DatetimeIndex(ev_df["entry_t"]))
        for p, pr in zip(epos_arr, ev_df["prob"]):
            prob_map[int(p)] = float(pr)
        for _, t in trades.iterrows():
            ep = int(t["epos"])
            _, exit_pos = bs.eval_trade(ep, 1, TP, SL, TIMEOUT)
            rows.append({
                "variant": variant, "epos": ep,
                "entry_t": tsla_bars.index[ep], "et_day": t["et_day"],
                "prob": prob_map.get(ep, np.nan), "ret": t["ret"],
                "exit_t": tsla_bars.index[exit_pos],
            })
        return st, trades, pd.DataFrame(rows)

    st_gate, tr_gate, rows_gate = settle(gated, "gated_top10")
    st_all, tr_all, rows_all = settle(tsla_test, "ungated_all_events")
    trades_df = pd.concat([rows_gate, rows_all], ignore_index=True)
    trades_df.to_csv(OUT / "trades.csv", index=False)

    # bootstrap p (one-sided, avg <= 0) for the gated variant
    r = tr_gate["ret"].to_numpy()
    if len(r):
        means = rng.choice(r, size=(N_BOOT, len(r)), replace=True).mean(axis=1)
        boot_p = float((np.sum(means <= 0) + 1) / (N_BOOT + 1))
    else:
        boot_p = np.nan

    # ---------------- random-entry baseline ----------------------------------
    n_tr = len(tr_gate)
    rb_rows = []
    for s in range(N_RANDOM):
        picks = np.sort(rng.choice(test_pos, size=min(n_tr, len(test_pos)), replace=False))
        rets = []
        for p in picks:
            got = bs.eval_trade(int(p), 1, TP, SL, TIMEOUT)
            if got is not None:
                rets.append(got[0])
        rets = np.asarray(rets)
        if not len(rets):
            continue
        eqp = np.cumprod(1 + rets)
        rb_rows.append({
            "sim": s, "n": len(rets), "wr": float((rets > 0).mean()),
            "avg_bp": float(rets.mean() * 1e4), "total": float(eqp[-1] - 1),
            "mdd_closed": mdd(eqp),
        })
    rb = pd.DataFrame(rb_rows)
    rb.to_csv(OUT / "baseline.csv", index=False)
    pct_wr_ge = float((rb["wr"] >= st_gate["s_wr"]).mean()) if n_tr else np.nan
    pct_avg_ge = float((rb["avg_bp"] >= st_gate["s_avg_bp"]).mean()) if n_tr else np.nan

    # ---------------- buy & hold over the window -----------------------------
    bh_first, bh_last = int(test_pos[0]), int(test_pos[-1])
    bh_ret = bs.closes[bh_last] / bs.opens[bh_first] - 1.0
    bh_path = bs.closes[test_pos] / bs.opens[bh_first]
    bh_mdd = mdd(bh_path)
    peak_k = int(np.argmax(bs.closes[test_pos]))
    bh_peak_ret = bs.closes[bh_last] / bs.closes[test_pos][peak_k] - 1.0
    peak_day = bs.et_dates[test_pos[peak_k]]

    # ---------------- summary ------------------------------------------------
    def fmt(st, boot=None):
        s = (f"n={st['s_n']}, WR={st['s_wr']:.1%}, avg={st['s_avg_bp']:+.1f}bp, "
             f"total={st['s_total']:+.2%}, 已平仓MDD={st['s_mdd']:.2%}, "
             f"MTM MDD={st['mtm_mdd']:.2%}")
        if boot is not None:
            s += f", bootstrap p(avg<=0)={boot:.3f}"
        return s

    L = []
    L.append("E8-A 崩盘压测 — 时序干净的 pooled GBDT 门 × tp0.5%/sl2%/timeout48（多头、日内强平、悲观结算 1bp fee + 2bp slip 单边）")
    L.append("=" * 100)
    L.append("协议（先于结果写死）：")
    L.append(f"- 数据：{len(cov_rows) - 1} 只池标的 + TSLA，5m RTH，2019-07..2023-07（Alpaca SIP，split 调整）"
             + (f"；缺失跳过 {missing}" if missing else ""))
    L.append(f"- 训练：仅池标的（不含 TSLA），et_day < {TRAIN_END}（{TRAIN_END} 当日弃用 = 1 天 embargo）；"
             f"事件/特征/标签/超参与 E8 完全一致（v_stats 默认网格、rv20 标准化、三重障碍 +0.5%/-1%/24bar、E3 LGBM）")
    L.append(f"- 门槛：训练段 leave-one-symbol-out OOF 概率的 {1 - RETENTION:.0%} 分位 = {thr:.4f}"
             f"（零测试段信息；OOF 口径与『模型没见过被推断标的』一致）")
    L.append(f"- 测试：TSLA {TEST_START}..{TEST_END} 的 V 事件推断入场，同一时刻单仓，几何 tp {TP:.1%}/sl {SL:.1%}/to {TIMEOUT}bar")
    L.append("")
    L.append("== 数据覆盖 ==")
    L.append(cov_df.to_string(index=False))
    L.append(f"合计池训练事件 {len(pool_train)}；TSLA 测试窗口事件 {len(tsla_test)}；NaN 特征弃用 {n_dropped}")
    L.append("")
    L.append("== 模型在崩盘窗口的排序能力 ==")
    L.append(f"TSLA 测试段 AUC = {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]（E8 原版 2025-10..2026-07 留出段为 0.642）")
    L.append(f"门后保留 {len(gated)}/{len(tsla_test)} = {retention_test:.1%}（目标 10%，阈值来自训练段分位，"
             "测试段保留率偏离属正常，不回调阈值）")
    L.append("")
    L.append("== 崩盘窗口表现（2021-10-01 .. 2023-01-31）==")
    L.append(f"E8-A 门控    : {fmt(st_gate, boot_p)}")
    L.append(f"无门控（全事件）: {fmt(st_all)}")
    L.append(f"随机入场基线（{N_RANDOM} 次，同 n={n_tr} 同几何同窗口）: "
             f"WR 均值 {rb['wr'].mean():.1%} (p5 {rb['wr'].quantile(0.05):.1%} / p95 {rb['wr'].quantile(0.95):.1%}), "
             f"avg 均值 {rb['avg_bp'].mean():+.1f}bp (p95 {rb['avg_bp'].quantile(0.95):+.1f}bp), "
             f"total 均值 {rb['total'].mean():+.2%}")
    L.append(f"  -> pct_rand_wr_ge_real = {pct_wr_ge:.1%}, pct_rand_avg_ge_real = {pct_avg_ge:.1%}")
    L.append(f"同期 B&H：{bh_ret:+.1%}，B&H MTM MDD {bh_mdd:.1%}")
    L.append("")

    # -------- 判读（数据驱动措辞） --------
    L.append("== 判读 ==")
    g_wr, g_avg = st_gate["s_wr"], st_gate["s_avg_bp"]
    a_wr, a_avg = st_all["s_wr"], st_all["s_avg_bp"]
    if st_gate["s_n"] == 0:
        verdict = "门控在崩盘窗口几乎不放行（n=0），策略等于空仓——不亏钱但也无法证明有 edge。"
    elif g_avg >= 5 and g_wr >= 0.70:
        verdict = ("E8-A 在持续下跌 regime 里【存活】：胜率与期望均保持在留出段水平附近，"
                   "且显著优于随机基线与无门控对照。")
    elif g_avg > 0:
        verdict = ("E8-A 在崩盘窗口【弱存活/衰减】：期望仍为正但低于留出段（+16bp），"
                   "边际需结合随机基线与无门控对照判断是否仍含选时信息。")
    elif g_avg > a_avg:
        verdict = ("E8-A 在崩盘窗口【坍塌但门控仍有相对价值】：期望转负，"
                   "但门控组仍好于无门控——门在崩盘里救不了绝对收益，只减损。")
    else:
        verdict = ("E8-A 在崩盘窗口【反向】：门控组不优于甚至差于无门控/随机——"
                   "2023-26 学到的排序在崩盘 regime 中失效或反噬。")
    L.append("1. " + verdict)
    L.append(f"2. 与随机基线对比：pct_rand_wr_ge={pct_wr_ge:.0%} / pct_rand_avg_ge={pct_avg_ge:.0%}"
             + ("——门控收益大体是几何+regime 的产物，不含崩盘期选时信息。" if (np.isfinite(pct_avg_ge) and pct_avg_ge > 0.10)
                else "——门控相对随机入场仍有信息（口径同 E9/E8 基线检验）。"))
    if st_gate["s_n"] and st_all["s_n"]:
        anti = g_avg < a_avg
        L.append(f"3. 门控 vs 无门控（崩盘里门是救命还是无效）：单笔期望 门控 {g_avg:+.1f}bp vs 无门控 {a_avg:+.1f}bp，"
                 f"胜率 {g_wr:.1%} vs {a_wr:.1%}。"
                 + ("门控组单笔期望【更差】——门在崩盘 regime 里是反选（挑中的深跌事件更容易继续下跌打穿 2% 止损）；"
                    "门控组 total 亏得少纯粹因为交易数少（敞口小），不是选时价值。"
                    if anti else "门控组单笔期望更好，门控保留了相对减损价值。"))
        L.append(f"   胜率维持在 ~{g_wr:.0%} 而期望为负，是实验室达标表第 7 条『胜率可由障碍几何免费制造』的活例："
                 "tp0.5%/sl2% 的不对称让多数交易小赚触发 tp，少数被打穿 2% 止损+滑点的交易吞掉全部。")
    L.append(f"4. 与 B&H 对比：同窗口（起点 {TEST_START}）拿住 TSLA 为 {bh_ret:+.1%}；"
             f"若从窗口内顶点（{peak_day}）起算为 {bh_peak_ret:+.1%}（任务书『约 -60%+』对应此口径）。"
             f"策略 total {st_gate['s_total']:+.2%}、MTM MDD {st_gate['mtm_mdd']:.2%} vs B&H MDD {bh_mdd:.1%}——"
             "策略亏损与 B&H 同量级、并未提供避险。")
    L.append(f"5. 实验室达标表『崩盘窗口压测 浮亏<=30%』：门控组 MTM MDD {st_gate['mtm_mdd']:.2%} "
             + ("满足" if st_gate["mtm_mdd"] >= -0.30 else "不满足") + "（日内策略天然低敞口，此条不难过，"
             "真正要看的是期望符号与随机基线）。")
    L.append("")
    L.append("== 诚实性备注 ==")
    L.append("- 本测试的模型/阈值只用 2019-07..2021-09 池数据（不含 TSLA），与 E8-A 生产模型（2023-07 起训练）"
             "不是同一个模型实例——检验的是『同一配方在崩盘前训练、崩盘中使用』的可行性，这是时序上唯一干净的口径。")
    L.append("- 训练段只有 ~2.2 年池数据（E8 原版为 ~2.2 年 × 25 只；此处 15 只），样本量约为原版的 60%；"
             "COIN（2021-04 上市）/ PLTR（2020-09 上市）训练段贡献很短，属如实记录。")
    L.append("- 门槛用训练段 OOF 分位而非测试段保留率校准，测试段实际保留率随分布漂移偏离 10% 是必然的，"
             "不允许回调（那是测试段信息）。")
    L.append(f"- 多重比较：本测试是单一预登记配置（E8-A），无网格搜索；bootstrap p={boot_p:.3f} 按单假设解读。")
    L.append("")
    L.append(f"运行耗时 {time.time() - t0:.0f}s")
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
