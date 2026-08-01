"""E17-A Monte Carlo risk deduction — leverage on the E8-A+S2 holdout stream.

PRE-REGISTration (frozen before first run, zero tuning):
  Base stream : outputs/e8a_replay/trades.csv, blocked==False, 54 trades in
                entry order (archived holdout 2025-10-01..2026-07-22, 294 cal
                days).  Contamination pool: outputs/e8a_crash_test/trades.csv
                variant gated_top10, 177 trades (2021-10-01..2023-01-31).
  Resampling  : N = 10000 paths per scheme, seed = 20260801, path length 54.
    (1) iid    — bootstrap with replacement from the 54 (order destroyed).
    (2) block  — circular block bootstrap, block length 5 (preserves local
                 serial dependence), 11 blocks truncated to 54.
    (3) contam — 11 blocks of length 5; each block independently with
                 p = 0.20 is a random contiguous slice of the crash-177
                 stream, else a circular block of the holdout 54.  Models the
                 "holdout was fair weather" optimism bias with archived bad-
                 regime trades; p=0.2 is an assumption, not an estimate.
  Leverage    : L in {1.0, 1.5, 2.0}; per-trade ret_lev = L*ret − (L−1)*6.5%/yr
                * actual holding seconds (same as e17_ab_tracks.lever_stream).
  Annualise   : every path keeps the archived time base (54 trades ≙ 294 cal
                days): ann = (1+total)^(365/294) − 1.  Loss-year prob =
                P(total < 0).
  MTM         : per-trade MAE (worst 5m-close excursion vs entry fill) is
                measured once from archived bars and travels with the trade;
                path MTM MDD interleaves eq_{t-1}*(1+L*MAE_t) troughs with
                closed equity.  P(DD>20%) and P(DD ≥ margin-call distance
                r_call(L)) are equity-drawdown proxies; the literal margin
                call needs a single-position price move of r_call(L) which the
                frozen sl2% geometry cannot reach without an intra-bar crash.
  Order risk  : same 54 trades, losses-grouped-first (provably the max-MDD
                permutation) vs losses spread round-robin (heuristic min).
  Outputs     : outputs/e17a_mc/{grid.csv, worst1pct.csv, order_risk.csv,
                summary.txt}.  Multiple-comparison ledger: 0 new hypotheses —
                this is risk pricing of the already-registered E17-A candidate,
                not a search.
Honest bounds: 54 trades is a tiny sample — bootstrap only reshuffles what was
seen; unseen regimes enter only through the crash-injection approximation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402

OUT = ROOT / "outputs" / "e17a_mc"
HOLDOUT_TRADES = ROOT / "outputs" / "e8a_replay" / "trades.csv"
CRASH_TRADES = ROOT / "outputs" / "e8a_crash_test" / "trades.csv"
HOLDOUT_BARS = ROOT / "data" / "TSLA_5m_rolling.csv"
CRASH_BARS = ROOT / "data" / "pool_crash" / "TSLA_5m_2019_2023.csv"

SEED = 20260801
N_PATHS = 10_000
BLOCK = 5
P_CONTAM = 0.20
LEVERAGE = [1.0, 1.5, 2.0]
FIN_RATE = 0.065
SLIP = 0.0002
YEAR_S = 365.25 * 24 * 3600
CAL_DAYS = 294.0          # archived holdout window
MAINT_MARGIN = 0.30


def r_call(lev: float) -> float:
    """Adverse price move that triggers a margin call (same as e17_ab)."""
    if lev <= 1.0:
        return -1.0
    return (MAINT_MARGIN * lev - 1.0) / ((1.0 - MAINT_MARGIN) * lev)


# ---------------------------------------------------------------- data
def load_holdout() -> pd.DataFrame:
    t = pd.read_csv(HOLDOUT_TRADES)
    t = t[~t["blocked"]].copy()
    t["entry_utc"] = pd.to_datetime(t["entry_t_utc"], utc=True)
    t["exit_utc"] = (
        pd.to_datetime(t["exit_et"]).dt.tz_localize(ET).dt.tz_convert("UTC"))
    t["entry_px"] = t["entry_fill"].astype(float)
    return t.sort_values("entry_utc").reset_index(drop=True)


def load_crash() -> pd.DataFrame:
    t = pd.read_csv(CRASH_TRADES)
    t = t[t["variant"] == "gated_top10"].copy()
    t["entry_utc"] = pd.to_datetime(t["entry_t"], utc=True)
    t["exit_utc"] = pd.to_datetime(t["exit_t"], utc=True)
    return t.sort_values("entry_utc").reset_index(drop=True)


def attach_mae(t: pd.DataFrame, bars: pd.DataFrame,
               entry_from_bars: bool) -> pd.DataFrame:
    """Worst close-to-entry excursion (MAE, <=0) inside each holding window."""
    close, op = bars["Close"], bars["Open"]
    maes, holds = [], []
    for _, tr in t.iterrows():
        entry_px = (float(op.loc[tr["entry_utc"]]) * (1 + SLIP)
                    if entry_from_bars else float(tr["entry_px"]))
        seg = close.loc[tr["entry_utc"]: tr["exit_utc"]]
        mae = float((seg / entry_px - 1.0).min()) if len(seg) else 0.0
        maes.append(min(mae, 0.0))
        holds.append(max((tr["exit_utc"] - tr["entry_utc"]).total_seconds(),
                         300.0))
    t = t.copy()
    t["mae"] = maes
    t["hold_s"] = holds
    return t


# ---------------------------------------------------------------- resampling
def sample_indices(rng: np.random.Generator, n_hold: int, n_crash: int
                   ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Index matrices (N_PATHS x 54) into the combined pool [holdout|crash].

    Returns {scheme: (idx, n_crash_blocks_per_path)}.
    """
    L = n_hold                     # path length = 54
    nb = int(np.ceil(L / BLOCK))   # 11 blocks

    iid = rng.integers(0, n_hold, size=(N_PATHS, L))

    starts = rng.integers(0, n_hold, size=(N_PATHS, nb))
    offs = np.arange(BLOCK)
    blk = ((starts[:, :, None] + offs[None, None, :]) % n_hold)
    block_idx = blk.reshape(N_PATHS, nb * BLOCK)[:, :L]

    coin = rng.random(size=(N_PATHS, nb)) < P_CONTAM
    h_starts = rng.integers(0, n_hold, size=(N_PATHS, nb))
    c_starts = rng.integers(0, n_crash - BLOCK + 1, size=(N_PATHS, nb))
    h_blk = (h_starts[:, :, None] + offs[None, None, :]) % n_hold
    c_blk = n_hold + c_starts[:, :, None] + offs[None, None, :]
    contam = np.where(coin[:, :, None], c_blk, h_blk)
    contam_idx = contam.reshape(N_PATHS, nb * BLOCK)[:, :L]
    # blocks that actually made the truncated path (block 11 keeps 4 trades)
    n_crash_blocks = coin.sum(axis=1)

    return {"iid": (iid, np.zeros(N_PATHS, dtype=int)),
            "block": (block_idx, np.zeros(N_PATHS, dtype=int)),
            "contam": (contam_idx, n_crash_blocks)}


def path_metrics(idx: np.ndarray, pool_ret: np.ndarray, pool_finu: np.ndarray,
                 pool_mae: np.ndarray, lev: float) -> dict[str, np.ndarray]:
    """Vectorised per-path stats for one leverage level."""
    r = lev * pool_ret[idx] - (lev - 1.0) * pool_finu[idx]     # N x 54
    mae = np.minimum(lev * pool_mae[idx], 0.0)
    eq = np.cumprod(1.0 + r, axis=1)
    eq_prev = np.concatenate([np.ones((eq.shape[0], 1)), eq[:, :-1]], axis=1)
    trough = eq_prev * (1.0 + mae)
    inter = np.empty((eq.shape[0], eq.shape[1] * 2))
    inter[:, 0::2] = trough
    inter[:, 1::2] = eq
    runmax = np.maximum.accumulate(np.maximum(inter, 1.0 - 1e-12), axis=1)
    mdd = (inter / runmax - 1.0).min(axis=1)
    total = eq[:, -1] - 1.0
    ann = (1.0 + total) ** (365.0 / CAL_DAYS) - 1.0
    loss_streak = _max_loss_streak(r)
    n_sl = (r < -0.014 * lev).sum(axis=1)     # sl-class exits (net ~ -2% x L)
    return {"total": total, "ann": ann, "mdd": mdd,
            "loss_streak": loss_streak, "n_sl": n_sl}


def _max_loss_streak(r: np.ndarray) -> np.ndarray:
    neg = r < 0
    out = np.zeros(r.shape[0], dtype=int)
    cur = np.zeros(r.shape[0], dtype=int)
    for j in range(r.shape[1]):
        cur = np.where(neg[:, j], cur + 1, 0)
        out = np.maximum(out, cur)
    return out


def seq_mdd(rets: np.ndarray, maes: np.ndarray) -> float:
    eq = np.cumprod(1.0 + rets)
    eq_prev = np.concatenate([[1.0], eq[:-1]])
    inter = np.empty(len(rets) * 2)
    inter[0::2] = eq_prev * (1.0 + np.minimum(maes, 0.0))
    inter[1::2] = eq
    runmax = np.maximum.accumulate(np.maximum(inter, 1.0))
    return float((inter / runmax - 1.0).min())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    hold = attach_mae(load_holdout(), load_bars(str(HOLDOUT_BARS)), False)
    crash = attach_mae(load_crash(), load_bars(str(CRASH_BARS)), True)
    n_h, n_c = len(hold), len(crash)
    assert n_h == 54 and n_c == 177, (n_h, n_c)
    print(f"[data] holdout n={n_h} total {(1 + hold.ret).prod() - 1:+.4%} "
          f"(archive +12.04%) | crash n={n_c} "
          f"total {(1 + crash.ret).prod() - 1:+.4%} (archive -31.42%)")
    print(f"[data] MAE holdout min {hold.mae.min():+.4%} / "
          f"crash min {crash.mae.min():+.4%}")

    pool_ret = np.concatenate([hold.ret.to_numpy(), crash.ret.to_numpy()])
    pool_mae = np.concatenate([hold.mae.to_numpy(), crash.mae.to_numpy()])
    pool_finu = FIN_RATE * np.concatenate(
        [hold.hold_s.to_numpy(), crash.hold_s.to_numpy()]) / YEAR_S

    schemes = sample_indices(rng, n_h, n_c)

    grid_rows, worst_rows = [], []
    for scheme, (idx, ncb) in schemes.items():
        for lev in LEVERAGE:
            m = path_metrics(idx, pool_ret, pool_finu, pool_mae, lev)
            ann, mdd_a, total = m["ann"], m["mdd"], m["total"]
            call = r_call(lev)
            q = np.percentile(ann, [5, 25, 50, 75, 95])
            grid_rows.append({
                "scheme": scheme, "L": lev,
                "ann_p5": q[0], "ann_p25": q[1], "ann_med": q[2],
                "ann_p75": q[3], "ann_p95": q[4],
                "p_loss_year": float((total < 0).mean()),
                "mdd_med": float(np.median(mdd_a)),
                "mdd_p95_deep": float(np.percentile(mdd_a, 5)),
                "p_dd_gt20": float((mdd_a <= -0.20).mean()),
                "r_call": call,
                "p_dd_ge_call": float((mdd_a <= call).mean()),
            })
            # worst 1% by total return
            k = max(1, N_PATHS // 100)
            wsel = np.argsort(total)[:k]
            worst_rows.append({
                "scheme": scheme, "L": lev, "n_paths": k,
                "ann_mean": float(ann[wsel].mean()),
                "ann_min": float(ann.min()),
                "total_mean": float(total[wsel].mean()),
                "mdd_mean": float(mdd_a[wsel].mean()),
                "mdd_worst": float(mdd_a.min()),
                "sl_hits_mean": float(m["n_sl"][wsel].mean()),
                "sl_hits_all_mean": float(m["n_sl"].mean()),
                "loss_streak_mean": float(m["loss_streak"][wsel].mean()),
                "crash_blocks_mean": float(ncb[wsel].mean()),
                "crash_blocks_all_mean": float(ncb.mean()),
            })
            print(f"[mc] {scheme:6s} L={lev:.1f} med_ann {q[2]:+.2%} "
                  f"p_loss {grid_rows[-1]['p_loss_year']:.3f} "
                  f"p_dd20 {grid_rows[-1]['p_dd_gt20']:.4f}")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "grid.csv", index=False)
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(OUT / "worst1pct.csv", index=False)

    # ------------------------------------------------------------ order risk
    order_rows = []
    base_ret = hold.ret.to_numpy()
    base_mae = hold.mae.to_numpy()
    base_finu = FIN_RATE * hold.hold_s.to_numpy() / YEAR_S
    loser_i = np.where(base_ret < 0)[0]
    winner_i = np.where(base_ret >= 0)[0]
    # spread: losers round-robin at even spacing among winners
    spread = []
    gap = len(winner_i) / (len(loser_i) + 1)
    wq = list(winner_i)
    lq = list(loser_i)
    next_at = gap
    placed = 0.0
    while wq or lq:
        if lq and placed >= next_at:
            spread.append(lq.pop(0))
            next_at += gap
        elif wq:
            spread.append(wq.pop(0))
            placed += 1
        else:
            spread.append(lq.pop(0))
    spread = np.asarray(spread)
    for lev in LEVERAGE:
        rl = lev * base_ret - (lev - 1.0) * base_finu
        ml = lev * base_mae
        asc = np.argsort(rl)              # losses grouped first = max MDD
        desc = asc[::-1]                  # losses grouped last (also bad)
        rows = {
            "worst_losses_grouped": seq_mdd(rl[asc], ml[asc]),
            "losses_at_end": seq_mdd(rl[desc], ml[desc]),
            "best_losses_spread": seq_mdd(rl[spread], ml[spread]),
            "archived_order": seq_mdd(rl, ml),
        }
        order_rows.append({"L": lev, **rows,
                           "worst_minus_best":
                               rows["worst_losses_grouped"]
                               - rows["best_losses_spread"]})
    order = pd.DataFrame(order_rows)
    order.to_csv(OUT / "order_risk.csv", index=False)

    # ------------------------------------------------------------ summary
    fmt = lambda v: f"{v * 100:+.1f}%"  # noqa: E731
    lines = [
        "E17-A 蒙特卡洛风险推演 — 杠杆 x 重抽样（预登记零调参，seed=20260801，"
        f"N={N_PATHS}/方案）",
        "=" * 100,
        f"基底：留出段 54 笔（2025-10-01..2026-07-22，294 自然日，存档核对 "
        f"total {(1 + hold.ret).prod() - 1:+.2%}）；"
        f"污染池：崩盘段 177 笔（2021-10..2023-01，total "
        f"{(1 + crash.ret).prod() - 1:+.2%}）。",
        "三种抽样：iid bootstrap（打散顺序）| 块 bootstrap（块长 5，保留序列相关）| "
        f"污染注入（每 5 笔块以 {P_CONTAM:.0%} 概率替换为崩盘段随机片段）。",
        "杠杆逐笔净收益 = L x ret − (L−1) x 6.5%/年 x 实际持仓时长（同 e17_ab 口径）。",
        "年化 = (1+total)^(365/294)−1，全部路径沿用存档时基；回撤 = 含逐笔 MAE 盘中"
        "谷底的 MTM 口径。",
        "",
        "== 九宫格风险定价表（3 抽样 x 3 杠杆）==",
    ]
    g = grid.set_index(["scheme", "L"])
    hdr = (f"{'抽样':8s} {'L':>4s} {'年化p5':>8s} {'p25':>8s} {'中位':>8s} "
           f"{'p75':>8s} {'p95':>8s} {'亏损年':>7s} {'回撤中位':>9s} "
           f"{'P(DD>20%)':>10s} {'P(DD≥追缴)':>10s}")
    lines.append(hdr)
    for scheme in ["iid", "block", "contam"]:
        for lev in LEVERAGE:
            r = g.loc[(scheme, lev)]
            lines.append(
                f"{scheme:8s} {lev:4.1f} {fmt(r.ann_p5):>8s} "
                f"{fmt(r.ann_p25):>8s} {fmt(r.ann_med):>8s} "
                f"{fmt(r.ann_p75):>8s} {fmt(r.ann_p95):>8s} "
                f"{r.p_loss_year * 100:6.1f}% {r.mdd_med * 100:8.1f}% "
                f"{r.p_dd_gt20 * 100:9.2f}% {r.p_dd_ge_call * 100:9.2f}%")
    lines += ["", "== 最坏 1% 路径的样貌（各方案 x L，按 total 取最差 100 条）=="]
    for _, r in worst.iterrows():
        extra = (f"，崩盘块均值 {r.crash_blocks_mean:.1f}/11"
                 f"（全体均值 {r.crash_blocks_all_mean:.1f}）"
                 if r.scheme == "contam" else "")
        lines.append(
            f"{r.scheme:6s} L={r.L:.1f}: 年化均值 {fmt(r.ann_mean)}"
            f"（极值 {fmt(r.ann_min)}），MTM 回撤均值 {r.mdd_mean * 100:.1f}%"
            f"（极值 {r.mdd_worst * 100:.1f}%），SL 级止损 "
            f"{r.sl_hits_mean:.1f} 次（全体均值 {r.sl_hits_all_mean:.1f}），"
            f"最长连亏 {r.loss_streak_mean:.1f} 笔{extra}")
    lines += ["", "== 顺序风险（同一批 54 笔，排列的代价）=="]
    for _, r in order.iterrows():
        lines.append(
            f"L={r.L:.1f}: 最差排列（亏损全聚）MDD {r.worst_losses_grouped * 100:.1f}% "
            f"| 亏损压尾 {r.losses_at_end * 100:.1f}% "
            f"| 最好排列（亏损匀开）{r.best_losses_spread * 100:.1f}% "
            f"| 实际历史序 {r.archived_order * 100:.1f}% "
            f"→ 最差−最好 = {r.worst_minus_best * 100:.1f}pp")
    lines += [
        "",
        "== 追缴口径注 ==",
        "P(DD≥追缴) 用的是权益 MTM 回撤达到 r_call(L) 距离的代理口径"
        "（L=1.5 为 −52.4%，L=2.0 为 −28.6%）；字面的追缴需要单笔持仓期内价格"
        "逆行到该距离——冻结 sl2% 几何下单笔净亏被止损截断在 −2.04%xL，"
        "只有盘中闪崩跳过止损才可能一步触及（8 年最深盘中回落 −14.2%）。",
        "",
        "== 诚实边界 ==",
        "1) n=54 的 bootstrap 只反映已见分布——它回答『这批交易换个顺序/换批抽样"
        "会怎样』，不回答『策略在未见 regime 下会怎样』；后者仅由污染注入近似，"
        f"且 {P_CONTAM:.0%} 污染概率是假设不是估计。",
        "2) 污染路径沿用 294 天时基做年化——把崩盘片段『塞进』同样长度的一年，"
        "是悲观化处理而非历史重演。",
        "3) MTM 谷底用逐笔 MAE（5m 收盘 vs 入场成交价）随笔携带，重排后仍是该笔"
        "自己的盘中最深；bar 内更深的瞬时低点未捕捉。",
        "4) 融资成本按实际持仓秒数计，54 笔合计 <4bp，与 e17_ab 结论一致可忽略。",
        "5) 全部指标一次生成，无任何参数搜索；多重比较记账 +0（风险定价，非假设"
        "检验）。",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
