#!/usr/bin/env python
"""E10 — TSLA 日线波段频率探索（docs/strategy-lab.md E10，协议先于结果写死）.

假设：5m 日内尺度 ~1380 组合已证明 edge/成本比太薄（最好 +1bp 信息 vs 6bp 成本）；
波段尺度持有数天、单笔波幅数个百分点，同样的 6bp 单边成本占比骤降，弱信息或可变现。

数据口径（如实声明）：
- data/TSLA_1h_alpaca.csv：8 年 RTH 小时线，时钟对齐（每日 6 根，bar-start
  10:00..15:00 ET，覆盖 10:00-16:00）。源数据不含 9:30-10:00 首半小时，
  也不含盘前盘后。日线由按 ET 交易日 groupby OHLCV 聚合而成——
  因此"日开盘"= 10:00 ET 开盘价（非 9:30），"日收盘"= 16:00 收盘。
  这是 RTH（且缺首半小时）口径，与含盘前后的日线不同；隔夜跳空保留在
  日与日之间，gap-through 止损按次日开盘悲观结算。

策略家族（全部多空双向）：
  1. donchian  N 日通道突破，N ∈ {10,20,55}（收盘破前 N 日高做多 / 破前 N 日低做空；
     short 变体 = 镜像方向反着做，即突破做反向——不，见下：双向 = 突破顺势多 + 突破顺势空
     各自独立成组合；另设 fade 变体：突破做反向）
  2. ma_pull   均线回踩：Close 在 MA{20,50} 上方且 0 < Close/MA-1 < 1% 买入；
     镜像空头：Close 在 MA 下方且 -1% < Close/MA-1 < 0 卖空
  3. streak    连跌 k ∈ {3,4,5} 日后买入；镜像：连涨 k 日后卖空
  4. bigdown   单日跌幅 > {3%,5%} 后次日买入；镜像：单日涨幅 > 同阈值后卖空

出场网格：bracket tp{3,5,8%} x sl{2,3,5%} x maxhold{5,10,20}日 = 27，
另 trailing {3%,5%} x maxhold{5,10,20} = 6，共 33 种出场。
入场 = 信号次日开盘（entry_fill 逆向滑点），bracket 用 settle_bracket 在日线
bar 上悲观结算（gap-through SL 按开盘、同日双触 SL 优先），成本 1bp fee +
2bp slip 单边。一次一仓。

协议：
- 训练 2018-07-01..2024-06-30（按入场日）/ 验证 2024-07-01..2026-07-31
- min_trades：训练 30 / 验证 15（波段频率低于日内，门槛相应放宽，预先声明）
- score = 单笔净期望(bp) 为主，附 CAGR / MTM MDD / 胜率（分列，防"胜率单独达标"伪装）
- 训练段 top10 → 验证段；验证幸存者(期望>0 且 n>=15) 做 ±10% 邻域扰动 +
  bootstrap p 值(2000) + 随机入场同几何基线(200 次)
- 崩盘窗口 2021-11-01..2023-01-31 单独报幸存者表现（无幸存者则报 top10）

Usage:  .venv/bin/python research/e10_daily_swing.py
Outputs: outputs/e10_daily_swing/{grid_train.csv, valid_results.csv,
         survivors.csv, crash_window.csv, summary.txt}
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402
from src.common.execution import CostModel, entry_fill, settle_bracket  # noqa: E402

BARS_1H = ROOT / "data" / "TSLA_1h_alpaca.csv"
OUT = ROOT / "outputs" / "e10_daily_swing"

COST = CostModel(fee_bp=1.0, slippage_bp=2.0)

TRAIN_LO, TRAIN_HI = pd.Timestamp("2018-07-01"), pd.Timestamp("2024-06-30")
VALID_LO, VALID_HI = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-07-31")
CRASH_LO, CRASH_HI = pd.Timestamp("2021-11-01"), pd.Timestamp("2023-01-31")

MIN_TRADES_TRAIN = 30
MIN_TRADES_VALID = 15
TOP_N = 10
N_BOOT = 2000
N_RANDOM = 200
RNG_SEED = 20260723


# ------------------------------------------------------------------ daily bars
def build_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """RTH 小时线按 ET 交易日聚合成日线（口径见模块 docstring）。"""
    et_date = df_1h.index.tz_convert(ET).date
    g = df_1h.groupby(et_date)
    daily = pd.DataFrame(
        {
            "Open": g["Open"].first(),
            "High": g["High"].max(),
            "Low": g["Low"].min(),
            "Close": g["Close"].last(),
            "Volume": g["Volume"].sum(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"
    return daily.sort_index()


# ------------------------------------------------------------------- signals
@dataclass(frozen=True)
class Entry:
    family: str            # donchian | ma_pull | streak | bigdown
    direction: int         # +1 long / -1 short（含义随 family 见 signal()）
    p1: float              # N / MA len / k / drop-threshold
    p2: float = 0.0        # ma_pull 的回踩带宽（比例）

    def label(self) -> str:
        if self.family == "donchian":
            return f"don{int(self.p1)}_{'L' if self.direction == 1 else 'S'}"
        if self.family == "ma_pull":
            return f"ma{int(self.p1)}b{self.p2:.3f}_{'L' if self.direction == 1 else 'S'}"
        if self.family == "streak":
            return f"stk{int(self.p1)}_{'L' if self.direction == 1 else 'S'}"
        return f"big{self.p1:.3f}_{'L' if self.direction == 1 else 'S'}"


def signal(entry: Entry, d: pd.DataFrame) -> pd.Series:
    """信号在 t 日收盘判定，入场 t+1 开盘。返回 bool Series（与 d 同 index）。"""
    c, h, lo = d["Close"], d["High"], d["Low"]
    n = int(round(entry.p1))
    if entry.family == "donchian":
        if entry.direction == 1:   # 收盘上破前 N 日最高 -> 做多
            return c > h.shift(1).rolling(n).max()
        else:                      # 收盘下破前 N 日最低 -> 做空
            return c < lo.shift(1).rolling(n).min()
    if entry.family == "ma_pull":
        ma = c.rolling(n).mean()
        rel = c / ma - 1
        band = entry.p2
        if entry.direction == 1:   # 站上均线且回踩到 1% 带内 -> 做多
            return (rel > 0) & (rel < band)
        else:                      # 跌破均线且反抽到 1% 带内 -> 做空
            return (rel < 0) & (rel > -band)
    if entry.family == "streak":
        down = (c < c.shift(1)).astype(int)
        up = (c > c.shift(1)).astype(int)
        if entry.direction == 1:   # 连跌 k 日 -> 买反转
            return down.rolling(n).sum() == n
        else:                      # 连涨 k 日 -> 卖反转
            return up.rolling(n).sum() == n
    if entry.family == "bigdown":
        r = c.pct_change()
        thr = entry.p1
        if entry.direction == 1:   # 大阴线次日买
            return r < -thr
        else:                      # 大阳线次日卖空（镜像）
            return r > thr
    raise ValueError(entry.family)


def entry_grid() -> list[Entry]:
    es: list[Entry] = []
    for n in (10, 20, 55):
        for di in (1, -1):
            es.append(Entry("donchian", di, n))
    for m in (20, 50):
        for di in (1, -1):
            es.append(Entry("ma_pull", di, m, 0.01))
    for k in (3, 4, 5):
        for di in (1, -1):
            es.append(Entry("streak", di, k))
    for thr in (0.03, 0.05):
        for di in (1, -1):
            es.append(Entry("bigdown", di, thr))
    return es


# --------------------------------------------------------------------- exits
@dataclass(frozen=True)
class Exit:
    kind: str              # bracket | trail
    tp: float = 0.0
    sl: float = 0.0
    trail: float = 0.0
    maxhold: int = 10

    def label(self) -> str:
        if self.kind == "bracket":
            return f"tp{self.tp:.2f}sl{self.sl:.2f}h{self.maxhold}"
        return f"tr{self.trail:.2f}h{self.maxhold}"


def exit_grid() -> list[Exit]:
    xs: list[Exit] = []
    for tp in (0.03, 0.05, 0.08):
        for sl in (0.02, 0.03, 0.05):
            for mh in (5, 10, 20):
                xs.append(Exit("bracket", tp=tp, sl=sl, maxhold=mh))
    for tr in (0.03, 0.05):
        for mh in (5, 10, 20):
            xs.append(Exit("trail", trail=tr, maxhold=mh))
    return xs


def settle_trailing(window: pd.DataFrame, direction: int, entry_px: float,
                    trail: float, cost: CostModel):
    """日线 trailing stop，悲观：当日先用「昨日为止的峰值」定的 stop 检查
    （开盘穿越按开盘成交 + 逆向滑点），再用当日 High/Low 更新峰值。"""
    stop = entry_px * (1 - direction * trail)
    peak = entry_px
    hit, exit_px, exit_time = "close", float(window["Close"].iloc[-1]), window.index[-1]
    for ts, row in window.iterrows():
        o, h, l = float(row["Open"]), float(row["High"]), float(row["Low"])
        if direction == 1:
            if o <= stop:
                return "sl", o * (1 - cost.slip), ts
            if l <= stop:
                return "sl", stop * (1 - cost.slip), ts
            peak = max(peak, h)
            stop = max(stop, peak * (1 - trail))
        else:
            if o >= stop:
                return "sl", o * (1 + cost.slip), ts
            if h >= stop:
                return "sl", stop * (1 + cost.slip), ts
            peak = min(peak, l)
            stop = min(stop, peak * (1 + trail))
    return hit, exit_px, exit_time


# ---------------------------------------------------------------- simulation
def simulate(daily: pd.DataFrame, sig: pd.Series, direction: int,
             ex: Exit) -> pd.DataFrame:
    """一次一仓：信号日 t -> t+1 开盘入场，日线 bar 悲观结算。
    返回 trades DataFrame（entry_date/exit_date/ret/hit/hold_days/entry_i/exit_i）。"""
    o = daily["Open"].to_numpy()
    n = len(daily)
    sigv = sig.to_numpy()
    trades = []
    i = 0
    while i < n - 1:
        if not sigv[i]:
            i += 1
            continue
        e_i = i + 1
        entry_px = entry_fill(o[e_i], direction, COST)
        w_end = min(e_i + ex.maxhold, n)
        window = daily.iloc[e_i:w_end]
        if ex.kind == "bracket":
            tp_px = entry_px * (1 + direction * ex.tp)
            sl_px = entry_px * (1 - direction * ex.sl)
            res = settle_bracket(window, direction, entry_px, tp_px, sl_px, COST)
            hit, exit_time, ret = res.hit, res.exit_time, res.ret
        else:
            hit, exit_px, exit_time = settle_trailing(window, direction, entry_px,
                                                      ex.trail, COST)
            ret = direction * (exit_px - entry_px) / entry_px - 2 * COST.fee
        x_i = daily.index.get_loc(exit_time)
        trades.append(
            dict(entry_date=daily.index[e_i], exit_date=exit_time, ret=ret,
                 hit=hit, hold_days=x_i - e_i + 1, entry_i=e_i, exit_i=x_i,
                 entry_px=entry_px)
        )
        i = x_i + 1  # 出场日之后才能再看信号（出场当日不重复入场）
    return pd.DataFrame(trades)


def mtm_curve(daily: pd.DataFrame, trades: pd.DataFrame, direction: int,
              lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    """段内逐日 MTM 权益（一次一仓、全额投入；持仓期按收盘价 mark，
    出场日记入含成本的已实现收益；空仓期延续最近已实现权益）。
    传入的 trades 须已按段过滤（入场日在段内），段首权益 = 1.0。"""
    idx = daily.index[(daily.index >= lo) & (daily.index <= hi)]
    marks = pd.Series(np.nan, index=daily.index)
    close = daily["Close"]
    cur = 1.0
    for t in trades.itertuples():
        seg = daily.index[t.entry_i: t.exit_i + 1]
        marks.loc[seg] = cur * (1 + direction * (close.loc[seg] / t.entry_px - 1))
        cur *= 1 + t.ret
        marks.loc[t.exit_date] = cur
    eq = marks.ffill().fillna(1.0)
    return eq.loc[idx]


def seg_metrics(daily: pd.DataFrame, trades: pd.DataFrame, direction: int,
                lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    tr = trades[(trades["entry_date"] >= lo) & (trades["entry_date"] <= hi)] \
        if not trades.empty else trades
    n = len(tr)
    out = dict(n=n, exp_bp=np.nan, wr=np.nan, cagr=np.nan, mdd=np.nan,
               med_hold=np.nan, max_hold=np.nan)
    if n == 0:
        return out
    r = tr["ret"].to_numpy()
    out["exp_bp"] = float(r.mean() * 1e4)
    out["wr"] = float((r > 0).mean())
    out["med_hold"] = float(tr["hold_days"].median())
    out["max_hold"] = int(tr["hold_days"].max())
    eq = mtm_curve(daily, tr.reset_index(drop=True), direction, lo, hi)
    if len(eq) > 1:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        if years > 0 and eq.iloc[0] > 0:
            out["cagr"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
        out["mdd"] = float((eq / eq.cummax() - 1).min())
    return out


# ------------------------------------------------------------- perturbation
def perturb_variants(en: Entry, ex: Exit) -> list[tuple[Entry, Exit]]:
    """±10% 一次动一个参数（整数参数至少 ±1）。"""
    vs: list[tuple[Entry, Exit]] = []

    def int_pm(v: int) -> list[int]:
        d = max(1, round(abs(v) * 0.1))
        return [v - d, v + d]

    if en.family in ("donchian", "ma_pull", "streak"):
        for v in int_pm(int(en.p1)):
            if v >= 2:
                vs.append((replace(en, p1=v), ex))
    else:
        for m in (0.9, 1.1):
            vs.append((replace(en, p1=en.p1 * m), ex))
    if en.family == "ma_pull":
        for m in (0.9, 1.1):
            vs.append((replace(en, p2=en.p2 * m), ex))

    if ex.kind == "bracket":
        for m in (0.9, 1.1):
            vs.append((en, replace(ex, tp=ex.tp * m)))
            vs.append((en, replace(ex, sl=ex.sl * m)))
    else:
        for m in (0.9, 1.1):
            vs.append((en, replace(ex, trail=ex.trail * m)))
    for v in int_pm(ex.maxhold):
        if v >= 1:
            vs.append((en, replace(ex, maxhold=v)))
    return vs


# --------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    df_1h = load_bars(str(BARS_1H))
    daily = build_daily(df_1h)
    print(f"daily bars: {len(daily)}  {daily.index[0].date()} .. {daily.index[-1].date()}")

    entries = entry_grid()
    exits = exit_grid()
    combos = [(en, ex) for en in entries for ex in exits]
    print(f"grid: {len(entries)} entries x {len(exits)} exits = {len(combos)} combos")

    sig_cache = {en: signal(en, daily).fillna(False) for en in entries}

    def run_combo(en: Entry, ex: Exit) -> pd.DataFrame:
        return simulate(daily, sig_cache.get(en) if en in sig_cache
                        else signal(en, daily).fillna(False), en.direction, ex)

    rows = []
    trades_cache: dict[tuple[Entry, Exit], pd.DataFrame] = {}
    for ci, (en, ex) in enumerate(combos):
        tr = run_combo(en, ex)
        trades_cache[(en, ex)] = tr
        m_tr = seg_metrics(daily, tr, en.direction, TRAIN_LO, TRAIN_HI)
        rows.append(
            dict(combo=ci, entry=en.label(), family=en.family,
                 direction=en.direction, exit=ex.label(),
                 n_train=m_tr["n"], exp_bp_train=m_tr["exp_bp"],
                 wr_train=m_tr["wr"], cagr_train=m_tr["cagr"],
                 mdd_train=m_tr["mdd"], med_hold_train=m_tr["med_hold"]))
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{len(combos)}  ({time.time() - t0:.0f}s)")
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid_train.csv", index=False)

    ok = grid[grid["n_train"] >= MIN_TRADES_TRAIN].copy()
    top = ok.sort_values("exp_bp_train", ascending=False).head(TOP_N)
    print(f"\ntrain: {len(ok)}/{len(grid)} combos with n>={MIN_TRADES_TRAIN}; top{TOP_N}:")
    print(top[["combo", "entry", "exit", "n_train", "exp_bp_train", "wr_train",
               "cagr_train", "mdd_train"]].to_string(index=False))

    # ---- validation of top10
    vrows = []
    for _, row in top.iterrows():
        en, ex = combos[int(row["combo"])]
        tr = trades_cache[(en, ex)]
        m_v = seg_metrics(daily, tr, en.direction, VALID_LO, VALID_HI)
        vrows.append(dict(combo=int(row["combo"]), entry=en.label(), exit=ex.label(),
                          n_train=row["n_train"], exp_bp_train=row["exp_bp_train"],
                          wr_train=row["wr_train"],
                          n_valid=m_v["n"], exp_bp_valid=m_v["exp_bp"],
                          wr_valid=m_v["wr"], cagr_valid=m_v["cagr"],
                          mdd_valid=m_v["mdd"], med_hold_valid=m_v["med_hold"],
                          max_hold_valid=m_v["max_hold"]))
    valid = pd.DataFrame(vrows)
    valid.to_csv(OUT / "valid_results.csv", index=False)
    print("\nvalidation of top10:")
    print(valid.to_string(index=False))

    # ---- survivors: valid n>=15 且 期望>0
    surv = valid[(valid["n_valid"] >= MIN_TRADES_VALID)
                 & (valid["exp_bp_valid"] > 0)].copy()
    print(f"\nsurvivors (n_valid>={MIN_TRADES_VALID} & exp>0): {len(surv)}")

    srows = []
    valid_days = daily.index[(daily.index >= VALID_LO) & (daily.index <= VALID_HI)]
    valid_pos = [daily.index.get_loc(d) for d in valid_days]
    for _, row in surv.iterrows():
        en, ex = combos[int(row["combo"])]
        tr_all = trades_cache[(en, ex)]
        tr = tr_all[(tr_all["entry_date"] >= VALID_LO)
                    & (tr_all["entry_date"] <= VALID_HI)]
        r = tr["ret"].to_numpy()
        # bootstrap p (H0: mean<=0)
        boots = rng.choice(r, size=(N_BOOT, len(r)), replace=True).mean(axis=1)
        p_boot = float((boots <= 0).mean())
        # ±10% 邻域扰动（验证段期望保留率）
        base = row["exp_bp_valid"]
        kept = 0
        variants = perturb_variants(en, ex)
        v_exps = []
        for ven, vex in variants:
            vtr = simulate(daily, signal(ven, daily).fillna(False),
                           ven.direction, vex)
            m = seg_metrics(daily, vtr, ven.direction, VALID_LO, VALID_HI)
            v_exps.append(m["exp_bp"])
            if not np.isnan(m["exp_bp"]) and m["exp_bp"] >= 0.5 * base:
                kept += 1
        retention = kept / len(variants) if variants else np.nan
        # 随机入场同几何基线：同笔数、同方向、同出场，200 次
        n_tr = len(tr)
        rand_means = []
        # 可行入场位置：入场 bar 在验证段内且不是最后一根
        feas = [p for p in valid_pos if p >= 1 and p < len(daily)]
        for _ in range(N_RANDOM):
            picks = rng.choice(feas, size=min(n_tr, len(feas)), replace=False)
            rets = []
            for e_i in sorted(picks):
                entry_px = entry_fill(float(daily["Open"].iloc[e_i]),
                                      en.direction, COST)
                w = daily.iloc[e_i:min(e_i + ex.maxhold, len(daily))]
                if ex.kind == "bracket":
                    res = settle_bracket(
                        w, en.direction, entry_px,
                        entry_px * (1 + en.direction * ex.tp),
                        entry_px * (1 - en.direction * ex.sl), COST)
                    rets.append(res.ret)
                else:
                    _, xp, _ = settle_trailing(w, en.direction, entry_px,
                                               ex.trail, COST)
                    rets.append(en.direction * (xp - entry_px) / entry_px
                                - 2 * COST.fee)
            rand_means.append(np.mean(rets))
        rand_means = np.asarray(rand_means)
        p_rand = float((rand_means >= r.mean()).mean())
        srows.append(dict(
            combo=int(row["combo"]), entry=en.label(), exit=ex.label(),
            n_valid=int(row["n_valid"]), exp_bp_valid=base,
            wr_valid=row["wr_valid"], cagr_valid=row["cagr_valid"],
            mdd_valid=row["mdd_valid"],
            perturb_n=len(variants), perturb_retention=retention,
            perturb_median_exp_bp=float(np.nanmedian(v_exps)) if v_exps else np.nan,
            p_bootstrap=p_boot,
            rand_mean_bp=float(rand_means.mean() * 1e4),
            p_vs_random=p_rand))
    survivors = pd.DataFrame(srows)
    survivors.to_csv(OUT / "survivors.csv", index=False)
    if len(survivors):
        print(survivors.to_string(index=False))

    # ---- crash window（幸存者；若无幸存者报 top10）
    crash_set = surv if len(surv) else valid
    crows = []
    for _, row in crash_set.iterrows():
        en, ex = combos[int(row["combo"])]
        tr = trades_cache[(en, ex)]
        m_c = seg_metrics(daily, tr, en.direction, CRASH_LO, CRASH_HI)
        crows.append(dict(combo=int(row["combo"]), entry=en.label(),
                          exit=ex.label(), is_survivor=int(row["combo"] in
                          set(surv["combo"])) if len(surv) else 0,
                          n_crash=m_c["n"], exp_bp_crash=m_c["exp_bp"],
                          wr_crash=m_c["wr"], cagr_crash=m_c["cagr"],
                          mdd_crash=m_c["mdd"]))
    crash = pd.DataFrame(crows)
    crash.to_csv(OUT / "crash_window.csv", index=False)
    print("\ncrash window (2021-11..2023-01):")
    print(crash.to_string(index=False))

    # ---- 70% 胜率联动条款检查（全网格，训练段 + top10 验证段）
    wr70_train = grid[(grid["n_train"] >= MIN_TRADES_TRAIN)
                      & (grid["wr_train"] >= 0.70)]
    wr70_pos = wr70_train[wr70_train["exp_bp_train"] > 0]
    wr70_valid = valid[(valid["wr_valid"] >= 0.70) & (valid["exp_bp_valid"] > 0)] \
        if len(valid) else valid

    # ---- summary
    bh_lo = daily["Close"].loc[daily.index >= VALID_LO]
    bh_cagr = ((bh_lo.iloc[-1] / bh_lo.iloc[0])
               ** (365.25 / max((bh_lo.index[-1] - bh_lo.index[0]).days, 1)) - 1)
    lines = []
    lines.append("E10 — TSLA 日线波段频率探索（2026-07-23）")
    lines.append("=" * 72)
    lines.append(f"数据口径：Alpaca 1h RTH（时钟对齐，10:00-16:00 ET，缺 9:30-10:00 首半小时，"
                 f"无盘前后）按 ET 交易日聚合日线；{len(daily)} 个交易日 "
                 f"{daily.index[0].date()}..{daily.index[-1].date()}。"
                 "日开盘=10:00 ET 开盘价。隔夜跳空保留，gap-through SL 按开盘悲观结算。")
    lines.append(f"网格：{len(entries)} 入场(4 家族全多空双向) x {len(exits)} 出场"
                 f"(27 bracket + 6 trailing) = {len(combos)} 组合。"
                 "成本 1bp fee + 2bp slip 单边（往返 6bp）。一次一仓，入场次日开盘。")
    lines.append(f"协议：训练 {TRAIN_LO.date()}..{TRAIN_HI.date()} / "
                 f"验证 {VALID_LO.date()}..{VALID_HI.date()}；"
                 f"min_trades 训练 {MIN_TRADES_TRAIN} / 验证 {MIN_TRADES_VALID}"
                 "（波段频率低于日内，门槛预先放宽并在此声明；注意达标线原文要求"
                 "验证段>=30 笔，放宽后即使其余全过也只能算「有条件通过」）。")
    lines.append("")
    lines.append(f"[训练段] n>={MIN_TRADES_TRAIN} 的组合 {len(ok)}/{len(grid)}；top{TOP_N}（按单笔净期望 bp）：")
    for _, r_ in top.iterrows():
        lines.append(f"  #{int(r_['combo']):3d} {r_['entry']:16s} {r_['exit']:18s} "
                     f"n={int(r_['n_train']):3d} exp={r_['exp_bp_train']:+7.1f}bp "
                     f"WR={r_['wr_train']:.1%} CAGR={r_['cagr_train']:+.1%} "
                     f"MDD={r_['mdd_train']:.1%}")
    lines.append("")
    lines.append("[验证段] top10 前向表现：")
    for _, r_ in valid.iterrows():
        lines.append(f"  #{int(r_['combo']):3d} {r_['entry']:16s} {r_['exit']:18s} "
                     f"n={int(r_['n_valid']):3d} exp={r_['exp_bp_valid']:+7.1f}bp "
                     f"WR={r_['wr_valid']:.1%} CAGR={r_['cagr_valid']:+.1%} "
                     f"MDD={r_['mdd_valid']:.1%}")
    lines.append(f"  （对照：验证段 B&H CAGR = {bh_cagr:+.1%}）")
    lines.append("")
    lines.append(f"[幸存者] 验证段 n>={MIN_TRADES_VALID} 且期望>0：{len(survivors)} 个")
    for _, r_ in survivors.iterrows():
        lines.append(f"  #{int(r_['combo']):3d} {r_['entry']} {r_['exit']}: "
                     f"exp={r_['exp_bp_valid']:+.1f}bp WR={r_['wr_valid']:.1%} "
                     f"扰动保留 {r_['perturb_retention']:.0%} "
                     f"(邻域中位 {r_['perturb_median_exp_bp']:+.1f}bp) "
                     f"bootstrap p={r_['p_bootstrap']:.3f} "
                     f"随机基线均值 {r_['rand_mean_bp']:+.1f}bp "
                     f"p_vs_random={r_['p_vs_random']:.3f}")
    lines.append("")
    lines.append("[崩盘窗口 2021-11..2023-01]" + ("（幸存者）" if len(surv) else "（无幸存者，报 top10）"))
    for _, r_ in crash.iterrows():
        if r_["n_crash"] and not np.isnan(r_["exp_bp_crash"]):
            lines.append(f"  #{int(r_['combo']):3d} {r_['entry']} {r_['exit']}: "
                         f"n={int(r_['n_crash'])} exp={r_['exp_bp_crash']:+.1f}bp "
                         f"WR={r_['wr_crash']:.1%} CAGR={r_['cagr_crash']:+.1%} "
                         f"MDD={r_['mdd_crash']:.1%}")
        else:
            lines.append(f"  #{int(r_['combo']):3d} {r_['entry']} {r_['exit']}: 窗口内 0 笔")
    lines.append("")
    lines.append("[70% 胜率联动条款]（达标线：WR>=70% 仅在单笔期望>=+5bp 同时成立时有效）")
    lines.append(f"  训练段 WR>=70% 且 n>=30 的组合：{len(wr70_train)} 个；"
                 f"其中期望>0：{len(wr70_pos)} 个；"
                 f"其中期望>=+5bp：{len(wr70_train[wr70_train['exp_bp_train'] >= 5])} 个")
    if len(wr70_valid):
        lines.append("  验证段（top10 内）WR>=70% 且期望>0：")
        for _, r_ in wr70_valid.iterrows():
            lines.append(f"    #{int(r_['combo'])} {r_['entry']} {r_['exit']}: "
                         f"WR={r_['wr_valid']:.1%} exp={r_['exp_bp_valid']:+.1f}bp")
    else:
        best_wr = valid.sort_values("wr_valid", ascending=False).head(3) if len(valid) else valid
        lines.append("  验证段（top10 内）无 WR>=70% 且期望>0 的组合；最接近：")
        for _, r_ in best_wr.iterrows():
            lines.append(f"    #{int(r_['combo'])} {r_['entry']} {r_['exit']}: "
                         f"WR={r_['wr_valid']:.1%} exp={r_['exp_bp_valid']:+.1f}bp")
    lines.append("")
    lines.append("（判读段由实验者据以上数字撰写，见下）")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT}/summary.txt  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
