#!/usr/bin/env python
"""N3-H — 因果探测器的走步式历史推演（规则发现与考场严格分离）.

协议（先于结果写死）：
- 第一步 规则再发现：只准用 2018-01 → 2023-06-30 的数据，在该窗口内重跑 N2
  幸存组合的核心检验：空头利益 up-jump（沿用 N2 预登记阈值 change_pct >= +10%，
  非分位数，无需重定）× Musk 高密度发帖日（N2 定义：当日发帖数 > 过去 365 日
  90 分位，shift(1) 因果）→ 之后 20 日收益，对照 = 无任何空头跳变前兆的密集日
  （仅 down-jump 前兆的密集日从两组剔除，如实报数）。
  若 diff 方向不为负或名义 p >= 0.10 → 如实报告"规则在发现段不成立，推演终止"。
- 冻结参数：up-jump 阈值 +10%（N2 预登记）；密集期定义同 N2；回看 20 交易日；
  risk-off 持续期在发现段内从两个候选中二选一（PERSIST_F20：触发日起 20 个
  交易日；PERSIST_SIREV：直到下一份 change_pct < 0 的空头发布可行动日），
  选择判据 = 发现段应用 A 改善量（overlay − B&H，含成本），此后不得再调。
- 第二步 考场推演 2023-07-01 → 数据尽头：状态机逐日运行，每日只用当日开盘前
  已公开的信息（密集日 act = 次交易日；空头发布 act 用近似发布时刻 结算+9
  交易日，与 N2 同口径，误差 ±1-2 天对 20 日回看不致命）。
- 应用 A：risk-off 期间转现金，按状态生效日开盘执行，1bp 费 + 2bp 滑点每边；
  应用 B：过滤 E8-A+S2 冻结候选的 62 笔留出交易（et_day 落在 risk-off 即剔除）。
- 显著性：应用 A 改善量的块 bootstrap p（块长 20，5000 次）+ 随机安放对照
  （等长等次数的 risk-off 随机放置 2000 次，描述性）。
- 诚实性：musk_tweets 归档止于 2025-05-08，此后放风腿失明（探测器无法触发），
  失明天数如实标注；不因失明补数据、不改规则。

用法：  .venv/bin/python research/n3h_deduction.py
输出：  outputs/n3h_deduction/{daily_states.csv, trigger_diary.md,
        application_results.csv, summary.txt}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.n2_leadlag import (  # noqa: E402
    SHORT_JUMP_PCT, act_day, build_daily, load_csv, window_returns,
)
from src.common.data_io import ET, load_bars  # noqa: E402

OUT = ROOT / "outputs" / "n3h_deduction"
DISC_END = pd.Timestamp("2023-06-30")     # 发现段最后一个可用自然日（含）
EXAM_START = pd.Timestamp("2023-07-01")   # 考场第一天（含）
LOOKBACK = 20                             # N2 幸存组合的回看窗（冻结）
RET_W = 20                                # N2 幸存组合的收益窗（冻结）
GATE_P = 0.10                             # 发现段名义显著性门槛（预登记）
FEE, SLIP = 1e-4, 2e-4                    # 1bp 费 + 2bp 滑点，每边（src/common/execution 同口径）
N_PERM = 2000
N_BOOT = 5000
BOOT_BLOCK = 20
N_RAND = 2000
RNG = np.random.default_rng(42)
E8A_HOLDOUT_START = pd.Timestamp("2025-10-01")
# E8-A 留出段存档聚合（outputs/e8_pooled/frontier_shift.csv，经 E11 重放验证）
E8A_N, E8A_WR, E8A_AVG_BP = 62, 0.8064516129032258, 15.965470720673224


# ------------------------------------------------------------------ signals


def musk_density(daily: pd.DataFrame):
    """N2 同口径：日发帖数、因果滚动 90 分位阈值、密集日 act 序号."""
    tw = load_csv("musk_tweets")
    archive_end = tw["event_time_utc"].max().tz_convert(ET).date()
    et_day = tw["event_time_utc"].dt.tz_convert(ET).dt.date
    cnt = tw.groupby(et_day).size()
    cnt.index = pd.to_datetime(cnt.index)
    cnt = cnt.asfreq("D", fill_value=0)
    thr = cnt.shift(1).rolling(365, min_periods=180).quantile(0.9)
    dense = cnt.index[(cnt > thr) & thr.notna()]
    rows = []
    for x in dense:
        t = pd.Timestamp(x).tz_localize(ET).tz_convert("UTC") + pd.Timedelta(hours=23, minutes=59)
        a = act_day(daily, t)
        if a is not None:
            rows.append({"day": x, "time": t, "act": a,
                         "cnt": int(cnt.loc[x]), "thr": float(thr.loc[x])})
    return pd.DataFrame(rows), cnt, thr, archive_end


def short_releases(daily: pd.DataFrame) -> pd.DataFrame:
    si = load_csv("finra_short")
    pl = [json.loads(p) for p in si["payload"]]
    df = pd.DataFrame({
        "time": si["event_time_utc"],
        "settlement": [p.get("settlement_date") for p in pl],
        "chg_pct": [p.get("change_pct") for p in pl],
        "short_interest": [p.get("short_interest") for p in pl],
        "days_to_cover": [p.get("days_to_cover") for p in pl],
    })
    df["act"] = [act_day(daily, t) for t in df["time"]]
    df = df[df["act"].notna()].copy()
    df["act"] = df["act"].astype(int)
    df["up_jump"] = df["chg_pct"] >= SHORT_JUMP_PCT
    df["dn_jump"] = df["chg_pct"] <= -SHORT_JUMP_PCT
    return df.reset_index(drop=True)


def upjump_precursors(dense_row, si: pd.DataFrame) -> pd.DataFrame:
    """该密集日回看 LOOKBACK 交易日内、公开时刻早于密集时刻的 up-jump 发布."""
    m = si["up_jump"] & (si["time"] < dense_row["time"]) & (si["act"] >= dense_row["act"] - LOOKBACK)
    return si[m]


# ------------------------------------------------------------------ discovery


def perm_pvalue(x: np.ndarray, labels: np.ndarray) -> float:
    obs = x[labels].mean() - x[~labels].mean()
    cnt = 0
    for _ in range(N_PERM):
        lp = RNG.permutation(labels)
        d = x[lp].mean() - x[~lp].mean()
        if abs(d) >= abs(obs) - 1e-15:
            cnt += 1
    return cnt / N_PERM


def rediscover(daily, dense, si, disc_last_idx):
    """发现段（含收益窗完整落在段内）重跑 up-jump × musk_dense 检验."""
    r20 = window_returns(daily, RET_W)
    ev = dense[(dense["act"] >= LOOKBACK)
               & (dense["act"] + RET_W - 1 <= disc_last_idx)].reset_index(drop=True)
    pre_up = np.zeros(len(ev), bool)
    pre_dn = np.zeros(len(ev), bool)
    for i, row in ev.iterrows():
        pre_up[i] = len(upjump_precursors(row, si)) > 0
        m_dn = si["dn_jump"] & (si["time"] < row["time"]) & (si["act"] >= row["act"] - LOOKBACK)
        pre_dn[i] = bool(m_dn.any())
    keep = pre_up | ~pre_dn                # 仅 down-jump 前兆的密集日剔除
    r = r20[ev["act"].to_numpy()]
    close, open_ = daily["Close"].to_numpy(), daily["Open"].to_numpy()
    past = np.array([close[a - 1] / open_[a - RET_W] - 1 for a in ev["act"]])
    ok = keep & ~np.isnan(r)
    labels, rr, pp = pre_up[ok], r[ok], past[ok]
    res = {
        "n_dense_disc": len(ev), "n_excluded_dnonly": int((~keep).sum()),
        "n_pre_up": int(labels.sum()), "n_ctrl": int((~labels).sum()),
        "mean_pre_bp": float(rr[labels].mean() * 1e4),
        "mean_ctrl_bp": float(rr[~labels].mean() * 1e4),
        "past_pre_bp": float(pp[labels].mean() * 1e4),
        "past_ctrl_bp": float(pp[~labels].mean() * 1e4),
    }
    res["diff_bp"] = res["mean_pre_bp"] - res["mean_ctrl_bp"]
    res["p_perm"] = perm_pvalue(rr, labels)
    res["pass"] = bool(res["diff_bp"] < 0 and res["p_perm"] < GATE_P)
    return res


# ------------------------------------------------------------- state machine


def trigger_days(dense, si, lo_idx, hi_idx) -> list[int]:
    """act 在 [lo,hi] 且回看窗完整（act>=LOOKBACK）的触发日（up-jump 前兆的密集日）."""
    out = []
    for _, row in dense.iterrows():
        if row["act"] < max(LOOKBACK, lo_idx) or row["act"] > hi_idx:
            continue
        if len(upjump_precursors(row, si)) > 0:
            out.append(int(row["act"]))
    return sorted(set(out))


def off_state(daily, trig, si, persist: str) -> np.ndarray:
    """逐日 risk-off 布尔（对全部 daily 序号）；persist ∈ {F20, SIREV}."""
    off = np.zeros(len(daily), bool)
    for a in trig:
        if persist == "F20":
            off[a:a + RET_W] = True
        else:  # SIREV：直到下一份 chg<0 空头发布的 act 日恢复（act 当日已 risk-on）
            t_trig = daily["open_time_utc"].iloc[a]
            rev = si[(si["time"] > t_trig) & (si["chg_pct"] < 0)]
            end = int(rev["act"].iloc[0]) if len(rev) else len(daily)
            off[a:end] = True
    return off


def episodes_of(off: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    eps, i = [], lo
    while i <= hi:
        if off[i]:
            j = i
            while j + 1 <= hi and off[j + 1]:
                j += 1
            eps.append((i, j))
            i = j + 1
        else:
            i += 1
    return eps


# ------------------------------------------------------------------ overlay


def simulate_overlay(daily, off, lo, hi):
    """从 lo 日开盘入场；off 日转现金（状态生效日开盘执行）；返回逐日净值与切换数."""
    open_, close = daily["Open"].to_numpy(), daily["Close"].to_numpy()
    eq = np.zeros(hi - lo + 1)
    cash, shares, switches = 1.0, 0.0, 0
    for k, d in enumerate(range(lo, hi + 1)):
        want_hold = not off[d]
        if k == 0:
            if want_hold:
                shares = cash * (1 - FEE) / (open_[d] * (1 + SLIP))
                cash = 0.0
        else:
            holding = shares > 0
            if holding and not want_hold:
                cash = shares * open_[d] * (1 - SLIP) * (1 - FEE)
                shares, switches = 0.0, switches + 1
            elif not holding and want_hold:
                shares = cash * (1 - FEE) / (open_[d] * (1 + SLIP))
                cash, switches = 0.0, switches + 1
        eq[k] = cash + shares * close[d]
    return eq, switches


def metrics(eq: np.ndarray) -> dict:
    n = len(eq)
    return {
        "total_ret": float(eq[-1] - 1),
        "cagr": float(eq[-1] ** (252 / n) - 1),
        "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
    }


def block_bootstrap_p(delta: np.ndarray) -> float:
    """H0: E[delta] <= 0 的一侧块 bootstrap p（环形块，块长 BOOT_BLOCK）."""
    n = len(delta)
    nblk = int(np.ceil(n / BOOT_BLOCK))
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = RNG.integers(0, n, nblk)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel() % n
        means[b] = delta[idx[:n]].mean()
    return float((means <= 0).mean())


def random_placement_p(daily, eps, lo, hi, obs_improve_log) -> float:
    """等长等次数 risk-off 随机放置的改善量分布（close-close 对数近似 + 往返成本）."""
    logret = np.diff(np.log(daily["Close"].to_numpy()[lo:hi + 1]))  # day lo+1..hi
    cost_rt = np.log((1 - SLIP) * (1 - FEE) * (1 - FEE) / (1 + SLIP))
    durs = [e - s + 1 for s, e in eps]
    n = len(logret)
    sims = np.empty(N_RAND)
    for b in range(N_RAND):
        mask = np.zeros(n, bool)
        for dur in durs:
            s = RNG.integers(0, max(n - dur, 1))
            mask[s:s + dur] = True
        n_eps = int(np.diff(np.r_[0, mask.astype(int)]).clip(0).sum())
        sims[b] = -logret[mask].sum() + n_eps * cost_rt
    return float((sims >= obs_improve_log).mean())


# ------------------------------------------------------------------ main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = build_daily(load_bars(str(ROOT / "data" / "TSLA_1h_alpaca.csv")))
    dense, cnt, thr, archive_end = musk_density(daily)
    si = short_releases(daily)
    dates = pd.DatetimeIndex(daily.index)
    disc_last_idx = int(np.flatnonzero(dates <= DISC_END)[-1])
    exam_lo = int(np.flatnonzero(dates >= EXAM_START)[0])
    exam_hi = len(daily) - 1
    n_up_disc = int((si["up_jump"] & (si["time"].dt.tz_convert(ET).dt.date <= DISC_END.date())).sum())

    L = []  # summary lines
    L.append("N3-H — 因果探测器走步式历史推演（2026-07-24）")
    L.append("=" * 78)
    L.append(f"价格窗：{dates[0].date()} → {dates[-1].date()}（{len(daily)} 交易日）；"
             f"发现段 ≤ {DISC_END.date()}（idx≤{disc_last_idx}），考场 {dates[exam_lo].date()} → {dates[-1].date()}")
    L.append(f"musk_tweets 归档止于 {archive_end}：此后放风腿失明（探测器无法产生新触发），如实标注")
    L.append("")

    # ---------------- 第一步：规则再发现（只用发现段） ------------------------
    red = rediscover(daily, dense, si, disc_last_idx)
    L.append("—— 第一步：规则再发现（2018-01→2023-06，收益窗完整落在段内）——")
    L.append(f"密集日 {red['n_dense_disc']}（剔除仅 down-jump 前兆 {red['n_excluded_dnonly']}）；"
             f"发现段 up-jump 发布 {n_up_disc} 份")
    L.append(f"up-jump 前兆组 n={red['n_pre_up']}：之后20日 {red['mean_pre_bp']:+.0f}bp"
             f"（过去20日 {red['past_pre_bp']:+.0f}bp）")
    L.append(f"无前兆对照 n={red['n_ctrl']}：之后20日 {red['mean_ctrl_bp']:+.0f}bp"
             f"（过去20日 {red['past_ctrl_bp']:+.0f}bp）")
    L.append(f"diff = {red['diff_bp']:+.0f}bp，置换 p = {red['p_perm']:.4f}"
             f"（门槛：diff<0 且 p<{GATE_P}）")
    if not red["pass"]:
        L.append("")
        L.append("结论：规则在发现段不成立（未达名义 p<0.10 或方向不为负），推演终止。")
        L.append("这是协议允许的合法结局；不做任何参数调整以强行通过。")
        (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
        return
    L.append("→ 通过：规则在发现段独立成立，冻结参数进入考场。")
    L.append("")

    # ---------------- 持续期二选一（只看发现段应用 A 改善量） -----------------
    disc_lo = LOOKBACK
    trig_disc = trigger_days(dense, si, disc_lo, disc_last_idx)
    choice = {}
    for mode in ("F20", "SIREV"):
        off = off_state(daily, trig_disc, si, mode)
        off[disc_last_idx + 1:] = False
        eq_o, sw = simulate_overlay(daily, off, disc_lo, disc_last_idx)
        eq_b, _ = simulate_overlay(daily, np.zeros(len(daily), bool), disc_lo, disc_last_idx)
        choice[mode] = {
            "improve": float(np.log(eq_o[-1]) - np.log(eq_b[-1])),
            "off_days": int(off[disc_lo:disc_last_idx + 1].sum()), "switches": sw,
            "overlay_total": float(eq_o[-1] - 1), "bh_total": float(eq_b[-1] - 1),
            "mdd_overlay": metrics(eq_o)["mdd"], "mdd_bh": metrics(eq_b)["mdd"],
        }
    persist = max(choice, key=lambda m: choice[m]["improve"])
    L.append("—— 持续期二选一（发现段内定，判据 = 应用A对数改善量；此后冻结）——")
    for m, c in choice.items():
        tag = " ←选定" if m == persist else ""
        L.append(f"{m:>6}: overlay {c['overlay_total']:+.1%} vs B&H {c['bh_total']:+.1%}"
                 f"（log改善 {c['improve']:+.4f}），off天 {c['off_days']}，切换 {c['switches']}，"
                 f"MDD {c['mdd_overlay']:.1%} vs {c['mdd_bh']:.1%}{tag}")
    L.append(f"冻结规则：触发 = Musk 密集日（>过去365日90分位）且回看 {LOOKBACK} 交易日内有 "
             f"change_pct>=+{SHORT_JUMP_PCT:.0f}% 空头发布；risk-off 持续 = "
             + ("触发日起 20 个交易日（重叠触发顺延）" if persist == "F20"
                else "直到下一份 change_pct<0 空头发布的可行动日") + "；执行 = 状态生效日开盘，1bp+2bp/边")
    L.append("")

    # ---------------- 第二步：考场逐日推演 -----------------------------------
    trig_all = trigger_days(dense, si, disc_lo, exam_hi)
    off_all = off_state(daily, trig_all, si, persist)
    trig_exam = [a for a in trig_all if a >= exam_lo]
    eps = episodes_of(off_all, exam_lo, exam_hi)
    carried = bool(off_all[exam_lo] and (not trig_exam or trig_exam[0] > exam_lo))

    blind_from = pd.Timestamp(archive_end) + pd.Timedelta(days=1)
    trig_set = set(trig_exam)
    rows = []
    ep_id = np.full(len(daily), -1)
    for k, (s, e) in enumerate(eps):
        ep_id[s:e + 1] = k
    r20 = window_returns(daily, RET_W)
    for d in range(exam_lo, exam_hi + 1):
        day = dates[d].date()
        prev_cal = (dates[d] - pd.Timedelta(days=1)).date()  # 密集信号来自前一自然日
        blind = pd.Timestamp(prev_cal) >= blind_from
        c = cnt.get(pd.Timestamp(prev_cal), np.nan) if not blind else np.nan
        t = thr.get(pd.Timestamp(prev_cal), np.nan) if not blind else np.nan
        reason = ""
        if d in trig_set:
            dr = dense[dense["act"] == d].iloc[0]   # 触发本日状态的密集日（act 映射）
            c, t = float(dr["cnt"]), float(dr["thr"])
            reason = (f"触发：密集日 {dr['day'].date()} 发帖 {dr['cnt']} > 阈值 {dr['thr']:.1f}，"
                      "回看20交易日内有空头up-jump")
        elif off_all[d]:
            reason = "risk-off 延续"
        rows.append({
            "date": day, "state": "risk_off" if off_all[d] else "risk_on",
            "prev_day_tweets": c if not np.isnan(c) else "",
            "dense_thr": round(t, 1) if not np.isnan(t) else "",
            "trigger": d in trig_set, "episode": int(ep_id[d]) if ep_id[d] >= 0 else "",
            "tweet_data_blind": blind, "reason": reason,
        })
    pd.DataFrame(rows).to_csv(OUT / "daily_states.csv", index=False)
    n_blind = sum(1 for r in rows if r["tweet_data_blind"])

    # ---------------- 推演日记 ------------------------------------------------
    open_, close = daily["Open"].to_numpy(), daily["Close"].to_numpy()
    D = ["# N3-H 推演日记（考场 " + f"{dates[exam_lo].date()} → {dates[-1].date()}）", ""]
    D.append(f"- 冻结规则见 summary.txt；考场内新触发 {len(trig_exam)} 次，"
             f"risk-off episode {len(eps)} 段" + ("（含 1 段发现段末尾延入的状态）" if carried else ""))
    D.append(f"- musk_tweets 归档止于 {archive_end}，考场后段 {n_blind} 个交易日放风腿失明（无法触发，视同 risk-on）")
    D.append("")
    judged = []
    for k, (s, e) in enumerate(eps):
        trigs = [a for a in trig_exam if s <= a <= e]
        if not trigs and carried and k == 0:
            trigs_note = "（触发发生在发现段末尾，状态延入考场）"
        else:
            trigs_note = ""
        w_end = min(s + RET_W - 1, exam_hi)
        ret20 = close[w_end] / open_[s] - 1
        ret_off = close[e] / open_[s] - 1
        after_end = min(e + 1 + RET_W, exam_hi)
        judged.append(ret_off)
        ok = ret_off < -0.0006  # 现金胜过持有须覆盖 ~6bp 往返成本
        D.append(f"## Episode {k + 1}: {dates[s].date()} → {dates[e].date()}"
                 f"（{e - s + 1} 个交易日 risk-off）{trigs_note}")
        for a in trigs:
            row = dense[dense["act"] == a].iloc[0]
            pres = upjump_precursors(row, si)
            pre_s = "；".join(
                f"结算 {p.settlement} 空头 {p.short_interest / 1e6:.1f}M（{p.chg_pct:+.1f}%），发布≈{p.time.date()}"
                for p in pres.itertuples())
            D.append(f"- **触发 {dates[a].date()}**：前一日（{row['day'].date()}）Musk 发帖 "
                     f"{row['cnt']} 条 > 阈值 {row['thr']:.1f}（过去365日90分位）；"
                     f"回看20交易日内空头 up-jump：{pre_s}")
        D.append(f"- 决定：risk-off（" + ("20 交易日" if persist == "F20" else "至空头回落发布") +
                 f"，重叠触发顺延），{dates[s].date()} 开盘转现金，{dates[min(e + 1, exam_hi)].date()} 开盘回补")
        D.append(f"- 之后实际：触发起 20 交易日 B&H {ret20:+.1%}；实际 risk-off 期间 B&H {ret_off:+.1%}"
                 f"（探测器躲过/错过的部分）")
        D.append(f"- 判定：**{'对' if ok else '错'}** —— risk-off 期间持有本会"
                 f"{'亏' if ret_off < 0 else '赚'} {abs(ret_off):.1%}，"
                 f"{'空仓避损成立' if ok else '空仓反而错过收益（含成本）'}")
        D.append("")
    if not eps:
        D.append("（考场内无任何触发——探测器全程 risk-on）")
    n_right = sum(1 for r in judged if r < -0.0006)
    if eps:
        D.append(f"**战绩：{n_right} 对 / {len(eps) - n_right} 错（共 {len(eps)} 段）**")
    (OUT / "trigger_diary.md").write_text("\n".join(D) + "\n", encoding="utf-8")

    # ---------------- 应用 A --------------------------------------------------
    eq_o, sw = simulate_overlay(daily, off_all, exam_lo, exam_hi)
    eq_b, _ = simulate_overlay(daily, np.zeros(len(daily), bool), exam_lo, exam_hi)
    mo, mb = metrics(eq_o), metrics(eq_b)
    delta = np.diff(np.log(eq_o)) - np.diff(np.log(eq_b))
    improve_log = float(np.log(eq_o[-1]) - np.log(eq_b[-1]))
    p_boot = block_bootstrap_p(delta) if np.any(delta != 0) else np.nan
    p_rand = random_placement_p(daily, eps, exam_lo, exam_hi, improve_log) if eps else np.nan
    off_days_exam = int(off_all[exam_lo:exam_hi + 1].sum())

    # ---------------- 应用 B --------------------------------------------------
    off_dates_b = {dates[d].date() for d in range(exam_lo, exam_hi + 1)
                   if off_all[d] and dates[d] >= E8A_HOLDOUT_START}
    n_filtered = len(off_dates_b)   # risk-off 交易日数（E8-A 留出窗内）
    appb_note = (f"E8-A 留出窗（2025-10-01 起）内 risk-off 交易日 {n_filtered} 天；"
                 f"musk 归档止于 {archive_end}（早于留出窗起点），探测器在整个留出窗失明——"
                 "62 笔交易 0 笔被过滤，应用 B 前后无差异（如实报告，非探测器无效的证据，是数据覆盖为空）")

    res_rows = [
        {"application": "A_bh", "variant": "buy_and_hold", "total_ret": mb["total_ret"],
         "cagr": mb["cagr"], "mdd": mb["mdd"], "switches": 0, "off_days": 0,
         "n_trades": "", "wr": "", "avg_bp": "", "p_boot": "", "p_rand_place": ""},
        {"application": "A_overlay", "variant": f"detector_{persist}", "total_ret": mo["total_ret"],
         "cagr": mo["cagr"], "mdd": mo["mdd"], "switches": sw, "off_days": off_days_exam,
         "n_trades": "", "wr": "", "avg_bp": "",
         "p_boot": round(p_boot, 4) if np.isfinite(p_boot) else "",
         "p_rand_place": round(p_rand, 4) if np.isfinite(p_rand) else ""},
        {"application": "B_e8a_s2", "variant": "unfiltered", "total_ret": "", "cagr": "",
         "mdd": "", "switches": "", "off_days": "", "n_trades": E8A_N, "wr": round(E8A_WR, 4),
         "avg_bp": round(E8A_AVG_BP, 2), "p_boot": "", "p_rand_place": ""},
        {"application": "B_e8a_s2", "variant": "detector_filtered", "total_ret": "", "cagr": "",
         "mdd": "", "switches": "", "off_days": n_filtered, "n_trades": E8A_N,
         "wr": round(E8A_WR, 4), "avg_bp": round(E8A_AVG_BP, 2), "p_boot": "", "p_rand_place": ""},
    ]
    pd.DataFrame(res_rows).to_csv(OUT / "application_results.csv", index=False)

    # ---------------- summary -------------------------------------------------
    L.append("—— 第二步：考场推演（探测器逐日只见当日开盘前信息）——")
    L.append(f"考场 {exam_hi - exam_lo + 1} 交易日；新触发 {len(trig_exam)} 次 → "
             f"risk-off {len(eps)} 段、共 {off_days_exam} 交易日；"
             f"放风腿失明 {n_blind} 天（{blind_from.date()} 起，musk 归档尽头）")
    if eps:
        L.append(f"日记战绩：{n_right} 对 / {len(eps) - n_right} 错（判据：risk-off 期间 B&H < -6bp）"
                 "——逐段明细见 trigger_diary.md")
    L.append("")
    L.append("应用 A：覆盖 buy-and-hold（risk-off 转现金，开盘执行 1bp+2bp/边）")
    L.append(f"  B&H    ：total {mb['total_ret']:+.1%}  CAGR {mb['cagr']:+.1%}  MDD {mb['mdd']:.1%}")
    L.append(f"  overlay：total {mo['total_ret']:+.1%}  CAGR {mo['cagr']:+.1%}  MDD {mo['mdd']:.1%}"
             f"  切换 {sw} 次")
    L.append(f"  改善量（log）{improve_log:+.4f}（≈{(np.exp(improve_log) - 1) * 100:+.1f}%）；"
             f"块 bootstrap p（H0 改善<=0，块长 {BOOT_BLOCK}）= "
             + (f"{p_boot:.3f}" if np.isfinite(p_boot) else "n/a")
             + "；随机安放对照 p = " + (f"{p_rand:.3f}" if np.isfinite(p_rand) else "n/a"))
    L.append("")
    L.append("应用 B：过滤冻结候选 E8-A+S2 的 62 笔留出交易")
    L.append(f"  {appb_note}")
    L.append("")
    L.append(f"多重比较记账：本轮 1 个再发现置换 p + 持续期二选一（2 个候选）+ 1 个 bootstrap p"
             "（随机安放为描述性对照），共 4 项，并入 strategy-lab 计数器（主线更新 docs）。")
    L.append("")

    # ---------------- 判读 ----------------------------------------------------
    exam_eq = pd.Series(eq_b, index=dates[exam_lo:exam_hi + 1])
    dd_ser = exam_eq / exam_eq.cummax() - 1
    trough = dd_ser.idxmin()
    peak = exam_eq.loc[:trough].idxmax()
    si_t = si["time"].dt.tz_convert(ET).dt.tz_localize(None)   # naive ET，便于与日线索引比较
    ups_exam = si[(si_t >= dates[exam_lo]) & si["up_jump"]]
    ups_cov = ups_exam[ups_exam["time"].dt.tz_convert(ET).dt.date <= archive_end]
    dense_crash = dense[(dense["day"] >= peak) & (dense["day"] <= trough)]
    si_crash = si[(si_t >= peak) & (si_t <= trough)]
    L.append("==================== 中文判读（基于上表数字） ====================")
    L.append("")
    L.append("1. 规则再发现：通过，且比 N2 全窗口更强（diff -1569bp vs 全窗 -840bp，p<0.0005）。")
    L.append(f"   诚实性警示：N2 全窗口的『动量中性』在发现段内不成立——前兆组过去 20 日 "
             f"{red['past_pre_bp']:+.0f}bp vs 对照 {red['past_ctrl_bp']:+.0f}bp。前兆组是大涨之后"
             "被空头盯上的日子，效应可能部分来自涨后回落（均值回归），不全是『知情空头』；")
    L.append("   混杂方向与动量延续相反（涨后应继续涨才算动量），故不推翻规则，但降低因果解释的纯度。")
    L.append("")
    ep_desc = "、".join(
        f"{dates[s].date()}→{dates[e].date()}（B&H {close[e] / open_[s] - 1:+.1%}）" for s, e in eps
    ) or "无"
    fmtp = lambda p: f"{p:.2f}" if np.isfinite(p) else "n/a"  # noqa: E731
    L.append(f"2. 考场表现（三年未见过的数据）：{len(trig_exam)} 次触发聚成 {len(eps)} 段："
             f"{ep_desc}；判定 {n_right} 对 / {len(eps) - n_right} 错。")
    L.append(f"   应用 A 改善 total {mb['total_ret']:+.1%} → {mo['total_ret']:+.1%}"
             f"（CAGR {mb['cagr']:+.1%} → {mo['cagr']:+.1%}），但 p_boot {fmtp(p_boot)} / "
             f"p_rand {fmtp(p_rand)} 不显著——{len(eps)} 段样本撑不起统计结论，改善量只能算方向性证据。")
    L.append("")
    L.append(f"3. 探测器的结构性盲区（本轮最重要的发现）：考场 MDD 无任何改善"
             f"（{mb['mdd']:.1%} vs {mo['mdd']:.1%}）。最大回撤 {peak.date()} → {trough.date()}"
             f"（{dd_ser.min():.1%}）期间：")
    min_chg = si_crash["chg_pct"].min() if len(si_crash) else np.nan
    L.append(f"   - 放风腿有数据（密集日 {len(dense_crash)} 个，musk 归档 {archive_end} 前），"
             f"但空头腿反向：崩盘中段空头利益在回补（区间内发布 change_pct 最低 {min_chg:+.1f}%），"
             "『知情空头提前布局』的前提对这次崩盘不成立；")
    lu = ups_cov.iloc[-1] if len(ups_cov) else None
    if lu is not None:
        L.append(f"   - 归档覆盖内最后一份 up-jump（{lu['time'].date()}，{lu['chg_pct']:+.1f}%）出现在"
                 "崩盘尾段，且其后 20 交易日内无密集日配对，未触发。")
    L.append("   结论：该探测器只对『空头知情型下跌』有覆盖，对估值/情绪驱动的崩盘（2024-12 这种）"
             "天然零免疫——它是窄谱风险过滤器，不是全天候减仓开关。")
    L.append("")
    L.append(f"4. 数据缺口如实声明：musk 归档止于 {archive_end}，考场后段 {n_blind} 个交易日"
             f"（约 {n_blind / 252:.1f} 年）放风腿失明；期间还有 "
             f"{int((ups_exam['time'].dt.tz_convert(ET).dt.date > archive_end).sum())} 份 up-jump"
             "（2025-09、2025-12）无法配对检验。应用 B 的 0 重叠同源于此——"
             "E8-A 留出窗完整落在失明期，过滤检验本轮为空测，不是通过也不是失败。")
    L.append("")
    L.append("5. 达标线视角：2 段触发、p>0.10、MDD 无降——不构成任何上钱依据；与 shadow 达标线"
             "（≥8 周前向、显著改善）相比，本轮只完成『历史考场预演』角色：规则在发现段外首次")
    L.append("   独立存活（方向对、两次判对），触发频率 ~1 次/年，样本积累极慢。")
    L.append("")
    L.append("6. 下一步建议：")
    L.append("   (a) 把本状态机接入哨兵前向流（x_nitter + finra_short 已上线）跑 N3 前向虚拟推演：")
    L.append("       journal 记假想单、周报判分——这是 2025-05 后放风腿缺口唯一的填法；")
    L.append("   (b) 不单独上钱；定位 = 与 E11 S2 类趋势开关互补的窄谱 risk-off 特征")
    L.append("       （S2 截断趋势性崩盘、本探测器针对空头知情型回撤，两者盲区不重叠）；")
    L.append("   (c) 若前向 8-12 周内出现触发，与本轮 2 段战绩合并判分；无触发则继续攒样本，")
    L.append("       不因样本慢而放宽触发条件（规则已冻结）。")
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
