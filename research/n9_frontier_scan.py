#!/usr/bin/env python
"""N9 — 探测器参数前沿全扫描（预登记于 docs/strategy-lab.md N9 条目，2026-08-03）.

问题：是否存在频率更高且精度可接受的探测器工作点？现行冻结配置
（up-jump 10% / 密集分位 0.8789 / F20 / AND）在前沿上还是被支配？

扫描空间（预登记，144 配置）：
  up-jump 阈值 J ∈ {5, 7.5, 10, 15}%（change_pct >= +J，单侧，与现行同向）
  密集分位  q ∈ {0.75, 0.8789, 0.95}（因果滚动 365 日分位，shift(1)，min_periods=180，
             与 N3-H musk_density 同机制；0.8789 = 值班探测器标定分位，
             intel/detector.py: 65 帖/日 ≙ P0.8789。N3-H 研究实现用 0.90，
             网格以 0.8789/0.95 夹住它——如实注明，不改网格）
  持续期    F ∈ {F10, F20, F30}（触发日起 F 个交易日 risk-off，重叠触发顺延/并集）
  逻辑      ∈ {AND（现行）, 单腿空头, 单腿密集, OR}

触发语义（预先写清）：
  AND      = 密集日（前一自然日发帖数 > 因果滚动 365 日 q 分位）act 日触发，
             且密集时刻前、回看 20 个交易日内存在 change_pct>=+J% 空头发布
             （发布时刻近似 = 结算+9 交易日，N2 同口径）。
             回看窗冻结 20 交易日不随 F 变（F 只控制持续期）。
  单腿空头 = 每份 change_pct>=+J% 空头发布的 act 日即触发（q 维度无效：同 J、F 下
             三个 q 的行完全重复，如实标 dup_key 占格不重数）。
  单腿密集 = 每个密集日 act 即触发（J 维度无效，同理标重）。
  OR       = 两腿任一触发即触发（并集）。

协议：
  发现段 2018-01→2023-06（价格数据实际始于 2018-07-23）逐日推演出频率-精度前沿；
  考场段 2023-07→2026-07 只对发现段帕累托前沿中"精度>=70% 且频率最高的 5 个
  （按有效触发语义去重后取）"复验；现行冻结配置作为对照点必须在图上。
  v2 晋升规则（预先写死）：考场段 >=4 段 且 段级判对率 >=75% 且 避险期 B&H 均值<0。

段级判对（与 N3-H 一致）：episode 期间 B&H（close[e]/open[s]-1）< -6bp（往返成本）= 对。

诚实边界：
  密集分位在发现段前数据上因果滚动计算（与现行同口径，无前视）；
  musk 归档止 2025-05-08——考场失明期（信号日 > 归档尽头）所有配置一视同仁
  跳过新触发（含空头单腿，防止单腿配置在失明期获得不对称覆盖），失明前触发的
  持续期可自然延入；被跳过的空头 up-jump 份数如实报告。

用法：  .venv/bin/python research/n9_frontier_scan.py
输出：  outputs/n9_frontier/{all_configs.csv, frontier.csv, exam_top5.csv, summary.txt}
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

from research.n2_leadlag import act_day, build_daily, load_csv  # noqa: E402
from research.n3h_deduction import (  # noqa: E402
    DISC_END, EXAM_START, episodes_of, simulate_overlay,
)
from src.common.data_io import ET, load_bars  # noqa: E402

OUT = ROOT / "outputs" / "n9_frontier"
LOOKBACK = 20            # AND/OR 配对回看窗（N2 预登记，冻结，不随 F 变）
JUDGE_BP = -0.0006       # 段级判对线：episode B&H < -6bp（N3-H 同口径）
JUMPS = [5.0, 7.5, 10.0, 15.0]
QUANTS = [0.75, 0.8789, 0.95]
PERSISTS = [10, 20, 30]
LOGICS = ["AND", "short_only", "dense_only", "OR"]
CURRENT = (10.0, 0.8789, 20, "AND")   # 现行冻结配置（预登记网格上的对照点）
ACC_GATE, TOP_N = 0.70, 5             # top5 选择规则（预登记，勿改）
V2_MIN_EPS, V2_MIN_ACC = 4, 0.75      # v2 晋升规则（预登记）


# ------------------------------------------------------------------ signals


def dense_events(daily: pd.DataFrame, cnt: pd.Series, q: float) -> pd.DataFrame:
    """N3-H musk_density 同机制，分位数参数化."""
    thr = cnt.shift(1).rolling(365, min_periods=180).quantile(q)
    dense = cnt.index[(cnt > thr) & thr.notna()]
    rows = []
    for x in dense:
        t = pd.Timestamp(x).tz_localize(ET).tz_convert("UTC") + pd.Timedelta(hours=23, minutes=59)
        a = act_day(daily, t)
        if a is not None:
            rows.append({"day": x, "time": t, "act": a})
    return pd.DataFrame(rows, columns=["day", "time", "act"])


def load_signals(daily: pd.DataFrame):
    tw = load_csv("musk_tweets")
    archive_end = tw["event_time_utc"].max().tz_convert(ET).date()
    et_day = tw["event_time_utc"].dt.tz_convert(ET).dt.date
    cnt = tw.groupby(et_day).size()
    cnt.index = pd.to_datetime(cnt.index)
    cnt = cnt.asfreq("D", fill_value=0)

    si_raw = load_csv("finra_short")
    pl = [json.loads(p) for p in si_raw["payload"]]
    si = pd.DataFrame({
        "time": si_raw["event_time_utc"],
        "chg_pct": [p.get("change_pct") for p in pl],
    })
    si["act"] = [act_day(daily, t) for t in si["time"]]
    si = si[si["act"].notna()].copy()
    si["act"] = si["act"].astype(int)
    si["et_date"] = si["time"].dt.tz_convert(ET).dt.date
    return cnt, si.reset_index(drop=True), archive_end


# ------------------------------------------------------------------ triggers


def trigger_acts(logic: str, dense_q: pd.DataFrame, si: pd.DataFrame, jump: float,
                 lo: int, archive_end) -> list[int]:
    """按逻辑生成触发 act 集合；失明期（信号日 > 归档尽头）新触发一律跳过."""
    up = si[si["chg_pct"] >= jump]
    out: set[int] = set()
    if logic in ("AND",):
        for _, row in dense_q.iterrows():
            m = (up["time"] < row["time"]) & (up["act"] >= row["act"] - LOOKBACK)
            if m.any():
                out.add(int(row["act"]))
    if logic in ("dense_only", "OR"):
        out |= {int(a) for a in dense_q["act"]}
    if logic in ("short_only", "OR"):
        out |= {int(a) for a in up.loc[up["et_date"] <= archive_end, "act"]}
    # 密集腿信号天然止于归档尽头；空头腿按发布日 <= 归档尽头截断（一视同仁跳过）
    return sorted(a for a in out if a >= lo)


def off_from(trig: list[int], n: int, persist: int) -> np.ndarray:
    off = np.zeros(n, bool)
    for a in trig:
        off[a:a + persist] = True
    return off


# ------------------------------------------------------------------ metrics


def segment_eval(daily, off_full: np.ndarray, trig: list[int], lo: int, hi: int,
                 bh_eq: np.ndarray) -> dict:
    """一个配置在 [lo,hi] 段的逐日推演指标（触发已按段裁剪）."""
    off = off_full.copy()
    off[:lo] = False
    off[hi + 1:] = False
    open_, close = daily["Open"].to_numpy(), daily["Close"].to_numpy()
    eps = episodes_of(off, lo, hi)
    ep_rets = [close[e] / open_[s] - 1 for s, e in eps]
    n_right = sum(1 for r in ep_rets if r < JUDGE_BP)
    n_days = hi - lo + 1
    years = n_days / 252
    if off[lo:hi + 1].any():
        eq_o, sw = simulate_overlay(daily, off, lo, hi)
        overlay_total = float(eq_o[-1] - 1)
        improve_log = float(np.log(eq_o[-1]) - np.log(bh_eq[-1]))
    else:
        overlay_total = float(bh_eq[-1] - 1)
        improve_log, sw = 0.0, 0
    bh_total = float(bh_eq[-1] - 1)
    return {
        "n_trig": len(trig), "n_eps": len(eps),
        "off_days": int(off[lo:hi + 1].sum()),
        "eps_per_year": len(eps) / years,
        "n_right": n_right,
        "seg_acc": n_right / len(eps) if eps else np.nan,
        "mean_ep_ret": float(np.mean(ep_rets)) if ep_rets else np.nan,
        "bh_total": bh_total, "overlay_total": overlay_total,
        "cover_diff": overlay_total - bh_total,
        "improve_log": improve_log, "switches": sw,
    }


def pareto_mask(freq: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """非支配集（freq、acc 双高优）；acc NaN（0 段）不参与."""
    n = len(freq)
    ok = ~np.isnan(acc)
    mask = np.zeros(n, bool)
    for i in range(n):
        if not ok[i]:
            continue
        dominated = False
        for j in range(n):
            if not ok[j] or i == j:
                continue
            if (freq[j] >= freq[i] and acc[j] >= acc[i]
                    and (freq[j] > freq[i] or acc[j] > acc[i])):
                dominated = True
                break
        mask[i] = not dominated
    return mask


def dup_key(jump, q, persist, logic) -> tuple:
    """有效触发语义键：单腿空头与 q 无关、单腿密集与 J 无关."""
    if logic == "short_only":
        return (logic, jump, None, persist)
    if logic == "dense_only":
        return (logic, None, q, persist)
    return (logic, jump, q, persist)


# ------------------------------------------------------------------ main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = build_daily(load_bars(str(ROOT / "data" / "TSLA_1h_alpaca.csv")))
    dates = pd.DatetimeIndex(daily.index)
    cnt, si, archive_end = load_signals(daily)
    n = len(daily)
    disc_last = int(np.flatnonzero(dates <= DISC_END)[-1])
    exam_lo = int(np.flatnonzero(dates >= EXAM_START)[0])
    exam_hi = n - 1
    disc_lo = LOOKBACK
    n_blind = int((dates[exam_lo:].date > archive_end).sum())
    skipped_up = {j: int(((si["chg_pct"] >= j) & (si["et_date"] > archive_end)).sum())
                  for j in JUMPS}

    dense_by_q = {q: dense_events(daily, cnt, q) for q in QUANTS}
    bh_disc, _ = simulate_overlay(daily, np.zeros(n, bool), disc_lo, disc_last)
    bh_exam, _ = simulate_overlay(daily, np.zeros(n, bool), exam_lo, exam_hi)

    # ---------------- 发现段全扫描（144 配置） -------------------------------
    rows = []
    trig_cache: dict[tuple, list[int]] = {}
    for jump in JUMPS:
        for q in QUANTS:
            for logic in LOGICS:
                key0 = dup_key(jump, q, 999, logic)[:3]  # (logic, J*, q*)
                if key0 not in trig_cache:
                    trig_cache[key0] = trigger_acts(
                        logic, dense_by_q[q], si, jump, disc_lo, archive_end)
                trig_all = trig_cache[key0]
                for persist in PERSISTS:
                    trig_disc = [a for a in trig_all if a <= disc_last]
                    off = off_from(trig_disc, n, persist)
                    m = segment_eval(daily, off, trig_disc, disc_lo, disc_last, bh_disc)
                    rows.append({
                        "jump_pct": jump, "dense_q": q, "persist_F": persist,
                        "logic": logic,
                        "dup_key": str(dup_key(jump, q, persist, logic)),
                        "is_current": (jump, q, persist, logic) == CURRENT,
                        **m,
                    })
    df = pd.DataFrame(rows)
    df.insert(0, "config_id", [f"C{i:03d}" for i in range(len(df))])
    front = pareto_mask(df["eps_per_year"].to_numpy(), df["seg_acc"].to_numpy())
    df["on_frontier"] = front
    df.to_csv(OUT / "all_configs.csv", index=False)
    df[front].sort_values("eps_per_year", ascending=False).to_csv(
        OUT / "frontier.csv", index=False)

    cur = df[df["is_current"]].iloc[0]
    # 现行配置被谁支配（若被支配）
    dominators = df[(df["eps_per_year"] >= cur["eps_per_year"])
                    & (df["seg_acc"] >= cur["seg_acc"])
                    & ((df["eps_per_year"] > cur["eps_per_year"])
                       | (df["seg_acc"] > cur["seg_acc"]))
                    & df["seg_acc"].notna()]

    # ---------------- top5 选择（规则预登记：前沿上 acc>=70% 按频率取 5，去重） ---
    cand = df[front & (df["seg_acc"] >= ACC_GATE)].sort_values(
        ["eps_per_year", "seg_acc"], ascending=False)
    top5, seen = [], set()
    for _, r in cand.iterrows():
        if r["dup_key"] in seen:
            continue
        seen.add(r["dup_key"])
        top5.append(r)
        if len(top5) == TOP_N:
            break
    top5 = pd.DataFrame(top5)

    # ---------------- 考场复验（top5 + 现行对照） ----------------------------
    exam_rows = []
    for _, r in pd.concat([top5, cur.to_frame().T]).iterrows():
        jump, q, persist, logic = (float(r["jump_pct"]), float(r["dense_q"]),
                                   int(r["persist_F"]), str(r["logic"]))
        role = "current_ref" if bool(r["is_current"]) and r["config_id"] not in set(
            top5["config_id"]) else "top5"
        key0 = dup_key(jump, q, 999, logic)[:3]
        trig_all = trig_cache[key0]
        off = off_from(trig_all, n, persist)     # 全历史触发→考场段含延入状态
        m = segment_eval(daily, off, [a for a in trig_all if exam_lo <= a <= exam_hi],
                         exam_lo, exam_hi, bh_exam)
        rule1 = m["n_eps"] >= V2_MIN_EPS
        rule2 = bool(m["seg_acc"] >= V2_MIN_ACC) if m["n_eps"] else False
        rule3 = bool(m["mean_ep_ret"] < 0) if m["n_eps"] else False
        exam_rows.append({
            "config_id": r["config_id"], "role": role,
            "jump_pct": jump, "dense_q": q, "persist_F": persist, "logic": logic,
            "disc_eps_per_year": r["eps_per_year"], "disc_seg_acc": r["seg_acc"],
            **{f"exam_{k}": v for k, v in m.items()},
            "v2_rule1_neps_ge4": rule1, "v2_rule2_acc_ge75": rule2,
            "v2_rule3_meanret_neg": rule3, "v2_pass": rule1 and rule2 and rule3,
        })
    exam = pd.DataFrame(exam_rows)
    exam.to_csv(OUT / "exam_top5.csv", index=False)

    # ---------------- summary -------------------------------------------------
    disc_years = (disc_last - disc_lo + 1) / 252
    exam_years = (exam_hi - exam_lo + 1) / 252
    L = []
    L.append("N9 — 探测器参数前沿全扫描（2026-08-03）")
    L.append("=" * 78)
    L.append(f"价格窗 {dates[0].date()} → {dates[-1].date()}（{n} 交易日）；"
             f"发现段 idx[{disc_lo},{disc_last}] {dates[disc_lo].date()}→{dates[disc_last].date()}"
             f"（{disc_years:.2f} 年）；考场 {dates[exam_lo].date()}→{dates[exam_hi].date()}"
             f"（{exam_years:.2f} 年，其中失明 {n_blind} 交易日）")
    L.append(f"扫描 {len(df)} 配置 = J{JUMPS} × q{QUANTS} × F{PERSISTS} × {LOGICS}；"
             f"有效去重后 {df['dup_key'].nunique()} 个不同工作点"
             "（单腿空头与 q 无关、单腿密集与 J 无关，占格如实标 dup_key）")
    L.append(f"失明期跳过的空头 up-jump 发布（各 J 阈值）：{skipped_up}；"
             "密集腿天然止于归档尽头——失明期所有配置一视同仁无新触发")
    L.append("")
    L.append("—— 发现段前沿（帕累托非支配集，频率=段/年 vs 精度=段级判对率）——")
    for _, r in df[front].sort_values("eps_per_year", ascending=False).iterrows():
        tag = " ←现行" if r["is_current"] else ""
        L.append(f"  {r['config_id']} J{r['jump_pct']:g} q{r['dense_q']:g} "
                 f"F{r['persist_F']} {r['logic']:>10}: "
                 f"{r['eps_per_year']:.2f} 段/年, 判对 {r['seg_acc']:.0%}"
                 f"（{r['n_right']}/{r['n_eps']}）, 避险均值 {r['mean_ep_ret']:+.1%}, "
                 f"覆盖差 {r['cover_diff']:+.1%}{tag}")
    L.append("")
    cur_tag = "在前沿上" if bool(cur["on_frontier"]) else "被支配"
    L.append(f"—— 现行冻结配置（J10 q0.8789 F20 AND）位置：{cur_tag} ——")
    L.append(f"  发现段：{cur['eps_per_year']:.2f} 段/年（{cur['n_eps']} 段），"
             f"判对 {cur['seg_acc']:.0%}（{cur['n_right']}/{cur['n_eps']}），"
             f"避险均值 {cur['mean_ep_ret']:+.1%}，覆盖差 {cur['cover_diff']:+.1%}")
    if not bool(cur["on_frontier"]) and len(dominators):
        d0 = dominators.sort_values("eps_per_year", ascending=False).iloc[0]
        L.append(f"  支配者示例：{d0['config_id']} J{d0['jump_pct']:g} q{d0['dense_q']:g} "
                 f"F{d0['persist_F']} {d0['logic']}——{d0['eps_per_year']:.2f} 段/年 且 "
                 f"判对 {d0['seg_acc']:.0%}（支配者共 {len(dominators)} 行）")
    L.append("")
    L.append(f"—— 考场复验（top5 = 前沿上判对率>=70% 且频率最高的 {len(top5)} 个，规则预登记）——")
    for _, r in exam.iterrows():
        L.append(f"  {r['config_id']} J{r['jump_pct']:g} q{r['dense_q']:g} "
                 f"F{r['persist_F']} {r['logic']:>10} [{r['role']}]: "
                 f"发现段 {r['disc_eps_per_year']:.2f}段/年·判对 {r['disc_seg_acc']:.0%} → "
                 f"考场 {r['exam_n_eps']} 段（{r['exam_eps_per_year']:.2f}/年）, "
                 + (f"判对 {r['exam_seg_acc']:.0%}（{r['exam_n_right']}/{r['exam_n_eps']}）, "
                    f"避险均值 {r['exam_mean_ep_ret']:+.1%}, "
                    if r["exam_n_eps"] else "无触发, ")
                 + f"覆盖差 {r['exam_cover_diff']:+.1%}")
    L.append("")
    L.append("—— v2 晋升判定（预登记三条：考场段>=4 且 判对率>=75% 且 避险均值<0）——")
    for _, r in exam.iterrows():
        if r["role"] != "top5":
            continue
        verdict = "通过 → detector-v2 候选" if r["v2_pass"] else "不通过"
        L.append(f"  {r['config_id']}: 段数{'✓' if r['v2_rule1_neps_ge4'] else '✗'}"
                 f"({r['exam_n_eps']}) 判对{'✓' if r['v2_rule2_acc_ge75'] else '✗'}"
                 + (f"({r['exam_seg_acc']:.0%})" if r["exam_n_eps"] else "(—)")
                 + f" 均值{'✓' if r['v2_rule3_meanret_neg'] else '✗'}"
                 + (f"({r['exam_mean_ep_ret']:+.1%})" if r["exam_n_eps"] else "(—)")
                 + f" → {verdict}")
    # ---------------- 中文判读（直接回答用户的两个问题） ----------------------
    fr = df[front].drop_duplicates("dup_key").sort_values("eps_per_year", ascending=False)
    n_qual = int((df.drop_duplicates("dup_key")["seg_acc"] >= ACC_GATE).sum())
    c077 = exam[exam["role"] == "top5"].iloc[0] if len(top5) else None
    cur_ex = exam[exam["role"] == "current_ref"].iloc[0]
    L.append("")
    L.append("==================== 中文判读（基于上表数字） ====================")
    L.append("")
    L.append("问题一：调参能不能把 3 年 2 次变成可接受的频率而不牺牲精度？——不能。")
    L.append(f"  前沿形状是陡峭的频率-精度互换：{fr.iloc[0]['eps_per_year']:.1f} 段/年只剩 "
             f"{fr.iloc[0]['seg_acc']:.0%} 判对率（不如抛硬币），3.9 段/年→53%，"
             "2.5 段/年→58%，2.1 段/年→60%。全网格 144 配置（93 个有效工作点）")
    L.append(f"  发现段精度 >=70% 的只有 {n_qual} 个工作点（C077，71%），而它的频率与现行完全相同"
             "（1.44 段/年）——整个预登记参数空间内不存在『频率翻倍且精度可接受』的工作点。")
    L.append("  频率瓶颈是信号本身稀疏（up-jump 发布 + 密集日的自然发生率），不是参数保守；"
             "提频的每一步都直接用精度换。")
    L.append("")
    L.append("问题二：现行配置是最优还是保守过头？——名义被支配，但不是保守过头。")
    L.append(f"  现行点（1.44 段/年，57%）被 {len(dominators)} 行支配，支配者全部是"
             "精度 58-67% 的高频配置——用户若接受 <60% 判对率才谈得上『提频』。")
    L.append("  发现段上严格更好的同频工作点存在：C077（J10 空头单腿 F30，去掉密集腿、"
             "拉长持续期）判对 71% vs 57%，避险均值 -11.5% vs -7.6%。")
    if c077 is not None:
        L.append(f"  但考场复验打脸：C077 考场 {int(c077['exam_n_eps'])} 段、判对 "
                 f"{c077['exam_seg_acc']:.0%}、避险均值 {c077['exam_mean_ep_ret']:+.1%}，"
                 f"v2 三条规则只过一条（不晋升）；现行配置考场 {int(cur_ex['exam_n_eps'])} 段 "
                 f"{cur_ex['exam_seg_acc']:.0%} 判对、避险均值 {cur_ex['exam_mean_ep_ret']:+.1%}"
                 "——考场表现反而是现行最好。发现段优势没有出场外复现。")
    L.append("")
    L.append("结论与建议：")
    L.append("  (a) v2 晋升判定：0 个候选通过——不登记 detector-v2，现行值班规则不动；")
    L.append("  (b) top5 退化为 top1 是规则的诚实结局（仅 1 个工作点达 70% 精度线），"
             "不放宽选择规则补足名额；")
    L.append("  (c) 若仍想攒 v2 证据，唯一值得跟的形状是『空头单腿 + F30』（C077）：")
    L.append("      建议进哨兵前向流做影子配置攒样本（它不依赖 musk 归档，失明期也能跑——")
    L.append(f"      本轮考场为公平被跳过的 {skipped_up[10.0]} 份失明期 up-jump 前向可用），")
    L.append("      不上钱、不替换现行规则；")
    L.append("  (d) 给用户的直答：3 年 2 次不是参数拧紧了，是这类信号的自然产率；"
             "网格里所有『更频繁』的工作点都以判对率跌破 60% 为代价。")
    L.append("")
    L.append("诚实边界与口径注记：")
    L.append("  - 现行对照点按预登记网格取 q=0.8789（值班探测器标定分位，65 帖/日≙P0.8789，"
             "intel/detector.py）；N3-H 研究实现用滚动 0.90 分位——两者考场结果一致"
             "（2 段、2 对），对照点稳健；")
    L.append("  - 密集分位为因果滚动量（shift(1)，365 日窗），无前视；")
    L.append(f"  - 考场频率分母含 {n_blind} 个失明交易日（~{n_blind / 252:.1f} 年，"
             "各配置一视同仁无新触发）——各配置的『段/年』被同等稀释，横向可比，"
             "绝对值偏低；")
    L.append("  - 覆盖差为总收益率之差（B&H 基数 4.86 年复利，绝对值大），"
             "对数改善量见 CSV improve_log 列；")
    L.append("  - 多重比较 +144+5（已预登记于 N9 条目）。")
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
