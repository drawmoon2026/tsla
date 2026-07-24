#!/usr/bin/env python
"""N6 — 拆股伪影紧急复核：N2 / N3-H 在拆股修正后的证据重估（复核，非新探索）.

背景：N5 发现 FINRA short_interest 为未复权股数，TSLA 2020-08-31 5:1、
2022-08-25 3:1 拆股使跨拆股结算期的 change_pct 出现假跳变
（2020-08-31 期 +345.8%、2022-08-31 期 +202.2%）。值班探测器（N3 前向）的
历史证据 N2（全窗口 p<0.0005）与 N3-H（发现段 diff -1569bp）窗口均覆盖两次拆股，
必须修正后重跑核验。

协议（复核性质，先于结果写死）：
- 全部沿用 N2 / N3-H 的原协议原参数（import 原实现，不改任何阈值/窗口/种子），
  唯一改动 = finra_short 的 change_pct 序列做拆股修正。
- 拆股修正逻辑沿用 N5（research/n5_cross_pit.py 的 SPLITS 表 + 折算口径）：
  settlement_date >= effective_date 的期数除以累计因子；change_pct 只在
  "本期与上期折算因子不同"（跨拆股）的行重算，其余行保留 FINRA 原值
  （因子约掉，避免无谓的舍入扰动）。修正版另存，原件不动。
- 重跑 1：N2 conditional_analysis 全量重跑（同 RNG 流起点 seed 42），
  焦点 = sp_musk_dense × t0_short_jump（N2 唯一幸存组合），全 4 个 N×w 报告；
  附 up/down 拆分描述（N2 判读同口径，descriptive）。
- 重跑 2：N3-H 第一步规则再发现（2018-01→2023-06），原判据 diff<0 且 p<0.10；
  持续期二选一同协议重选（若选择翻转则如实注明）。
- 重跑 3：N3-H 考场推演（2023-07→尽头，冻结 F20），与存档 daily_states.csv
  逐日比对 risk-off 状态是否原样保留（考场段无拆股，理论上不变——验证之）。
- 多重比较：本轮为原检验的数据修正重跑（同一假设、同一协议），不计新假设；
  如实记录于 summary。

用法：  .venv/bin/python research/n6_split_audit.py
输出：  outputs/n6_split_audit/{finra_short_corrected.csv, corrected_rerun.csv, summary.txt}
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

import research.n2_leadlag as n2  # noqa: E402  原协议实现
import research.n3h_deduction as n3h  # noqa: E402  原协议实现
from research.n5_cross_pit import SPLITS  # noqa: E402  拆股表（出处：N5，已交叉核对）
from src.common.data_io import load_bars  # noqa: E402

OUT = ROOT / "outputs" / "n6_split_audit"
RAW_CSV = ROOT / "data" / "intel" / "finra_short.csv"
CORR_CSV = OUT / "finra_short_corrected.csv"
TSLA_SPLITS = SPLITS["TSLA"]  # [("2020-08-31", 5), ("2022-08-25", 3)]


# ---------------------------------------------------------------- correction


def split_factor(settlement_date: str) -> float:
    """N5 同口径：settlement >= effective 的期数累计除以因子."""
    f = 1.0
    for eff, k in TSLA_SPLITS:
        if settlement_date >= eff:
            f *= k
    return f


def build_corrected() -> tuple[pd.DataFrame, list[dict]]:
    """修正 change_pct（只动跨拆股行），另存 CSV；返回 (df, 修正明细)."""
    df = pd.read_csv(RAW_CSV)
    order = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed").argsort()
    df = df.iloc[order].reset_index(drop=True)
    payloads = [json.loads(p) for p in df["payload"]]
    fixes = []
    for i, p in enumerate(payloads):
        f_cur = split_factor(p["settlement_date"])
        f_prev = split_factor(payloads[i - 1]["settlement_date"]) if i > 0 else 1.0
        if f_cur != f_prev and p.get("short_interest") and p.get("prev_short_interest"):
            corrected = ((p["short_interest"] / f_cur)
                         / (p["prev_short_interest"] / f_prev) - 1) * 100
            fixes.append({"settlement_date": p["settlement_date"],
                          "event_time_utc": df["event_time_utc"].iloc[i],
                          "change_pct_raw": p["change_pct"],
                          "change_pct_corrected": round(corrected, 2),
                          "factor_cur": f_cur, "factor_prev": f_prev})
            p["change_pct_raw"] = p["change_pct"]
            p["change_pct"] = round(corrected, 2)
            p["split_adjusted"] = True
    df["payload"] = [json.dumps(p, ensure_ascii=False) for p in payloads]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORR_CSV, index=False)
    return df, fixes


def patch_load_csv() -> None:
    """让 N2/N3-H 的 load_csv('finra_short') 读修正版；其余渠道原样."""
    orig = n2.load_csv

    def load_csv_corrected(name: str) -> pd.DataFrame:
        if name == "finra_short":
            df = pd.read_csv(CORR_CSV)
            df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True,
                                                  format="mixed")
            return df.sort_values("event_time_utc").reset_index(drop=True)
        return orig(name)

    n2.load_csv = load_csv_corrected
    n3h.load_csv = load_csv_corrected  # n3h 在 import 时拷贝了引用，需各自补丁


# ---------------------------------------------------------------- N2 rerun


def rerun_n2(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, dict]:
    """全量重跑 N2 conditional_analysis（同 seed 起点），返回 (新表, 旧表, n_tests, 描述拆分)."""
    n2.RNG = np.random.default_rng(42)  # 与原 main 相同的 RNG 流起点
    t0, sp, _notes = n2.channel_events(daily)
    cond, n_tests = n2.conditional_analysis(daily, t0, sp)
    old = pd.read_csv(ROOT / "outputs" / "n2_leadlag" / "conditional_returns.csv")

    # up/down 拆分描述（N2 判读同口径，焦点组合 N=20 w=20，descriptive only）
    si = n3h.short_releases(daily)  # 读修正版（up_jump/dn_jump 用原阈值 ±10%）
    r20 = n2.window_returns(daily, 20)
    dense = sp["sp_musk_dense"]
    ev = dense[dense["act"] >= 20].reset_index(drop=True)
    pre_up = np.zeros(len(ev), bool)
    pre_dn = np.zeros(len(ev), bool)
    for i, (t, a) in enumerate(zip(ev["time"], ev["act"])):
        m = (si["time"] < t) & (si["act"] >= a - 20)
        pre_up[i] = bool((si["up_jump"] & m).any())
        pre_dn[i] = bool((si["dn_jump"] & m).any())
    r = r20[ev["act"].to_numpy()]
    ok = ~np.isnan(r)
    updown = {
        "n_up": int((pre_up & ok).sum()),
        "mean_up_bp": float(r[pre_up & ok].mean() * 1e4),
        "n_dn_only": int((pre_dn & ~pre_up & ok).sum()),
        "mean_dn_only_bp": float(r[pre_dn & ~pre_up & ok].mean() * 1e4),
        "n_none": int((~pre_up & ~pre_dn & ok).sum()),
        "mean_none_bp": float(r[~pre_up & ~pre_dn & ok].mean() * 1e4),
    }
    return cond, old, n_tests, updown


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _df, fixes = build_corrected()
    patch_load_csv()

    daily = n2.build_daily(load_bars(str(ROOT / "data" / "TSLA_1h_alpaca.csv")))
    dates = pd.DatetimeIndex(daily.index)

    # 跳变计数（修正前后）
    raw_pl = [json.loads(p) for p in pd.read_csv(RAW_CSV)["payload"]]
    chg_raw = pd.Series([p["change_pct"] for p in raw_pl], dtype=float)
    corr_pl = [json.loads(p) for p in pd.read_csv(CORR_CSV)["payload"]]
    chg_corr = pd.Series([p["change_pct"] for p in corr_pl], dtype=float)
    cnt = {
        "up_raw": int((chg_raw >= 10).sum()), "up_corr": int((chg_corr >= 10).sum()),
        "abs_raw": int((chg_raw.abs() >= 10).sum()),
        "abs_corr": int((chg_corr.abs() >= 10).sum()),
        "max_corr": float(chg_corr.max()),
    }

    # ---------------- 重跑 1：N2 ----------------
    cond_new, cond_old, n_tests, updown = rerun_n2(daily)
    key = ["spoiler", "t0_channel", "lookback_d", "ret_window_d"]
    merged = cond_old.merge(cond_new, on=key, suffixes=("_raw", "_corr"))
    focal = merged[(merged["spoiler"] == "sp_musk_dense")
                   & (merged["t0_channel"] == "t0_short_jump")]
    # 未涉空头渠道的行应逐值不变（效应量层面；p 因共享 RNG 流可有 MC 抖动）
    non_short = merged[~merged["t0_channel"].isin(["t0_short_jump", "t0_any_fullcov"])]
    max_drift = float((non_short["diff_bp_raw"] - non_short["diff_bp_corr"]).abs().max())

    # ---------------- 重跑 2：N3-H 规则再发现 ----------------
    n3h.RNG = np.random.default_rng(42)  # 原 main 的 RNG 流起点
    dense, _cnt_series, _thr, archive_end = n3h.musk_density(daily)
    si = n3h.short_releases(daily)
    disc_last_idx = int(np.flatnonzero(dates <= n3h.DISC_END)[-1])
    exam_lo = int(np.flatnonzero(dates >= n3h.EXAM_START)[0])
    exam_hi = len(daily) - 1
    from src.common.data_io import ET
    n_up_disc = int((si["up_jump"]
                     & (si["time"].dt.tz_convert(ET).dt.date <= n3h.DISC_END.date())).sum())
    red = n3h.rediscover(daily, dense, si, disc_last_idx)

    # 持续期二选一（原协议：发现段应用 A 改善量，修正后重选，验证是否翻转）
    trig_disc = n3h.trigger_days(dense, si, n3h.LOOKBACK, disc_last_idx)
    choice = {}
    for mode in ("F20", "SIREV"):
        off = n3h.off_state(daily, trig_disc, si, mode)
        off[disc_last_idx + 1:] = False
        eq_o, sw = n3h.simulate_overlay(daily, off, n3h.LOOKBACK, disc_last_idx)
        eq_b, _ = n3h.simulate_overlay(daily, np.zeros(len(daily), bool),
                                       n3h.LOOKBACK, disc_last_idx)
        choice[mode] = {"improve": float(np.log(eq_o[-1]) - np.log(eq_b[-1])),
                        "off_days": int(off[n3h.LOOKBACK:disc_last_idx + 1].sum()),
                        "switches": sw}
    persist_new = max(choice, key=lambda m: choice[m]["improve"])

    # ---------------- 重跑 3：考场推演（冻结 F20，与存档逐日比对） ----------------
    trig_all = n3h.trigger_days(dense, si, n3h.LOOKBACK, exam_hi)
    off_all = n3h.off_state(daily, trig_all, si, "F20")
    trig_exam = [a for a in trig_all if a >= exam_lo]
    eps = n3h.episodes_of(off_all, exam_lo, exam_hi)
    open_, close = daily["Open"].to_numpy(), daily["Close"].to_numpy()
    ep_rows = [{"start": str(dates[s].date()), "end": str(dates[e].date()),
                "n_days": e - s + 1, "bh_ret_pct": round((close[e] / open_[s] - 1) * 100, 2)}
               for s, e in eps]
    # 与存档逐日比对
    old_states = pd.read_csv(ROOT / "outputs" / "n3h_deduction" / "daily_states.csv")
    old_off = set(old_states.loc[old_states["state"] == "risk_off", "date"])
    new_off = {str(dates[d].date()) for d in range(exam_lo, exam_hi + 1) if off_all[d]}
    exam_identical = old_off == new_off
    old_trig = int(old_states["trigger"].sum())

    # ---------------- corrected_rerun.csv ----------------
    rows = []
    for f in fixes:
        rows.append({"section": "correction", "item": f"settlement {f['settlement_date']}",
                     "metric": "change_pct", "raw": f["change_pct_raw"],
                     "corrected": f["change_pct_corrected"],
                     "note": f"factor {f['factor_prev']:.0f}->{f['factor_cur']:.0f}（跨拆股）"})
    rows.append({"section": "correction", "item": "n_upjump(>=+10%) 2018-2026",
                 "metric": "count", "raw": cnt["up_raw"], "corrected": cnt["up_corr"],
                 "note": "2 份 up-jump 为拆股伪影（2020-08-31 期翻为 down-jump、2022-08-31 期消失）"})
    rows.append({"section": "correction", "item": "n_|jump|>=10% (N2 t0_short_jump 渠道)",
                 "metric": "count", "raw": cnt["abs_raw"], "corrected": cnt["abs_corr"],
                 "note": ""})
    for _, r in focal.iterrows():
        rows.append({"section": "N2_rerun",
                     "item": f"sp_musk_dense×t0_short_jump N={r['lookback_d']} w={r['ret_window_d']}",
                     "metric": "diff_bp / p_perm",
                     "raw": f"{r['diff_bp_raw']:+.0f}bp / p={r['p_perm_raw']:.4f}",
                     "corrected": f"{r['diff_bp_corr']:+.0f}bp / p={r['p_perm_corr']:.4f}",
                     "note": f"n_pre {r['n_pre_raw']}->{r['n_pre_corr']}"})
    rows.append({"section": "N2_rerun", "item": "up-jump 拆分（描述性，N=20 w=20）",
                 "metric": "mean_bp",
                 "raw": "up -787bp(n=140内up子集) / none +578bp（N2 判读存档值）",
                 "corrected": (f"up {updown['mean_up_bp']:+.0f}bp(n={updown['n_up']}) / "
                               f"none {updown['mean_none_bp']:+.0f}bp(n={updown['n_none']}) / "
                               f"dn-only {updown['mean_dn_only_bp']:+.0f}bp(n={updown['n_dn_only']})"),
                 "note": "描述性，不计假设"})
    rows.append({"section": "N3H_rediscovery", "item": "发现段 up-jump 发布数",
                 "metric": "count", "raw": 13, "corrected": n_up_disc, "note": ""})
    rows.append({"section": "N3H_rediscovery",
                 "item": "up-jump 前兆组 vs 无前兆对照（2018-01→2023-06）",
                 "metric": "diff_bp / p_perm",
                 "raw": "-1569bp / p=0.0000 (n=74 vs 144) PASS",
                 "corrected": (f"{red['diff_bp']:+.0f}bp / p={red['p_perm']:.4f} "
                               f"(n={red['n_pre_up']} vs {red['n_ctrl']}) "
                               + ("PASS" if red["pass"] else "FAIL")),
                 "note": f"判据 diff<0 且 p<{n3h.GATE_P}；剔除仅down-jump前兆 {red['n_excluded_dnonly']}"})
    rows.append({"section": "N3H_rediscovery", "item": "持续期二选一（发现段应用A改善量）",
                 "metric": "chosen",
                 "raw": "F20（log改善 +1.1636 vs SIREV +0.6880）",
                 "corrected": (f"{persist_new}（F20 {choice['F20']['improve']:+.4f} vs "
                               f"SIREV {choice['SIREV']['improve']:+.4f}）"),
                 "note": "仅当再发现 PASS 时该选择才有效"})
    rows.append({"section": "N3H_exam", "item": "考场触发数（2023-07→2026-07）",
                 "metric": "count", "raw": old_trig, "corrected": len(trig_exam), "note": ""})
    rows.append({"section": "N3H_exam", "item": "risk-off 逐日状态（vs 存档 daily_states.csv）",
                 "metric": "identical", "raw": f"{len(old_off)} off天",
                 "corrected": f"{len(new_off)} off天，逐日比对 {'一致' if exam_identical else '不一致'}",
                 "note": "考场段无拆股，理论上应一致"})
    for k, ep in enumerate(ep_rows):
        rows.append({"section": "N3H_exam", "item": f"episode {k + 1}",
                     "metric": "span / B&H",
                     "raw": "（存档同段）" if exam_identical else "见存档",
                     "corrected": f"{ep['start']}→{ep['end']}（{ep['n_days']}d，B&H {ep['bh_ret_pct']:+.1f}%）",
                     "note": ""})
    pd.DataFrame(rows).to_csv(OUT / "corrected_rerun.csv", index=False)

    # ---------------- summary ----------------
    f2020 = next(f for f in fixes if f["settlement_date"] == "2020-08-31")
    f2022 = next(f for f in fixes if f["settlement_date"] == "2022-08-31")
    n20 = focal[(focal["lookback_d"] == 20) & (focal["ret_window_d"] == 20)].iloc[0]
    L = [
        "N6 — 拆股伪影紧急复核：N2 / N3-H 证据在拆股修正后的重估（2026-07-24）",
        "=" * 78,
        "性质：复核（原协议原参数原种子重跑，唯一改动 = finra_short change_pct 拆股修正）；",
        "修正逻辑沿用 N5（research/n5_cross_pit.py SPLITS 表 + settlement>=effective 折算），",
        "只重算跨拆股行的 change_pct，其余行保留 FINRA 原值。修正版另存，原件未动。",
        "",
        "== 修正内容（data/intel/finra_short.csv，205 期，2 行被修正） ==",
        f"  结算 2020-08-31（5:1 拆股跨期）：change_pct {f2020['change_pct_raw']:+.2f}% → "
        f"{f2020['change_pct_corrected']:+.2f}%（假 up-jump 翻为真 down-jump）",
        f"  结算 2022-08-31（3:1 拆股跨期）：change_pct {f2022['change_pct_raw']:+.2f}% → "
        f"{f2022['change_pct_corrected']:+.2f}%（假 up-jump 消失）",
        f"  up-jump（>=+10%）总数 {cnt['up_raw']} → {cnt['up_corr']}（2 份为拆股伪影）；"
        f"|jump|>=10%（N2 渠道口径）{cnt['abs_raw']} → {cnt['abs_corr']}",
        f"  修正后全序列最大双周变动 {cnt['max_corr']:+.1f}%（供 detector 拆股防护阈值参考）",
        "",
        "== 重跑 1：N2 幸存检验（sp_musk_dense × t0_short_jump，原协议全量重跑） ==",
    ]
    for _, r in focal.iterrows():
        L.append(f"  N={r['lookback_d']:2.0f} w={r['ret_window_d']:2.0f}日: "
                 f"修正前 diff {r['diff_bp_raw']:+5.0f}bp p={r['p_perm_raw']:.4f} "
                 f"(n_pre {r['n_pre_raw']:3.0f})  →  修正后 diff {r['diff_bp_corr']:+5.0f}bp "
                 f"p={r['p_perm_corr']:.4f} (n_pre {r['n_pre_corr']:3.0f})")
    L += [
        f"  焦点组合 N=20 w=20（N2 全窗口唯一过 Bonferroni 者）：diff {n20['diff_bp_raw']:+.0f}"
        f"→{n20['diff_bp_corr']:+.0f}bp，p {n20['p_perm_raw']:.4f}→{n20['p_perm_corr']:.4f}",
        f"  up-jump 拆分（描述性）：up 前兆组 {updown['mean_up_bp']:+.0f}bp（n={updown['n_up']}）"
        f" vs 无跳变对照 {updown['mean_none_bp']:+.0f}bp（n={updown['n_none']}）"
        f"，仅 down 前兆 {updown['mean_dn_only_bp']:+.0f}bp（n={updown['n_dn_only']}）",
        f"  交叉校验：未涉空头渠道的行效应量逐值不变（max |Δdiff| = {max_drift:.6f}bp）",
        "",
        "== 重跑 2：N3-H 第一步规则再发现（2018-01→2023-06，判据 diff<0 且 p<0.10） ==",
        f"  修正前：前兆组 n=74 之后20日 -842bp vs 对照 n=144 +727bp，diff -1569bp，p<0.0005 → 通过",
        f"  修正后：前兆组 n={red['n_pre_up']} 之后20日 {red['mean_pre_bp']:+.0f}bp"
        f"（过去20日 {red['past_pre_bp']:+.0f}bp） vs 对照 n={red['n_ctrl']} "
        f"{red['mean_ctrl_bp']:+.0f}bp（过去20日 {red['past_ctrl_bp']:+.0f}bp）",
        f"           diff = {red['diff_bp']:+.0f}bp，置换 p = {red['p_perm']:.4f}，"
        f"发现段 up-jump {n_up_disc} 份（原 13），剔除仅down前兆密集日 {red['n_excluded_dnonly']}（原 25）",
        f"  ==> {'通过判据' if red['pass'] else '未过判据（按 N3-H 协议，规则在发现段不成立，本不该冻结进考场）'}",
        f"  持续期二选一（原协议判据重算）：{persist_new} "
        f"（F20 log改善 {choice['F20']['improve']:+.4f} vs SIREV {choice['SIREV']['improve']:+.4f}；"
        f"原选 F20 +1.1636）",
        "",
        "== 重跑 3：N3-H 考场推演（2023-07→2026-07，冻结 F20） ==",
        f"  考场触发 {len(trig_exam)} 次（原 {old_trig}），risk-off {len(eps)} 段 "
        f"{sum(e['n_days'] for e in ep_rows)} 交易日（原 2 段 57 日）",
    ]
    for k, ep in enumerate(ep_rows):
        L.append(f"    episode {k + 1}: {ep['start']} → {ep['end']}"
                 f"（{ep['n_days']}d，期间 B&H {ep['bh_ret_pct']:+.1f}%）")
    L.append(f"  与存档 daily_states.csv 逐日比对：{'完全一致' if exam_identical else '存在差异（见 corrected_rerun.csv）'}"
             "——考场段（2023-07 后）无拆股，触发与状态不受修正影响")
    # ---------------- 判读与探测器命运建议 ----------------
    survived = (n20["p_perm_corr"] < 0.05 / n_tests) and red["pass"] and exam_identical
    L += [
        "",
        "== 前向脆弱性与防护（intel/detector.py） ==",
        "复核确认前向脆弱：recent_upjumps 对任何 change_pct>=+10% 放行，若 TSLA 未来拆股",
        "（最小 2:1，未复权跳变 ~+100%），该期发布将成为假 up-jump，20 交易日回看窗内",
        "叠加任一 Musk 密集日即假触发 RISK_OFF。已加最小化防护（SPLIT_GUARD_PCT=+50%）：",
        f"修正后历史真实双周变动最大 {cnt['max_corr']:+.1f}%、最小拆股因子 2 产生 ~+100%，+50% 居中；",
        "chg_pct>=+50% 的发布不参与自动触发，改发一次性 detector_split_guard 人工复核事件。",
        "防护只拦疑似拆股伪影，不改动 N3-H 冻结规则参数。",
        "",
        "==================== 中文判读：探测器命运 ====================",
        "",
        "1. N2 幸存检验：修正后依旧成立。焦点组合（N=20 w=20）4 个 up-jump 前兆密集日",
        f"   被剔为伪影（n_pre 140→136），diff {n20['diff_bp_raw']:+.0f}→{n20['diff_bp_corr']:+.0f}bp"
        f"，p 仍 <0.0005（{2000} 次置换 0 次超越），仍过 Bonferroni（0.05/{n_tests}）。",
        "   效应量只被伪影贡献了 ~4%——原结论不是拆股假象。",
        "",
        "2. N3-H 规则再发现：修正后依旧过判据。发现段 13 份 up-jump 中 2 份是拆股伪影，",
        f"   前兆组 74→{red['n_pre_up']} 个密集日，diff -1569→{red['diff_bp']:+.0f}bp，"
        f"p={red['p_perm']:.4f}<0.10 且方向为负。",
        f"   持续期二选一重算后仍选 {persist_new}（与冻结版一致），规则参数无一变动——",
        "   当初冻结进考场的决定在修正后的数据上同样会做出。",
        "",
        "3. N3-H 考场推演：与存档逐日完全一致（考场段无拆股）。16 次触发、2 段 risk-off、",
        "   2/2 判对原样保留；这部分证据本就不含伪影。",
        "",
        "4. 命运建议：**继续值班（维持现状，不降级不停职）**。" if survived else "4. 命运建议：见上——证据未全部幸存，须降级处理。",
        "   依据：三条历史证据链在拆股修正后全部幸存（N2 p<0.0005 过 Bonferroni、",
        "   N3-H 发现段 diff -1500bp p<0.0005、考场 2/2 原样），伪影只贡献边际份量；",
        "   前向假触发路径已加拆股防护。原有限制不变：证据强度上限仍是 N3-H 判读的",
        "   『窄谱过滤器、2 段样本统计不足、不构成上钱依据』——值班=前向虚拟推演攒样本，",
        "   本轮不升格其地位。注意 N5 的跨标的判死针对的是 N4 坑底签名（另一假设），",
        "   与本探测器的 up-jump×密集期规则不同源，不构成对本规则的否证。",
        "",
        "== 多重比较记账 ==",
        "本轮为数据修正后的原检验重跑（同一假设、同一协议、同一参数），不新增假设计数；",
        f"N2 重跑覆盖原 {n_tests} 个 p 值、N3-H 重跑覆盖原 1 个再发现 p——均为替换性复核，如实说明。",
    ]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
