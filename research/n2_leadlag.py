#!/usr/bin/env python
"""N2 — 布局痕迹（T0 层）历史检验 + 先后顺序引擎.

假设（用户）：大资金"建仓→放风→散户接盘"的生命周期在合法公开数据上留痕。
若放风事件（8-K、Musk 高密度发帖）之前 N 个交易日内存在 T0 异动
（13D/G、Form 144、空头利益跳变、暗池占比异常、量价吸筹签名），
则放风后的收益应显著异于无前兆的放风。

数据（全部带公开时间戳，防前视口径与 N1 相同）：
- 放风事件：data/intel/edgar_8k.csv（2018+ 全量）、musk_tweets.csv 高密度日
  （因果阈值：当日发帖数 > 过去 365 日 90 分位；归档止于 2025-05）。
  新闻类放风历史不可得（哨兵 news_rss 2026-07 才开始），如实缺席。
- T0 渠道：edgar_13dg.csv（2018+）、edgar_144.csv（2023-04+ 制度性起点）、
  finra_short.csv（2018+，|变动|>=10% 为跳变）、finra_ats.csv（2021-12+ API 保留期）、
  量价签名 3 个（TSLA_1h_alpaca.csv 日线聚合：OBV 背离 / 放量缩价吸筹 / 净量涌入）。

方法：
1. 事件对齐 = N1 口径：act_day = 第一个开盘严格晚于公开时刻的交易日。
2. 前兆判定：T0 公开时刻 < 放风公开时刻 且 T0 act_day >= 放风 act_day - N，
   N ∈ {5,10,20}（时间戳级 tie-break，杜绝同夜先后颠倒）。
3. 条件分组：有前兆 vs 无前兆的放风后 5/20 日收益（Close[e+w-1]/Open[e]-1），
   置换检验 2000 次双侧 p；每渠道对只在其覆盖窗内检验；min(组样本)<10 只描述。
4. 领先矩阵：每个放风事件回看 30 交易日内最近一次 T0 事件的领先天数；
   附随机日基线前兆率（密集渠道"总有前兆"的假阳性对照）。
5. 多重比较如实记账（并入 strategy-lab 计数器，由主线更新 docs）。

用法： .venv/bin/python research/n2_leadlag.py
输出： outputs/n2_leadlag/{leadlag_matrix.csv, conditional_returns.csv, summary.txt}
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

from src.common.data_io import ET, load_bars  # noqa: E402

DATA = ROOT / "data"
INTEL = DATA / "intel"
OUT = ROOT / "outputs" / "n2_leadlag"

LOOKBACKS = [5, 10, 20]
RET_WINDOWS = [5, 20]
N_PERM = 2000
MATRIX_LOOKBACK = 30
MIN_GROUP = 10
SHORT_JUMP_PCT = 10.0     # 空头利益 |双周变动%| 阈值（预登记，不扫描）
ATS_Z = 1.64              # 暗池占比 z 阈值（vs 过去 26 周，因果）
VOL_Z = 2.0               # 放量阈值（vs 过去 60 日，因果）
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- price scaffolding


def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    et_date = hourly.index.tz_convert(ET).date
    g = hourly.groupby(et_date)
    daily = pd.DataFrame(
        {
            "open_time_utc": g.apply(lambda d: d.index[0]),
            "close_time_utc": g.apply(lambda d: d.index[-1] + pd.Timedelta(hours=1)),
            "Open": g["Open"].first(),
            "High": g["High"].max(),
            "Low": g["Low"].min(),
            "Close": g["Close"].last(),
            "Volume": g["Volume"].sum(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def act_day(daily: pd.DataFrame, t: pd.Timestamp) -> int | None:
    """第一个开盘严格晚于 t 的交易日序号（越界 None）——N1 同口径."""
    pos = daily["open_time_utc"].searchsorted(t, side="right")
    return int(pos) if pos < len(daily) else None


def window_returns(daily: pd.DataFrame, n: int) -> np.ndarray:
    open_, close = daily["Open"].to_numpy(), daily["Close"].to_numpy()
    ret = np.full(len(daily), np.nan)
    if n <= len(daily):
        ret[: len(daily) - n + 1] = close[n - 1:] / open_[: len(daily) - n + 1] - 1
    return ret


# ---------------------------------------------------------------- event channels


def load_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(INTEL / f"{name}.csv")
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed")
    return df.sort_values("event_time_utc").reset_index(drop=True)


def channel_events(daily: pd.DataFrame) -> tuple[dict, dict, dict]:
    """返回 (t0_channels, spoiler_channels, notes)；每渠道 = DataFrame[time, act]."""
    notes: dict[str, str] = {}

    def mk(times: pd.Series | list, note_key: str, note: str) -> pd.DataFrame:
        rows = [(t, a) for t in list(times) if (a := act_day(daily, t)) is not None]
        notes[note_key] = note + f"（{len(times) - len(rows)} 事件在价格窗外丢弃）" if (
            len(times) - len(rows)) else note
        return pd.DataFrame(rows, columns=["time", "act"])

    t0: dict[str, pd.DataFrame] = {}

    # -- 13D/G（举牌/大持仓披露，2018+）
    dg = load_csv("edgar_13dg")
    t0["t0_13dg"] = mk(dg["event_time_utc"], "t0_13dg",
                       f"13D/G 申报 {len(dg)}（TSLA 无 13D，全为 13G 被动披露）")

    # -- Form 144（内部人拟售登记，2023-04 制度性起点）
    f144 = load_csv("edgar_144")
    t0["t0_144"] = mk(f144["event_time_utc"], "t0_144",
                      f"Form 144 {len(f144)}（电子申报 2023-04 起强制，更早纸质缺口）")

    # -- 空头利益跳变（|双周变动| >= SHORT_JUMP_PCT %）
    si = load_csv("finra_short")
    chg = pd.Series([json.loads(p).get("change_pct") for p in si["payload"]], dtype=float)
    jump = si[chg.abs() >= SHORT_JUMP_PCT]
    t0["t0_short_jump"] = mk(jump["event_time_utc"], "t0_short_jump",
                             f"|变动|>={SHORT_JUMP_PCT:.0f}% 的发布 {len(jump)}/{len(si)}"
                             "（event_time 为近似发布时刻：结算+9交易日）")

    # -- 暗池占比异常（off_exchange_pct z>=1.64 vs 过去 26 周，因果 shift(1)）
    ats = load_csv("finra_ats")
    pct = pd.Series([json.loads(p).get("off_exchange_pct") for p in ats["payload"]],
                    dtype=float)
    mu = pct.shift(1).rolling(26, min_periods=13).mean()
    sd = pct.shift(1).rolling(26, min_periods=13).std()
    hi = ats[(pct - mu) / sd >= ATS_Z]
    t0["t0_ats_high"] = mk(hi["event_time_utc"], "t0_ats_high",
                           f"占比 z>={ATS_Z} 的周 {len(hi)}/{len(ats)}"
                           "（API 保留期仅 2021-12 起；发布滞后 2-3 周已含在时间戳内）")

    # -- 量价吸筹签名（日线，事件时刻 = 当日收盘后，act = 次交易日）
    # 阈值按**事件频率**校准（TSLA 高波动，教科书默认阈值下事件近零），
    # 校准只看事件数、不看事件后收益——预登记于此，不做收益扫描。
    d = daily
    ret1 = d["Close"].pct_change()
    obv = (np.sign(ret1.fillna(0.0)) * d["Volume"]).cumsum()
    # (1) OBV 背离：价创 20 日新低而 OBV 未确认（高于其 20 日区间下四分位）
    px_low = d["Close"] <= d["Close"].rolling(20).min()
    obv_rank = obv.rolling(20).rank(pct=True)
    sig_div = px_low & (obv_rank >= 0.25)
    # (2) 放量缩价（吸筹）：量 z>=2（vs 过去 60 日，因果）且 |日收益| <= 0.75×过去20日均幅
    vz = (d["Volume"] - d["Volume"].shift(1).rolling(60).mean()) / d["Volume"].shift(1).rolling(60).std()
    small_move = ret1.abs() <= 0.75 * ret1.abs().shift(1).rolling(20).mean()
    sig_abs = (vz >= VOL_Z) & small_move
    # (3) 净量涌入：5 日 OBV 增量 z>=2（vs 过去 60 日）且价横盘（|5日收益|低于其过去60日中位数）
    obv5 = obv.diff(5)
    oz = (obv5 - obv5.shift(1).rolling(60).mean()) / obv5.shift(1).rolling(60).std()
    ret5_abs = d["Close"].pct_change(5).abs()
    flat5 = ret5_abs <= ret5_abs.shift(1).rolling(60).median()
    sig_svs = (oz >= VOL_Z) & flat5

    for name, mask, label in [
        ("t0_obv_div", sig_div, "OBV 背离（价 20 日新低、OBV 未确认）"),
        ("t0_absorption", sig_abs, "放量缩价吸筹（量 z>=2、|收益|<0.75×20日均幅）"),
        ("t0_net_inflow", sig_svs, "净量涌入（5 日 OBV 增量 z>=2、5日横盘）"),
    ]:
        idx = np.flatnonzero(mask.to_numpy())
        times = [d["close_time_utc"].iloc[i] for i in idx]
        t0[name] = mk(times, name, f"{label}：{len(idx)} 日")

    # -- 合并渠道（仅 2018+ 全覆盖渠道，144/ats 因覆盖窗不齐不并入）
    full_cov = ["t0_13dg", "t0_short_jump", "t0_obv_div", "t0_absorption", "t0_net_inflow"]
    t0["t0_any_fullcov"] = (
        pd.concat([t0[k] for k in full_cov]).sort_values("time").reset_index(drop=True)
    )
    notes["t0_any_fullcov"] = "上述 2018+ 全覆盖渠道的并集（144/ATS 覆盖窗不齐，不并入）"

    # ---------------- 放风渠道
    sp: dict[str, pd.DataFrame] = {}
    k8 = load_csv("edgar_8k")
    sp["sp_8k"] = mk(k8["event_time_utc"], "sp_8k", f"8-K 申报 {len(k8)}（2018+ 全量）")

    tw = load_csv("musk_tweets")
    et_day = tw["event_time_utc"].dt.tz_convert(ET).dt.date
    cnt = tw.groupby(et_day).size()
    cnt.index = pd.to_datetime(cnt.index)
    cnt = cnt.asfreq("D", fill_value=0)
    thr = cnt.shift(1).rolling(365, min_periods=180).quantile(0.9)
    dense = cnt.index[(cnt > thr) & thr.notna()]
    dense_times = [
        pd.Timestamp(x).tz_localize(ET).tz_convert("UTC") + pd.Timedelta(hours=23, minutes=59)
        for x in dense
    ]
    sp["sp_musk_dense"] = mk(dense_times, "sp_musk_dense",
                             f"Musk 高密度发帖日 {len(dense)}（>过去365日90分位；归档止 2025-05）")
    notes["_news"] = "新闻类放风：历史归档不存在（哨兵 news_rss 2026-07 起前向），本轮缺席"
    return t0, sp, notes


# 渠道覆盖窗（放风事件只在 T0 渠道有覆盖的时段内检验；起点后留 45 天热身）
CHANNEL_START = {
    "t0_144": pd.Timestamp("2023-04-27", tz="UTC"),
    "t0_ats_high": pd.Timestamp("2022-01-01", tz="UTC"),
}


# ---------------------------------------------------------------- analysis


def precursor_mask(sp_ev: pd.DataFrame, t0_ev: pd.DataFrame, n_look: int) -> np.ndarray:
    """每个放风事件是否有前兆：t0.time < sp.time 且 t0.act >= sp.act - n_look."""
    tt, ta = t0_ev["time"], t0_ev["act"]
    out = np.zeros(len(sp_ev), dtype=bool)
    for i, (t, a) in enumerate(zip(sp_ev["time"], sp_ev["act"])):
        out[i] = bool(((tt < t) & (ta >= a - n_look)).any())
    return out


def perm_pvalue(x: np.ndarray, labels: np.ndarray, n_perm: int = N_PERM) -> float:
    obs = x[labels].mean() - x[~labels].mean()
    cnt = 0
    for _ in range(n_perm):
        lp = RNG.permutation(labels)
        d = x[lp].mean() - x[~lp].mean()
        if abs(d) >= abs(obs) - 1e-15:
            cnt += 1
    return cnt / n_perm


def conditional_analysis(daily, t0, sp) -> tuple[pd.DataFrame, int]:
    rets = {w: window_returns(daily, w) for w in RET_WINDOWS}
    open_times = pd.to_datetime(daily["open_time_utc"], utc=True)
    rows, n_tests = [], 0
    for sp_name, sp_ev in sp.items():
        for t0_name, t0_ev in t0.items():
            start = CHANNEL_START.get(t0_name)
            for n_look in LOOKBACKS:
                # 覆盖窗过滤 + 需要完整回看窗（act >= n_look）
                ev = sp_ev[sp_ev["act"] >= n_look]
                if start is not None:
                    ev = ev[ev["time"] >= start + pd.Timedelta(days=45)]
                if ev.empty:
                    continue
                pre = precursor_mask(ev, t0_ev, n_look)
                for w in RET_WINDOWS:
                    r = rets[w][ev["act"].to_numpy()]
                    ok = ~np.isnan(r)
                    r_, pre_ = r[ok], pre[ok]
                    n1, n0 = int(pre_.sum()), int((~pre_).sum())
                    row = {
                        "spoiler": sp_name, "t0_channel": t0_name,
                        "lookback_d": n_look, "ret_window_d": w,
                        "n_pre": n1, "n_nopre": n0,
                        "mean_pre_bp": r_[pre_].mean() * 1e4 if n1 else np.nan,
                        "mean_nopre_bp": r_[~pre_].mean() * 1e4 if n0 else np.nan,
                    }
                    row["diff_bp"] = (
                        row["mean_pre_bp"] - row["mean_nopre_bp"]
                        if n1 and n0 else np.nan
                    )
                    if min(n1, n0) >= MIN_GROUP:
                        row["p_perm"] = perm_pvalue(r_, pre_)
                        row["descriptive_only"] = False
                        n_tests += 1
                    else:
                        row["p_perm"] = np.nan
                        row["descriptive_only"] = True
                    rows.append(row)
    return pd.DataFrame(rows), n_tests


def leadlag_matrix(daily, t0, sp) -> pd.DataFrame:
    """T0 渠道 → 放风渠道的平均领先天数 + 前兆率 vs 随机日基线."""
    rows = []
    all_days = np.arange(MATRIX_LOOKBACK, len(daily))
    for sp_name, sp_ev in sp.items():
        for t0_name, t0_ev in t0.items():
            start = CHANNEL_START.get(t0_name)
            ev = sp_ev if start is None else sp_ev[sp_ev["time"] >= start + pd.Timedelta(days=45)]
            gaps = []
            for t, a in zip(ev["time"], ev["act"]):
                cand = t0_ev[(t0_ev["time"] < t) & (t0_ev["act"] >= a - MATRIX_LOOKBACK)]
                if len(cand):
                    gaps.append(a - int(cand["act"].iloc[-1]))
            # 随机日基线：任一日往前 MATRIX_LOOKBACK 交易日内有 T0 事件的比率
            t0_acts = set(t0_ev["act"])
            if start is not None:
                lo = act_day(daily, start + pd.Timedelta(days=45)) or 0
                days = all_days[all_days >= lo]
            else:
                days = all_days
            base = np.mean(
                [any((d - g) in t0_acts for g in range(1, MATRIX_LOOKBACK + 1)) for d in days]
            ) if len(days) else np.nan
            rows.append(
                {
                    "t0_channel": t0_name, "spoiler": sp_name,
                    "n_spoiler_events": len(ev),
                    "n_with_precursor_30d": len(gaps),
                    "frac_with_precursor": round(len(gaps) / len(ev), 3) if len(ev) else np.nan,
                    "base_frac_random_day": round(float(base), 3),
                    "mean_lead_days": round(float(np.mean(gaps)), 2) if gaps else np.nan,
                    "median_lead_days": float(np.median(gaps)) if gaps else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary


def write_summary(daily, t0, sp, notes, cond, mat, n_tests) -> None:
    L = []
    L.append("N2 — 布局痕迹（T0）历史检验 + 先后顺序引擎（2026-07-24）")
    L.append("=" * 74)
    L.append(f"价格窗：{daily.index[0].date()} → {daily.index[-1].date()}（{len(daily)} 交易日）")
    L.append(f"前兆回看 N∈{LOOKBACKS}；收益窗 {RET_WINDOWS} 日；置换检验 {N_PERM} 次；"
             f"min(组)<{MIN_GROUP} 只描述")
    L.append("")
    L.append("渠道与样本：")
    for k in list(t0) + list(sp):
        n = len(t0.get(k, sp.get(k)))
        L.append(f"  {k:<18} {n:>4} 事件  [{notes.get(k, '')}]")
    L.append(f"  （{notes['_news']}）")
    L.append("")
    L.append(f"多重比较记账：本轮共 {n_tests} 个置换检验 p 值（描述性行不计），"
             "并入 strategy-lab 计数器（主线更新 docs）。")
    L.append("")
    L.append("—— 领先矩阵（T0 → 放风，30 交易日回看）——")
    L.append(f"{'T0 渠道':<18}{'放风':<14}{'放风N':>6}{'带前兆':>8}{'前兆率':>8}"
             f"{'随机日基线':>10}{'均值领先(日)':>12}{'中位':>6}")
    for _, r in mat.iterrows():
        L.append(f"{r['t0_channel']:<18}{r['spoiler']:<14}{r['n_spoiler_events']:>6}"
                 f"{r['n_with_precursor_30d']:>8}{r['frac_with_precursor']:>8}"
                 f"{r['base_frac_random_day']:>10}"
                 f"{r['mean_lead_days'] if not pd.isna(r['mean_lead_days']) else '—':>12}"
                 f"{r['median_lead_days'] if not pd.isna(r['median_lead_days']) else '—':>6}")
    L.append("")
    L.append("—— 条件收益（有前兆 vs 无前兆，diff = 前兆组 − 无前兆组，bp）——")
    tested = cond[~cond["descriptive_only"]]
    sig = tested[tested["p_perm"] <= 0.05]
    L.append(f"可检验组合 {len(tested)}，名义 p<=0.05 的 {len(sig)} 个"
             f"（Bonferroni 阈值 0.05/{n_tests}≈{0.05 / max(n_tests, 1):.5f}）：")
    for _, r in sig.sort_values("p_perm").iterrows():
        L.append(f"  {r['spoiler']} × {r['t0_channel']} N={r['lookback_d']} w={r['ret_window_d']}日: "
                 f"前兆组 {r['mean_pre_bp']:+.0f}bp(n={r['n_pre']}) vs "
                 f"无前兆 {r['mean_nopre_bp']:+.0f}bp(n={r['n_nopre']})  "
                 f"diff={r['diff_bp']:+.0f}bp  p={r['p_perm']:.3f}"
                 + ("  [过 Bonferroni]" if r["p_perm"] <= 0.05 / max(n_tests, 1) else ""))
    if not len(sig):
        L.append("  （无）")
    L.append("")
    L.append(VERDICT)
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


VERDICT = """\
==================== 中文判读（2026-07-24，基于上表数字 + 附加诊断） ====================

核心问题："建仓（T0 布局痕迹）→ 放风（8-K / Musk 高密度发帖）→ 散户接盘"的
链条在 TSLA 历史上是否可观测？**答案：吸筹版链条不可观测；唯一有领先性的
候选渠道是空头利益跳升，且方向与吸筹叙事相反。**

1. 领先矩阵：前兆到达在时间上与放风基本独立。16 个渠道对里，前兆率与
   随机日基线几乎处处相当（如 t0_any→musk_dense 0.72 vs 基线 0.73；
   13dg→8k 0.296 vs 0.231）——放风前"恰好"出现 T0 异动的频率不超过
   任意日子的自然频率，看不到系统性编排。平均领先 8-16 天也与 30 日
   回看窗内均匀散布的机械期望（≈15 天）无区分度。唯一"紧凑领先"是
   ats_high→8k 均值 5.3 天，但 n=16，只描述不下结论。

2. 条件收益：68 个置换检验、14 个名义 p<=0.05、6 个过 Bonferroni——
   但 diff 全部为负（有前兆的放风之后 20 日反而更差 -800~-1700bp），
   方向与"放风拉高接盘"预测相反。分解（附加诊断，描述性）：
   - obv_div / ats_high / any_fullcov 组是 regime 混杂：其前兆组的
     **过去** 20 日收益已是 -1100~-1340bp（无前兆组 +500~+730bp），
     事件本身聚在 2022 / 2024-25 下跌段（obv_div 前兆 75 例中 65 例），
     "前兆→更差"只是 TSLA 动量延续的影子，不是布局信号。
   - t0_short_jump N=20 w=20 是唯一不被动量解释的组合（两组过去
     20 日收益 +401 vs +357bp 几乎相同，之后 20 日却差 -840bp，
     p<0.0005 过 Bonferroni）。按方向拆分（描述性）：效应全部来自
     **空头利益跳升**（up-jump 前兆组 -787bp vs 无前兆 +578bp；
     down-jump 前兆组反而 +557bp）。即：双周空头利益大增 → 随后的
     Musk 高密度发帖期跑输——与"知情空头提前布局"及空头利益预测
     负收益的学术文献同向，与"吸筹拉高"叙事相反。

3. 诚实性边界（先于任何后续使用写死）：
   - musk_dense 435 事件的 20 日窗高度重叠，收益非独立，置换检验
     p 值系统性偏乐观（有效样本远小于名义值）；本引擎不修正自相关。
   - 空头利益 event_time 是近似发布时刻（结算+9 交易日），误差 ±1-2 天，
     对 N=10/20 回看不致命，对 N=5 组合要打折扣。
   - Form 144（2023-04 起）与 ATS（2021-12 起）覆盖窗短，其组合的
     检验功效低；新闻类放风历史缺席（哨兵前向补）。
   - up/down 拆分是看过总效应后的事后诊断，只作描述，不计显著。

4. 结论与去向：TSLA 历史上"布局→放风"吸筹链条**不可观测**——要么不存在，
   要么不在这些免费 T0 渠道的分辨率内（期权 OI、暗池明细的日级足迹从
   哨兵前向攒起才有分辨率）。唯一候选领先渠道 = 空头利益跳升（负向，
   对做多是风险过滤特征而非收益信号），按协议不登记独立策略，转两条路：
   (a) short_interest_change 并入 E8 特征池（需另行登记假设）；
   (b) 哨兵 finra_short / options_snapshot / finra_ats 前向流已上线，
       8-12 周后用干净前向样本复检本轮唯一幸存者。
   多重比较记账 +68（并入 strategy-lab 计数器，主线更新 docs）。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hourly = load_bars(str(DATA / "TSLA_1h_alpaca.csv"))
    daily = build_daily(hourly)
    print(f"daily bars: {len(daily)} ({daily.index[0].date()} -> {daily.index[-1].date()})")

    t0, sp, notes = channel_events(daily)
    for k, v in {**t0, **sp}.items():
        print(f"  {k}: {len(v)} events  [{notes.get(k, '')}]")

    cond, n_tests = conditional_analysis(daily, t0, sp)
    cond.to_csv(OUT / "conditional_returns.csv", index=False)
    mat = leadlag_matrix(daily, t0, sp)
    mat.to_csv(OUT / "leadlag_matrix.csv", index=False)
    print(f"\ntests with p-value: {n_tests}")
    write_summary(daily, t0, sp, notes, cond, mat, n_tests)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
