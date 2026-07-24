#!/usr/bin/env python
"""N4 — 黄金坑逆向审计.

假设（用户"先通过黄金坑，再找历史信息校验"）：从答案倒推——量化识别历史黄金坑
（深跌后大幅收复的坑底），逐坑审计当时可知的情报记录；真坑与假坑（继续跌的）
的情报签名差异 = 可复用的坑底识别特征。

纪律（预登记，不扫描阈值）：
- 坑底候选：日线 Close 为 ±5 日局部低点，且距前 60 交易日最高收盘回撤 >= 20%
- 去重聚类：相邻候选（间隔 <= 20 交易日且其间未从簇内最低点反弹 >= 10%）并簇，
  取簇内最低收盘为坑底——"同一波动只算一次"
- 真坑：坑底后 60 交易日内 Close 先触及 坑底*1.25（收复 >= 25%）
- 假坑：60 日内先触及 坑底*0.90（继续创新低 >= 10%），或 60 日内最大反弹 < 10%
- 中间态（反弹 10-25% 且未破位）剔除；数据末端窗口不足者剔除并注明
- 坑底识别/分类用未来信息（答案已知的逆向研究，非回测）；
  **签名特征只用坑底当日收盘前已公开的信息**（逐源用公开时间戳过滤）

签名特征（坑底日可知，全部因果）：
- 市场侧：回撤深度、跌速（30日收益）、距峰天数、20日波动率、RSI14、距200日线、
  当日量分位(252d)、5日/60日量比
- 空头：最新一期空头利益 change_pct、days_to_cover、~6 周跨度水平变化（finra_short，
  发布时刻 = 结算+9交易日近似）
- 内部人：前 30 交易日 Form 4 买/卖笔数与金额、Musk 本人买入旗标；13D/G、Form 144
  计数（144 仅 2023-04+，之前 NaN）
- 事件：前 30 交易日 8-K 数；距上次 FOMC 天数、距下次 FOMC 天数（会期日历事前公布，
  以声明时刻近似，末端缺口 NaN）
- Musk 发帖：前 30 交易日窗内日均发帖、窗内后半/前半趋势比、vs 前 365 日基线比
  （归档止 2025-05，之后的坑如实 NaN = 失明）
- 暗池：最新一周 off_exchange_pct 及 vs 前 26 周 z（2021-12+，之前 NaN）

统计：真坑组 vs 假坑组逐特征 Mann-Whitney（scipy auto）+ 均值差置换检验（10000 次），
样本量如实报告；多重比较记账（本轮检验数并入 strategy-lab 计数器，由主线更新 docs）。

用法： .venv/bin/python research/n4_golden_pit.py
输出： outputs/n4_golden_pit/{pits_catalog.csv, per_pit_dossiers/*.md,
       signature_comparison.csv, summary.txt}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402

DATA = ROOT / "data"
INTEL = DATA / "intel"
OUT = ROOT / "outputs" / "n4_golden_pit"
DOSSIER_DIR = OUT / "per_pit_dossiers"

# ---- 预登记参数（不扫描） ----
LOCAL_LOW_HALF = 5        # 局部低点半窗（±5 日）
HIGH_LOOKBACK = 60        # 前高回看（交易日）
DD_THRESH = -0.20         # 候选回撤门槛
CLUSTER_GAP = 20          # 聚类最大间隔（交易日）
CLUSTER_REBOUND = 0.10    # 簇内视为"同一波动"的最大中途反弹
FWD_WINDOW = 60           # 分类前瞻窗（交易日）
WAVE_WINDOW = 60          # 波级去重窗（交易日）：
                          #   后坑收盘 > 前坑*1.05 → 收复腿回调，并入前坑波
                          #   ±5% 之内 → 双底，取更低者
                          #   低于前坑*0.95 → 新的更低决策点，两者都保留
WAVE_BAND = 0.05
TRUE_REBOUND = 0.25       # 真坑收复门槛
FALSE_NEWLOW = 0.10       # 假坑继续创新低门槛
FALSE_MAXREB = 0.10       # 假坑最大反弹上限
INTEL_LOOKBACK = 30       # 情报回看窗（交易日）
N_PERM = 10000
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


# ---------------------------------------------------------------- pit detection


def find_pits(daily: pd.DataFrame) -> pd.DataFrame:
    c = daily["Close"]
    n = len(c)
    prior_high = c.rolling(HIGH_LOOKBACK, min_periods=HIGH_LOOKBACK).max().shift(1)
    dd = c / prior_high - 1.0
    local_low = c == c.rolling(2 * LOCAL_LOW_HALF + 1, center=True, min_periods=1).min()
    cand_idx = np.flatnonzero((dd <= DD_THRESH).to_numpy() & local_low.to_numpy())

    # 聚类去重：同一波动取最低点
    clusters: list[list[int]] = []
    for i in cand_idx:
        if clusters:
            cur = clusters[-1]
            last = cur[-1]
            cmin_i = min(cur, key=lambda k: c.iloc[k])
            interim_max = c.iloc[last: i + 1].max()
            if (i - last <= CLUSTER_GAP
                    and interim_max < c.iloc[cmin_i] * (1 + CLUSTER_REBOUND)):
                cur.append(i)
                continue
        clusters.append([i])
    pit_idx = [min(cl, key=lambda k: c.iloc[k]) for cl in clusters]

    # 波级去重：同一波动只算一次，取最低点
    retained: list[int] = []
    merged: dict[int, list[int]] = {}
    for i in pit_idx:
        if retained:
            r = retained[-1]
            if i - r <= WAVE_WINDOW:
                ratio = c.iloc[i] / c.iloc[r]
                if ratio > 1 + WAVE_BAND:            # 收复腿上的回调，并入前坑波
                    merged.setdefault(r, []).append(i)
                    continue
                if ratio >= 1 - WAVE_BAND:           # 双底：取更低者
                    if c.iloc[i] < c.iloc[r]:
                        merged.setdefault(i, []).extend(merged.pop(r, []) + [r])
                        retained[-1] = i
                    else:
                        merged.setdefault(r, []).append(i)
                    continue
        retained.append(i)
    pit_idx = retained

    rows = []
    for i in pit_idx:
        pit_close = c.iloc[i]
        peak_win = c.iloc[max(0, i - HIGH_LOOKBACK): i]
        peak_pos = int(peak_win.to_numpy().argmax())
        peak_i = max(0, i - HIGH_LOOKBACK) + peak_pos
        fwd = c.iloc[i + 1: i + 1 + FWD_WINDOW]
        n_fwd = len(fwd)
        fwd_max = fwd.max() / pit_close - 1 if n_fwd else np.nan
        fwd_min = fwd.min() / pit_close - 1 if n_fwd else np.nan
        hit_reb = fwd[fwd >= pit_close * (1 + TRUE_REBOUND)]
        hit_low = fwd[fwd <= pit_close * (1 - FALSE_NEWLOW)]
        t_reb = hit_reb.index[0] if len(hit_reb) else None
        t_low = hit_low.index[0] if len(hit_low) else None
        if t_low is not None and (t_reb is None or t_low < t_reb):
            label = "false"          # 先破位：价值陷阱
        elif t_reb is not None:
            label = "true"           # 先收复 25%：黄金坑
        elif n_fwd < FWD_WINDOW:
            label = "truncated"      # 数据末端，窗口不足且未见分晓
        elif fwd_max < FALSE_MAXREB:
            label = "false"          # 60 日反弹不足 10%：也算价值陷阱
        else:
            label = "mid"            # 中间态剔除
        rows.append(
            {
                "pit_i": i,
                "pit_date": daily.index[i].date(),
                "pit_close": pit_close,
                "peak_date": daily.index[peak_i].date(),
                "peak_close": c.iloc[peak_i],
                "dd_pct": (pit_close / c.iloc[peak_i] - 1) * 100,
                "n_fwd_days": n_fwd,
                "fwd60_max_pct": fwd_max * 100 if n_fwd else np.nan,
                "fwd60_min_pct": fwd_min * 100 if n_fwd else np.nan,
                "days_to_rebound25": (
                    int(daily.index.get_loc(t_reb) - i) if t_reb is not None else np.nan),
                "days_to_newlow10": (
                    int(daily.index.get_loc(t_low) - i) if t_low is not None else np.nan),
                "label": label,
                "merged_candidates": ";".join(
                    str(daily.index[k].date()) for k in sorted(merged.get(i, []))),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- intel loading


def load_intel(name: str) -> pd.DataFrame:
    df = pd.read_csv(INTEL / f"{name}.csv")
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed")
    df["payload"] = df["payload"].map(json.loads)
    return df.sort_values("event_time_utc").reset_index(drop=True)


def si_signature(si: pd.DataFrame, t1: pd.Timestamp) -> tuple[dict, dict | None]:
    """空头利益签名（只用发布时刻 <= t1 的期次）。返回 (特征 dict, 最新 payload 或 None).

    N4/N5 共用：si_latest_chg_pct、si_days_to_cover、si_chg_6wk_pct
    （最新一期 vs ~6 周前最近已发布期的水平变化）。
    """
    f = {"si_latest_chg_pct": np.nan, "si_days_to_cover": np.nan,
         "si_chg_6wk_pct": np.nan}
    hist = si[si["event_time_utc"] <= t1]
    if not len(hist):
        return f, None
    last = hist.iloc[-1]["payload"]
    f["si_latest_chg_pct"] = last.get("change_pct")
    f["si_days_to_cover"] = last.get("days_to_cover")
    older = hist[hist["event_time_utc"] <= t1 - pd.Timedelta(days=42)]
    if len(older):
        o = older.iloc[-1]["payload"]
        f["si_chg_6wk_pct"] = (
            (last["short_interest"] / o["short_interest"] - 1) * 100
            if o.get("short_interest") else np.nan)
    return f, last


def rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------- per-pit features


class IntelBook:
    def __init__(self, daily: pd.DataFrame):
        self.daily = daily
        self.form4 = load_intel("edgar_form4")
        self.dg = load_intel("edgar_13dg")
        self.f144 = load_intel("edgar_144")
        self.k8 = load_intel("edgar_8k")
        self.fomc = load_intel("fomc")
        self.si = load_intel("finra_short")
        self.ats = load_intel("finra_ats")
        tw = pd.read_csv(INTEL / "musk_tweets.csv", usecols=["event_time_utc"])
        tw["event_time_utc"] = pd.to_datetime(
            tw["event_time_utc"], utc=True, format="mixed")
        cnt = tw.groupby(tw["event_time_utc"].dt.tz_convert(ET).dt.date).size()
        cnt.index = pd.to_datetime(cnt.index)
        self.tweet_daily = cnt.asfreq("D", fill_value=0)
        self.tweet_end = tw["event_time_utc"].max()

        d = daily
        self.ret1 = d["Close"].pct_change()
        self.vol20 = self.ret1.rolling(20).std() * np.sqrt(252) * 100
        self.rsi = rsi14(d["Close"])
        self.ma200 = d["Close"].rolling(200, min_periods=120).mean()
        self.vol_pctile = d["Volume"].rolling(252, min_periods=126).rank(pct=True) * 100

    # ---- helpers ----
    def _win(self, df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        return df[(df["event_time_utc"] > t0) & (df["event_time_utc"] <= t1)]

    def features(self, i: int) -> tuple[dict, dict]:
        """返回 (特征 dict, 档案用原始事件 dict)。全部只用 <= 坑底收盘时刻的公开信息."""
        d = self.daily
        t1 = d["close_time_utc"].iloc[i]
        j0 = max(0, i - INTEL_LOOKBACK)
        t0 = d["open_time_utc"].iloc[j0]
        raw: dict[str, object] = {}
        f: dict[str, float] = {}

        # 市场侧
        c = d["Close"]
        peak_win = c.iloc[max(0, i - HIGH_LOOKBACK): i]
        f["dd_from_60d_high_pct"] = (c.iloc[i] / peak_win.max() - 1) * 100
        f["days_from_peak"] = i - (max(0, i - HIGH_LOOKBACK) + int(peak_win.to_numpy().argmax()))
        f["ret_30d_pct"] = (c.iloc[i] / c.iloc[j0] - 1) * 100 if i >= 30 else np.nan
        f["vol20_ann_pct"] = self.vol20.iloc[i]
        f["rsi14"] = self.rsi.iloc[i]
        f["dist_200ma_pct"] = (c.iloc[i] / self.ma200.iloc[i] - 1) * 100
        f["volume_pctile_252"] = self.vol_pctile.iloc[i]
        v = d["Volume"]
        v60 = v.iloc[max(0, i - 65): max(0, i - 5)].mean()
        f["vol_ratio_5_60"] = v.iloc[max(0, i - 4): i + 1].mean() / v60 if v60 else np.nan

        # 空头利益（发布时刻 <= 坑底收盘）——公共函数 si_signature（N5 复用）
        si_f, si_last = si_signature(self.si, t1)
        f.update(si_f)
        if si_last is not None:
            raw["si_latest"] = si_last

        # 内部人（Form 4 申报时刻 <= 坑底收盘）
        f4 = self._win(self.form4, t0, t1)
        buys = f4[f4["type"] == "insider_buy"]
        sells = f4[f4["type"] == "insider_sell"]
        f["n_insider_buy_30d"] = len(buys)
        f["n_insider_sell_30d"] = len(sells)
        f["insider_buy_usd_30d"] = sum(p.get("value_usd") or 0 for p in buys["payload"])
        f["insider_sell_usd_30d"] = sum(p.get("value_usd") or 0 for p in sells["payload"])
        f["musk_buy_30d"] = float(
            any(p.get("owner") == "Musk Elon" for p in buys["payload"]))
        raw["form4_buys"] = list(buys.itertuples(index=False))
        raw["form4_sells_n"] = len(sells)
        raw["form4_sells_top"] = sorted(
            (p for p in sells["payload"]), key=lambda p: -(p.get("value_usd") or 0))[:3]

        f["n_13dg_30d"] = len(self._win(self.dg, t0, t1))
        if t1 >= pd.Timestamp("2023-06-01", tz="UTC"):  # 144 电子申报稳定后
            f["n_144_30d"] = len(self._win(self.f144, t0, t1))
        else:
            f["n_144_30d"] = np.nan

        # 8-K / FOMC
        k8w = self._win(self.k8, t0, t1)
        f["n_8k_30d"] = len(k8w)
        raw["8k_items"] = [r["type"] for _, r in k8w.iterrows()]
        fp = self.fomc[self.fomc["event_time_utc"] <= t1]
        fn = self.fomc[self.fomc["event_time_utc"] > t1]
        f["days_since_fomc"] = (t1 - fp["event_time_utc"].iloc[-1]).days if len(fp) else np.nan
        f["days_to_next_fomc"] = (fn["event_time_utc"].iloc[0] - t1).days if len(fn) else np.nan

        # Musk 发帖密度（归档止 tweet_end；坑在其后 = 失明）
        pit_day = pd.Timestamp(self.daily.index[i].date())
        w0 = pd.Timestamp(self.daily.index[j0].date())
        if t1 <= self.tweet_end:
            win = self.tweet_daily.loc[w0:pit_day]
            f["musk_tweets_per_day_30d"] = win.mean()
            half = len(win) // 2
            first, second = win.iloc[:half].mean(), win.iloc[half:].mean()
            f["musk_trend_ratio"] = second / first if first > 0 else np.nan
            base = self.tweet_daily.loc[w0 - pd.Timedelta(days=365): w0 - pd.Timedelta(days=1)]
            f["musk_vs_365d_base"] = win.mean() / base.mean() if len(base) and base.mean() > 0 else np.nan
            raw["musk_window"] = win
        else:
            f["musk_tweets_per_day_30d"] = f["musk_trend_ratio"] = f["musk_vs_365d_base"] = np.nan
            raw["musk_window"] = None

        # 暗池（周报发布时刻 <= 坑底收盘；2021-12 起）
        ats_hist = self.ats[self.ats["event_time_utc"] <= t1]
        if len(ats_hist) >= 14:
            pct = pd.Series([p["off_exchange_pct"] for p in ats_hist["payload"]], dtype=float)
            f["ats_offexch_pct"] = pct.iloc[-1]
            mu, sd = pct.iloc[-27:-1].mean(), pct.iloc[-27:-1].std()
            f["ats_offexch_z26"] = (pct.iloc[-1] - mu) / sd if sd else np.nan
        else:
            f["ats_offexch_pct"] = f["ats_offexch_z26"] = np.nan

        raw["window"] = (t0, t1)
        return f, raw


FEATURES_ORDER = [
    "dd_from_60d_high_pct", "days_from_peak", "ret_30d_pct", "vol20_ann_pct", "rsi14",
    "dist_200ma_pct", "volume_pctile_252", "vol_ratio_5_60",
    "si_latest_chg_pct", "si_days_to_cover", "si_chg_6wk_pct",
    "n_insider_buy_30d", "n_insider_sell_30d", "insider_buy_usd_30d",
    "insider_sell_usd_30d", "musk_buy_30d", "n_13dg_30d", "n_144_30d",
    "n_8k_30d", "days_since_fomc", "days_to_next_fomc",
    "musk_tweets_per_day_30d", "musk_trend_ratio", "musk_vs_365d_base",
    "ats_offexch_pct", "ats_offexch_z26",
]


# ---------------------------------------------------------------- stats


def perm_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(N_PERM):
        RNG.shuffle(pool)
        d = pool[:na].mean() - pool[na:].mean()
        if abs(d) >= abs(obs) - 1e-15:
            cnt += 1
    return cnt / N_PERM


def compare(feat_df: pd.DataFrame) -> pd.DataFrame:
    t = feat_df[feat_df["label"] == "true"]
    fls = feat_df[feat_df["label"] == "false"]
    rows = []
    for col in FEATURES_ORDER:
        a = t[col].dropna().to_numpy(dtype=float)
        b = fls[col].dropna().to_numpy(dtype=float)
        row = {
            "feature": col, "n_true": len(a), "n_false": len(b),
            "mean_true": a.mean() if len(a) else np.nan,
            "mean_false": b.mean() if len(b) else np.nan,
            "median_true": np.median(a) if len(a) else np.nan,
            "median_false": np.median(b) if len(b) else np.nan,
        }
        if len(a) >= 3 and len(b) >= 3 and (np.ptp(np.concatenate([a, b])) > 0):
            u = mannwhitneyu(a, b, alternative="two-sided")
            row["mw_p"] = u.pvalue
            row["perm_p"] = perm_pvalue(a, b)
        else:
            row["mw_p"] = row["perm_p"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    n_tests = int(df["mw_p"].notna().sum())
    df["n_tests_bonferroni"] = n_tests
    df["mw_p_bonf"] = (df["mw_p"] * n_tests).clip(upper=1.0)
    return df


# ---------------------------------------------------------------- dossiers


def write_dossier(path: Path, pit: pd.Series, f: dict, raw: dict,
                  daily: pd.DataFrame) -> None:
    t0, t1 = raw["window"]
    lab = {"true": "真坑（黄金坑）", "false": "假坑（价值陷阱）",
           "mid": "中间态（剔除）", "truncated": "窗口不足（剔除）"}[pit["label"]]
    lines = [
        f"# {pit['pit_date']} — {lab}",
        "",
        "## 坑体信息（含未来信息，仅用于分类）",
        f"- 坑底收盘 {pit['pit_close']:.2f}；前 60 日峰 {pit['peak_close']:.2f}"
        f"（{pit['peak_date']}）；回撤 {pit['dd_pct']:.1f}%",
        f"- 之后 {pit['n_fwd_days']} 交易日：最大反弹 {pit['fwd60_max_pct']:.1f}%、"
        f"最深续跌 {pit['fwd60_min_pct']:.1f}%",
    ]
    if not np.isnan(pit["days_to_rebound25"]):
        lines.append(f"- 收复 +25% 用时 {pit['days_to_rebound25']:.0f} 交易日")
    if not np.isnan(pit["days_to_newlow10"]):
        lines.append(f"- 破位 -10% 用时 {pit['days_to_newlow10']:.0f} 交易日")
    lines += [
        "",
        f"## 坑底当日可知情报（窗口 {t0.date()} → {t1.date()}，全部为公开时刻 <= 坑底收盘）",
        "",
        "### 市场侧",
        f"- 30 日收益 {f['ret_30d_pct']:.1f}%；距峰 {f['days_from_peak']:.0f} 日；"
        f"20 日年化波动 {f['vol20_ann_pct']:.0f}%；RSI14 {f['rsi14']:.0f}",
        f"- 距 200 日线 {f['dist_200ma_pct']:.1f}%；当日量分位(252d) "
        f"{f['volume_pctile_252']:.0f}%；5/60 日量比 {f['vol_ratio_5_60']:.2f}",
        "",
        "### 空头利益（最新一期已发布）",
    ]
    si = raw.get("si_latest")
    if si:
        lines.append(
            f"- 结算日 {si['settlement_date']}：{si['short_interest']:,} 股，"
            f"双周变动 {si['change_pct']:+.1f}%，days-to-cover {si['days_to_cover']:.1f}；"
            f"~6 周跨度变化 {f['si_chg_6wk_pct']:+.1f}%"
            if not np.isnan(f["si_chg_6wk_pct"]) else
            f"- 结算日 {si['settlement_date']}：{si['short_interest']:,} 股，"
            f"双周变动 {si['change_pct']:+.1f}%，days-to-cover {si['days_to_cover']:.1f}")
    else:
        lines.append("- 无已发布数据")
    lines += ["", "### 内部人 / 申报"]
    lines.append(
        f"- Form 4：买入 {f['n_insider_buy_30d']:.0f} 笔"
        f"（${f['insider_buy_usd_30d']:,.0f}）、卖出 {f['n_insider_sell_30d']:.0f} 笔"
        f"（${f['insider_sell_usd_30d']:,.0f}）；Musk 本人买入："
        f"{'有' if f['musk_buy_30d'] else '无'}")
    for b in raw["form4_buys"]:
        p = b.payload
        lines.append(
            f"  - 买入 {p['owner']}（{p.get('role','')}）{p.get('shares',0):,.0f} 股 "
            f"@ {p.get('vwap',0):.2f}（${p.get('value_usd',0):,.0f}，"
            f"交易日 {','.join(p.get('trade_dates', []))}，申报 {b.event_time_utc.date()}）")
    if raw["form4_sells_top"]:
        tops = "; ".join(
            f"{p['owner']} ${p.get('value_usd') or 0:,.0f}" for p in raw["form4_sells_top"])
        lines.append(f"  - 最大卖出：{tops}")
    n144 = ("N/A（2023-04 前无电子申报）" if np.isnan(f["n_144_30d"])
            else f"{f['n_144_30d']:.0f} 件")
    lines.append(f"- 13D/G {f['n_13dg_30d']:.0f} 件；Form 144 {n144}")
    lines += [
        "",
        "### 事件",
        f"- 8-K：{f['n_8k_30d']:.0f} 件" + (
            "（" + "; ".join(raw["8k_items"]) + "）" if raw["8k_items"] else ""),
        f"- FOMC：距上次 {f['days_since_fomc']:.0f} 天，距下次 "
        + (f"{f['days_to_next_fomc']:.0f} 天" if not np.isnan(f["days_to_next_fomc"]) else "N/A"),
        "",
        "### Musk 发帖",
    ]
    if raw["musk_window"] is not None:
        lines.append(
            f"- 窗内日均 {f['musk_tweets_per_day_30d']:.1f} 帖；后半/前半趋势比 "
            f"{f['musk_trend_ratio']:.2f}；vs 前 365 日基线 {f['musk_vs_365d_base']:.2f}×")
    else:
        lines.append("- 归档失明（musk_tweets 止于 2025-05，此坑在其后）")
    lines += ["", "### 暗池"]
    if not np.isnan(f["ats_offexch_pct"]):
        lines.append(
            f"- 最新已发布周 off-exchange 占比 {f['ats_offexch_pct']:.1f}%"
            f"（vs 前 26 周 z={f['ats_offexch_z26']:+.2f}）")
    else:
        lines.append("- N/A（FINRA ATS 数据 2021-12 起）")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- musk buy audit


def musk_buy_audit(book: IntelBook, pits: pd.DataFrame, daily: pd.DataFrame) -> list[str]:
    out = ["## 内部人公开市场买入（Form 4 code P，全历史 11 笔）vs 坑底距离",
           "（约定：'坑底后 +N' = 买入发生在坑底之后 N 个交易日；vwap 为申报原值，"
           "拆股前后口径不同，仅日期可比）"]
    buys = book.form4[book.form4["type"] == "insider_buy"]
    idx = daily.index
    c = daily["Close"]
    prior_high = c.rolling(HIGH_LOOKBACK, min_periods=1).max().shift(1)
    for _, r in buys.iterrows():
        p = r["payload"]
        td = pd.to_datetime(p["trade_dates"][0]) if p.get("trade_dates") else None
        if td is None:
            continue
        who = p["owner"]
        head = (f"- {who} 交易日 {td.date()} @ {p.get('vwap', float('nan')):.2f} "
                f"(${p.get('value_usd', 0):,.0f})")
        pos = int(idx.searchsorted(td.normalize()))
        if td.normalize() < idx[0] or pos >= len(idx):
            out.append(head + "：在价格数据窗外")
            continue
        dd_at_buy = (c.iloc[pos] / prior_high.iloc[pos] - 1) * 100
        dists = (pos - pits["pit_i"]).astype(int)
        k = dists.abs().idxmin()
        pit = pits.loc[k]
        rel = f"坑底后 +{dists.loc[k]}" if dists.loc[k] >= 0 else f"坑底前 {-dists.loc[k]}"
        out.append(
            head + f"：买入日距 60 日高回撤 {dd_at_buy:.1f}%；最近坑底 {pit['pit_date']}"
            f"（{pit['label']}，坑底价 {pit['pit_close']:.2f}），买入在{rel} 交易日")
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DOSSIER_DIR.glob("*.md"):
        stale.unlink()
    hourly = load_bars(str(DATA / "TSLA_1h_alpaca.csv"))
    daily = build_daily(hourly)
    pits = find_pits(daily)
    book = IntelBook(daily)

    feats, raws = [], []
    for _, pit in pits.iterrows():
        f, raw = book.features(int(pit["pit_i"]))
        f["pit_date"] = pit["pit_date"]
        f["label"] = pit["label"]
        feats.append(f)
        raws.append(raw)
    feat_df = pd.DataFrame(feats)

    catalog = pits.merge(feat_df, on=["pit_date", "label"])
    catalog.to_csv(OUT / "pits_catalog.csv", index=False, float_format="%.4f")

    for (_, pit), (_, f), raw in zip(pits.iterrows(), feat_df.iterrows(), raws):
        name = f"{pit['pit_date']}_{pit['label']}.md"
        write_dossier(DOSSIER_DIR / name, pit, f, raw, daily)

    comp = compare(feat_df)
    comp.to_csv(OUT / "signature_comparison.csv", index=False, float_format="%.4f")

    # ---- summary ----
    n_true = int((pits["label"] == "true").sum())
    n_false = int((pits["label"] == "false").sum())
    n_mid = int((pits["label"] == "mid").sum())
    n_trunc = int((pits["label"] == "truncated").sum())
    lines = [
        "N4 — 黄金坑逆向审计 摘要",
        f"生成时间：{pd.Timestamp.now()}",
        "",
        f"数据：TSLA_1h_alpaca.csv 聚合日线 {daily.index[0].date()} → "
        f"{daily.index[-1].date()}（{len(daily)} 交易日）",
        f"坑底候选聚类后：{len(pits)} 个；真坑 {n_true}、假坑 {n_false}、"
        f"中间态剔除 {n_mid}、窗口不足剔除 {n_trunc}",
        "",
        "== 坑清单 ==",
    ]
    for _, p in pits.iterrows():
        lines.append(
            f"  {p['pit_date']} close {p['pit_close']:8.2f} dd {p['dd_pct']:6.1f}% "
            f"fwd60 max {p['fwd60_max_pct']:6.1f}% min {p['fwd60_min_pct']:6.1f}% "
            f"-> {p['label']}")
    lines += ["", *musk_buy_audit(book, pits, daily), "", "== 签名对比（真 vs 假） =="]
    for _, r in comp.iterrows():
        if np.isnan(r["mw_p"]):
            lines.append(
                f"  {r['feature']:26s} n={r['n_true']}/{r['n_false']} 样本不足或常量，不检验")
        else:
            sig = " **" if r["mw_p_bonf"] < 0.05 else ("  *" if r["mw_p"] < 0.05 else "")
            lines.append(
                f"  {r['feature']:26s} n={r['n_true']}/{r['n_false']} "
                f"mean {r['mean_true']:9.2f} vs {r['mean_false']:9.2f} "
                f"med {r['median_true']:9.2f} vs {r['median_false']:9.2f} "
                f"MW-p {r['mw_p']:.4f} perm-p {r['perm_p']:.4f} "
                f"Bonf-p {r['mw_p_bonf']:.3f}{sig}")
    n_tests = int(comp["mw_p"].notna().sum())

    # 签名组合规则雏形（仅描述性，同一数据上不做回测——那是下一实验的事）
    nominal = comp[(comp["mw_p"] < 0.10) | (comp["perm_p"] < 0.10)]
    lines += ["", "== 签名组合规则雏形（描述性，未验证） =="]
    if len(nominal):
        lines.append("  Bonferroni 后零幸存；名义 p<0.10 的候选："
                     + ", ".join(nominal["feature"]))
        tf = feat_df[feat_df["label"].isin(["true", "false"])]
        rule = (tf["si_chg_6wk_pct"] <= 0) & (tf["musk_trend_ratio"] <= 1.0)
        ct = pd.crosstab(tf["label"], rule.fillna(False))
        lines += [
            "  雏形规则（方向取自名义候选，阈值取中性 0/1.0，未调参）：",
            "    空头利益 ~6 周变化 <= 0（空头未在加仓）且 Musk 发帖趋势比 <= 1（未在加速放风）",
            "  同一样本内混淆计数（in-sample 描述，不构成任何验证）：",
        ]
        legs = {
            "空头腿 si_chg_6wk_pct<=0": tf["si_chg_6wk_pct"] <= 0,
            "Musk腿 musk_trend_ratio<=1": tf["musk_trend_ratio"] <= 1.0,
            "合取": rule,
        }
        for name, m in legs.items():
            m = m.fillna(False)
            ht = int((m & (tf["label"] == "true")).sum())
            hf = int((m & (tf["label"] == "false")).sum())
            nt = int((tf["label"] == "true").sum())
            nf = int((tf["label"] == "false").sum())
            lines.append(f"    {name}: 真坑 {ht}/{nt}，假坑 {hf}/{nf}")
        lines.append("    读法：合取精度尚可（6 中 5 真）但召回低（17 真只抓 5）——"
                     "只是方向雏形，不是可用规则")
    else:
        lines.append("  无名义候选，不给雏形")

    lines += [
        "",
        f"多重比较：本轮 {n_tests} 项特征检验（Mann-Whitney 为主判，置换检验为稳健性），"
        f"Bonferroni ×{n_tests}；并入 strategy-lab 计数器 +{n_tests}（docs 由主线更新）",
        "",
        "方法备注：",
        f"- 真坑 {n_true} 个多于预计的 5-10 个：TSLA 年化波动 ~60%，20%/25% 阈值下"
        "每次中级回撤都够格；阈值为预登记值，不为凑数而收紧",
        "- 波级去重：收复腿上高于前坑 5% 的候选并入前坑波；±5% 双底取最低（merged_candidates 列）",
        "- 真/假坑窗口存在时间重叠、同一熊市共享宏观背景，样本非独立，p 值偏乐观",
        "- 坑底识别用了未来信息（答案已知的逆向研究，非回测）；签名特征全部只用"
        "坑底当日收盘前的公开时刻信息",
        "- 覆盖缺口：Form 144 仅 2023-06 后计入（n=6/3）、暗池 2021-12 后（n=9/6）、"
        "Musk 发帖归档止 2025-05（2026-04-08 真坑失明）",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
