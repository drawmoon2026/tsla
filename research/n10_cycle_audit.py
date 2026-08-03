#!/usr/bin/env python
"""N10 — 康波嵌套周期买卖点反推（用户三步法）.

用户三步：①康波周期找买卖点 ②复合信息源反推依据 ③依据顺推买卖。
诚实转译：康波本体（50-60 年）不可检验；用其嵌套的可测周期操作化——
利率周期（FEDFUNDS 相位）、库存周期（INDPRO/MANEMP proxy，NAPM 已从
FRED 下架）、流动性周期（M2 同比）、长端与曲线（DGS10/T10Y2Y）。

第一步 · 买卖点目录
- 买点：复用 N4 的 27 坑目录（真 17/假 9/中间态 1，拆股修正后版本，
  outputs/n4_golden_pit/pits_catalog.csv，不重算）
- 卖点（新建，预登记判据，镜像 N4 口径）：
  顶部候选 = 日线 Close 为 ±5 日局部高点，且较前 60 交易日最低收盘涨幅 >= 25%
  聚类去重：相邻候选（间隔 <= 20 交易日且其间未从簇内最高点回落 >= 10%）并簇，
  取簇内最高收盘为顶；波级去重镜像 N4：60 日窗内后顶低于前顶 5% 以上 = 下跌腿
  反抽并入前顶波，±5% 之内 = 双顶取更高者，高于前顶 5% 以上 = 新决策点保留
  真顶 = 之后 60 交易日内先触及 顶*0.80（回撤 >= 20%）且中途未创收盘新高
  假顶 = 60 日内先触及 顶*1.10（继续创新高 >= 10%）
  中间态（回撤前先创了 <10% 的边际新高、或 60 日内两者都未触及）剔除；
  数据末端窗口不足剔除并注明

第二步 · 宏观相位变量采集与反推
- 数据：intel/collectors/macro_fred.py 落盘 data/intel/macro/*.csv，
  双时间戳（observation_date + available_from_utc；vintage 近似 = 月度滞后
  1 个月、日度滞后 1 天；非真 ALFRED vintage，月度值含事后修订——已声明局限）
- 相位定义（预登记，不扫描阈值）：
  rate_phase        FEDFUNDS 6 个月变化：>= +0.25pp 加息中 / <= -0.25pp 降息中 / 其余平台
  fedfunds_chg_6m   上述连续值（pp）
  indpro_yoy        INDPRO 同比 %（库存周期 proxy 水平）
  indpro_yoy_dir    同比 vs 6 个月前同比：up（补库上行）/ down（去库下行）
  manemp_chg_6m     MANEMP 6 个月变化 %（库存周期第二 proxy）
  m2_yoy            M2 同比 %（流动性水平）
  m2_yoy_dir        M2 同比 vs 6 个月前同比：up / down（流动性增速方向）
  dgs10_chg_6m      DGS10 6 个月变化（pp，正 = 长端上行）
  t10y2y            10Y-2Y 利差水平（pp）
  curve_inverted    利差 < 0（倒挂）与否
- 签名对比：真 vs 假，坑与顶分开，只在发现段（<= 2023-06-30）做检验——
  考场段（2023-07 起）留给第三步复验，不参与签名筛选（防泄漏）。
  连续变量 Mann-Whitney + 均值差置换（复用 N4 框架）；类别变量 Fisher 精确
  （二类）/ 卡方统计量置换（三类）。N4 已有微观变量：坑侧 N4 已测过（26 项，
  已入总计数器）不重测，只作目录对照列；顶侧为新维度，作对照层一并检验。
- 多重比较记账：宏观 10 变量 × 坑/顶 2 组 = 20 + 顶侧微观对照 26 = 46 项

第三步 · 顺推（仅当有幸存签名）
- 名义 p < 0.10 且方向可解释的宏观签名 → 规则雏形（类别取真组富集类，
  变化类连续变量取中性阈 0），发现段构建、考场段复验；转折点少，
  样本如实报告，结论按方向性证据封顶

用法： .venv/bin/python research/n10_cycle_audit.py
输出： outputs/n10_cycle_audit/{tops_catalog.csv, signature_macro.csv, summary.txt}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "research")):
    if p not in sys.path:
        sys.path.insert(0, p)

import n4_golden_pit as n4  # noqa: E402  最大复用：日线聚合/坑框架/IntelBook/置换检验
from intel.collectors.macro_fred import load_series  # noqa: E402
from src.common.data_io import load_bars  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "n10_cycle_audit"
N4_CATALOG = ROOT / "outputs" / "n4_golden_pit" / "pits_catalog.csv"

# ---- 预登记参数（顶侧镜像 N4，不扫描） ----
LOCAL_HIGH_HALF = 5      # 局部高点半窗（±5 日）
LOW_LOOKBACK = 60        # 前低回看（交易日）
RISE_THRESH = 0.25       # 候选涨幅门槛（较前 60 日最低收盘）
CLUSTER_GAP = 20
CLUSTER_PULLBACK = 0.10  # 簇内视为"同一波动"的最大中途回落
FWD_WINDOW = 60
WAVE_WINDOW = 60
WAVE_BAND = 0.05
TRUE_DD = 0.20           # 真顶回撤门槛
FALSE_NEWHIGH = 0.10     # 假顶继续创新高门槛
DISCOVERY_END = pd.Timestamp("2023-06-30")  # 发现段/考场段分界（与 N3/N4 口径一致）
N_PERM = 10000
RNG = np.random.default_rng(42)

MACRO_SERIES = ["FEDFUNDS", "M2SL", "INDPRO", "MANEMP", "DGS10", "T10Y2Y"]
MACRO_CONT = ["fedfunds_chg_6m", "indpro_yoy", "manemp_chg_6m",
              "m2_yoy", "dgs10_chg_6m", "t10y2y"]
MACRO_CAT = ["rate_phase", "indpro_yoy_dir", "m2_yoy_dir", "curve_inverted"]
MACRO_ALL = MACRO_CONT + MACRO_CAT


# ---------------------------------------------------------------- top detection


def find_tops(daily: pd.DataFrame) -> pd.DataFrame:
    """镜像 n4.find_pits 的顶部版本（预登记判据见模块 docstring）。"""
    c = daily["Close"]
    prior_low = c.rolling(LOW_LOOKBACK, min_periods=LOW_LOOKBACK).min().shift(1)
    rise = c / prior_low - 1.0
    local_high = c == c.rolling(2 * LOCAL_HIGH_HALF + 1, center=True, min_periods=1).max()
    cand_idx = np.flatnonzero((rise >= RISE_THRESH).to_numpy() & local_high.to_numpy())

    # 聚类去重：同一波动取最高点
    clusters: list[list[int]] = []
    for i in cand_idx:
        if clusters:
            cur = clusters[-1]
            last = cur[-1]
            cmax_i = max(cur, key=lambda k: c.iloc[k])
            interim_min = c.iloc[last: i + 1].min()
            if (i - last <= CLUSTER_GAP
                    and interim_min > c.iloc[cmax_i] * (1 - CLUSTER_PULLBACK)):
                cur.append(i)
                continue
        clusters.append([i])
    top_idx = [max(cl, key=lambda k: c.iloc[k]) for cl in clusters]

    # 波级去重：镜像 N4
    retained: list[int] = []
    merged: dict[int, list[int]] = {}
    for i in top_idx:
        if retained:
            r = retained[-1]
            if i - r <= WAVE_WINDOW:
                ratio = c.iloc[i] / c.iloc[r]
                if ratio < 1 - WAVE_BAND:            # 下跌腿上的反抽高点，并入前顶波
                    merged.setdefault(r, []).append(i)
                    continue
                if ratio <= 1 + WAVE_BAND:           # 双顶：取更高者
                    if c.iloc[i] > c.iloc[r]:
                        merged.setdefault(i, []).extend(merged.pop(r, []) + [r])
                        retained[-1] = i
                    else:
                        merged.setdefault(r, []).append(i)
                    continue
        retained.append(i)
    top_idx = retained

    rows = []
    for i in top_idx:
        top_close = c.iloc[i]
        trough_win = c.iloc[max(0, i - LOW_LOOKBACK): i]
        trough_pos = int(trough_win.to_numpy().argmin())
        trough_i = max(0, i - LOW_LOOKBACK) + trough_pos
        fwd = c.iloc[i + 1: i + 1 + FWD_WINDOW]
        n_fwd = len(fwd)
        fwd_max = fwd.max() / top_close - 1 if n_fwd else np.nan
        fwd_min = fwd.min() / top_close - 1 if n_fwd else np.nan
        hit_dd = fwd[fwd <= top_close * (1 - TRUE_DD)]
        hit_nh = fwd[fwd >= top_close * (1 + FALSE_NEWHIGH)]
        t_dd = hit_dd.index[0] if len(hit_dd) else None
        t_nh = hit_nh.index[0] if len(hit_nh) else None
        if t_nh is not None and (t_dd is None or t_nh < t_dd):
            label = "false"          # 先创新高 >= 10%：卖飞
        elif t_dd is not None:
            interim_max = fwd.loc[:t_dd].iloc[:-1].max() if len(fwd.loc[:t_dd]) > 1 else top_close
            label = "true" if interim_max <= top_close else "mid"  # 真顶要求中途未创收盘新高
        elif n_fwd < FWD_WINDOW:
            label = "truncated"      # 数据末端未见分晓
        else:
            label = "mid"            # 60 日内既未回撤 20% 也未新高 10%
        rows.append({
            "top_i": i,
            "top_date": daily.index[i].date(),
            "top_close": top_close,
            "trough_date": daily.index[trough_i].date(),
            "trough_close": c.iloc[trough_i],
            "rise_pct": (top_close / c.iloc[trough_i] - 1) * 100,
            "days_from_trough": i - trough_i,
            "n_fwd_days": n_fwd,
            "fwd60_max_pct": fwd_max * 100 if n_fwd else np.nan,
            "fwd60_min_pct": fwd_min * 100 if n_fwd else np.nan,
            "days_to_dd20": (int(daily.index.get_loc(t_dd) - i) if t_dd is not None else np.nan),
            "days_to_newhigh10": (int(daily.index.get_loc(t_nh) - i) if t_nh is not None else np.nan),
            "label": label,
            "merged_candidates": ";".join(
                str(daily.index[k].date()) for k in sorted(merged.get(i, []))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- macro phases


class MacroBook:
    """宏观相位查询：全部按 available_from_utc <= t 的 as-of 口径（vintage 近似）。"""

    def __init__(self):
        self.s = {sid: load_series(sid) for sid in MACRO_SERIES}

    def _asof_pos(self, sid: str, t: pd.Timestamp) -> int | None:
        df = self.s[sid]
        ok = np.flatnonzero((df["available_from_utc"] <= t).to_numpy())
        return int(ok[-1]) if len(ok) else None

    def phases(self, t: pd.Timestamp) -> dict:
        f: dict[str, object] = {k: np.nan for k in MACRO_ALL}

        def mval(sid: str, p: int, back: int) -> float:
            v = self.s[sid]["value"]
            return float(v.iloc[p - back]) if p - back >= 0 else np.nan

        # 利率周期
        p = self._asof_pos("FEDFUNDS", t)
        if p is not None:
            now, ago = mval("FEDFUNDS", p, 0), mval("FEDFUNDS", p, 6)
            chg = now - ago
            f["fedfunds_chg_6m"] = chg
            f["rate_phase"] = ("hiking" if chg >= 0.25
                               else "cutting" if chg <= -0.25 else "plateau")
        # 库存周期 proxy：INDPRO 同比 + 方向
        p = self._asof_pos("INDPRO", t)
        if p is not None and p >= 18:
            yoy = (mval("INDPRO", p, 0) / mval("INDPRO", p, 12) - 1) * 100
            yoy6 = (mval("INDPRO", p, 6) / mval("INDPRO", p, 18) - 1) * 100
            f["indpro_yoy"] = yoy
            f["indpro_yoy_dir"] = "up" if yoy > yoy6 else "down"
        # 库存周期第二 proxy：制造业就业 6 个月变化
        p = self._asof_pos("MANEMP", t)
        if p is not None and p >= 6:
            f["manemp_chg_6m"] = (mval("MANEMP", p, 0) / mval("MANEMP", p, 6) - 1) * 100
        # 流动性：M2 同比 + 方向
        p = self._asof_pos("M2SL", t)
        if p is not None and p >= 18:
            yoy = (mval("M2SL", p, 0) / mval("M2SL", p, 12) - 1) * 100
            yoy6 = (mval("M2SL", p, 6) / mval("M2SL", p, 18) - 1) * 100
            f["m2_yoy"] = yoy
            f["m2_yoy_dir"] = "up" if yoy > yoy6 else "down"
        # 长端方向：DGS10 6 个月变化（日度序列按日历回看）
        p = self._asof_pos("DGS10", t)
        if p is not None:
            df = self.s["DGS10"]
            d_now = df.index[p]
            past = df.iloc[:p + 1]
            past6 = past[past.index <= d_now - pd.DateOffset(months=6)]
            if len(past6):
                f["dgs10_chg_6m"] = float(df["value"].iloc[p]) - float(past6["value"].iloc[-1])
        # 曲线状态
        p = self._asof_pos("T10Y2Y", t)
        if p is not None:
            lvl = float(self.s["T10Y2Y"]["value"].iloc[p])
            f["t10y2y"] = lvl
            f["curve_inverted"] = "inverted" if lvl < 0 else "normal"
        return f


# ---------------------------------------------------------------- stats


def chi2_perm_p(labels: np.ndarray, cats: np.ndarray) -> float:
    """类别 × 真假 的卡方统计量置换检验（三类以上用；小样本下不依赖渐近分布）。"""
    def stat(lab: np.ndarray) -> float:
        s = 0.0
        n = len(lab)
        n_t = (lab == "true").sum()
        for cv in np.unique(cats):
            m = cats == cv
            for gv, ng in (("true", n_t), ("false", n - n_t)):
                obs = ((lab == gv) & m).sum()
                exp = ng * m.sum() / n
                if exp > 0:
                    s += (obs - exp) ** 2 / exp
        return s
    obs = stat(labels)
    lab = labels.copy()
    cnt = 0
    for _ in range(N_PERM):
        RNG.shuffle(lab)
        if stat(lab) >= obs - 1e-12:
            cnt += 1
    return cnt / N_PERM


def cat_dist(s: pd.Series) -> str:
    vc = s.value_counts()
    return "/".join(f"{k}:{v}" for k, v in vc.items()) if len(vc) else ""


def compare_group(df: pd.DataFrame, features_cont: list[str], features_cat: list[str],
                  group: str, layer: str) -> list[dict]:
    """发现段真 vs 假签名检验。df 需含 label 列（true/false）。"""
    t = df[df["label"] == "true"]
    fls = df[df["label"] == "false"]
    rows = []
    for col in features_cont:
        a = t[col].dropna().to_numpy(dtype=float)
        b = fls[col].dropna().to_numpy(dtype=float)
        row = {"group": group, "layer": layer, "feature": col, "kind": "cont",
               "n_true": len(a), "n_false": len(b),
               "true_stat": f"{a.mean():.3f}" if len(a) else "",
               "false_stat": f"{b.mean():.3f}" if len(b) else "",
               "median_true": np.median(a) if len(a) else np.nan,
               "median_false": np.median(b) if len(b) else np.nan}
        if len(a) >= 3 and len(b) >= 3 and np.ptp(np.concatenate([a, b])) > 0:
            row["p_main"] = mannwhitneyu(a, b, alternative="two-sided").pvalue
            row["p_perm"] = n4.perm_pvalue(a, b)
        else:
            row["p_main"] = row["p_perm"] = np.nan
        rows.append(row)
    for col in features_cat:
        sub = pd.concat([t, fls])[[col, "label"]].dropna()
        a, b = sub[sub["label"] == "true"][col], sub[sub["label"] == "false"][col]
        row = {"group": group, "layer": layer, "feature": col, "kind": "cat",
               "n_true": len(a), "n_false": len(b),
               "true_stat": cat_dist(a), "false_stat": cat_dist(b),
               "median_true": np.nan, "median_false": np.nan}
        cats = sub[col].unique()
        if len(a) >= 3 and len(b) >= 3 and len(cats) >= 2:
            if len(cats) == 2:
                table = [[(a == cv).sum() for cv in cats],
                         [(b == cv).sum() for cv in cats]]
                row["p_main"] = fisher_exact(table)[1]
                row["p_perm"] = np.nan
            else:
                row["p_main"] = np.nan
                row["p_perm"] = chi2_perm_p(sub["label"].to_numpy(), sub[col].to_numpy())
        else:
            row["p_main"] = row["p_perm"] = np.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------- step 3: rules


NEUTRAL_THRESH = {  # 变化类连续变量的中性阈（预登记：变化量以 0 为界）
    "fedfunds_chg_6m": 0.0, "manemp_chg_6m": 0.0, "dgs10_chg_6m": 0.0, "t10y2y": 0.0,
}


def build_rule_legs(surv: pd.DataFrame, disc: pd.DataFrame) -> list[tuple[str, str]]:
    """幸存签名 → 规则腿（列名, 表达式描述）。返回 [(feature, desc)]，
    同时在 disc/exam DataFrame 上以 _leg_<feature> 布尔列物化。"""
    legs = []
    for _, r in surv.iterrows():
        col = r["feature"]
        if r["kind"] == "cat":
            t_dist = disc[disc["label"] == "true"][col].value_counts(normalize=True)
            f_dist = disc[disc["label"] == "false"][col].value_counts(normalize=True)
            enrich = (t_dist - f_dist.reindex(t_dist.index).fillna(0)).idxmax()
            legs.append((col, f"{col} == {enrich!r}（真组富集类）"))
        else:
            thr = NEUTRAL_THRESH.get(col)
            side = np.sign(float(r["true_stat"]) - float(r["false_stat"]))
            if thr is None:
                thr = float(pd.concat([disc[disc["label"] == "true"][col],
                                       disc[disc["label"] == "false"][col]]).median())
                note = "（发现段合并中位数阈——弱数据依赖，如实标注）"
            else:
                note = "（中性阈 0）"
            op = ">=" if side > 0 else "<="
            legs.append((col, f"{col} {op} {thr:.3g} {note}"))
    return legs


def eval_rule(df: pd.DataFrame, legs: list[tuple[str, str]]) -> dict:
    m = pd.Series(True, index=df.index)
    for col, desc in legs:
        if "==" in desc:
            cat = desc.split("==")[1].split("（")[0].strip().strip("'\"")
            m &= df[col] == cat
        else:
            op = ">=" if ">=" in desc else "<="
            thr = float(desc.split(op)[1].split("（")[0])
            m &= (df[col] >= thr) if op == ">=" else (df[col] <= thr)
    m = m.fillna(False)
    lab = df["label"]
    return {"hit_true": int((m & (lab == "true")).sum()),
            "n_true": int((lab == "true").sum()),
            "hit_false": int((m & (lab == "false")).sum()),
            "n_false": int((lab == "false").sum())}


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hourly = load_bars(str(DATA / "TSLA_1h_alpaca.csv"))
    daily = n4.build_daily(hourly)
    macro = MacroBook()

    # ---- 第一步 A：买点目录（复用 N4，不重算） ----
    pits = pd.read_csv(N4_CATALOG, parse_dates=["pit_date"])

    # ---- 第一步 B：卖点目录（新建） ----
    tops = find_tops(daily)
    tops["top_date"] = pd.to_datetime(tops["top_date"])

    # ---- 第二步：宏观相位 + 顶侧微观对照 ----
    book = n4.IntelBook(daily)  # 顶侧微观对照列（N4 特征原样复用）

    def attach_macro(df: pd.DataFrame, date_col: str, idx_col: str) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            i = int(r[idx_col]) if idx_col in r and not pd.isna(r.get(idx_col)) else None
            if i is None or daily.index[i].date() != r[date_col].date():
                i = int(daily.index.get_loc(pd.Timestamp(r[date_col].date())))
            t1 = daily["close_time_utc"].iloc[i]
            rows.append(macro.phases(t1))
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(rows)], axis=1)

    pits = attach_macro(pits, "pit_date", "pit_i")
    micro_rows = [book.features(int(i))[0] for i in tops["top_i"]]
    tops = pd.concat([tops.reset_index(drop=True), pd.DataFrame(micro_rows)], axis=1)
    tops = attach_macro(tops, "top_date", "top_i")
    tops.to_csv(OUT / "tops_catalog.csv", index=False, float_format="%.4f")
    pits.to_csv(OUT / "pits_macro.csv", index=False, float_format="%.4f")  # 复核用：坑目录+宏观相位列

    # 分段
    pits_disc = pits[pits["pit_date"] <= DISCOVERY_END]
    pits_exam = pits[pits["pit_date"] > DISCOVERY_END]
    tops_disc = tops[tops["top_date"] <= DISCOVERY_END]
    tops_exam = tops[tops["top_date"] > DISCOVERY_END]

    # 签名检验（仅发现段；考场段留给第三步）
    rows = []
    rows += compare_group(pits_disc, MACRO_CONT, MACRO_CAT, "pit", "macro")
    rows += compare_group(tops_disc, MACRO_CONT, MACRO_CAT, "top", "macro")
    rows += compare_group(tops_disc, n4.FEATURES_ORDER, [], "top", "micro_ref")
    sig = pd.DataFrame(rows)
    sig["p_nominal"] = sig["p_main"].fillna(sig["p_perm"])
    n_tests = int(sig["p_nominal"].notna().sum())
    sig["n_tests_bonferroni"] = n_tests
    sig["p_bonf"] = (sig["p_nominal"] * n_tests).clip(upper=1.0)
    sig.to_csv(OUT / "signature_macro.csv", index=False, float_format="%.4f")

    # ---- 第三步：顺推（仅宏观层幸存签名；微观对照层不进规则——N5 前车之鉴） ----
    surv_pit = sig[(sig["group"] == "pit") & (sig["layer"] == "macro")
                   & (sig["p_nominal"] < 0.10)]
    surv_top = sig[(sig["group"] == "top") & (sig["layer"] == "macro")
                   & (sig["p_nominal"] < 0.10)]

    # ---- summary ----
    L = ["N10 — 康波嵌套周期买卖点反推 摘要",
         f"生成时间：{pd.Timestamp.now()}",
         "",
         f"数据：TSLA 日线 {daily.index[0].date()} → {daily.index[-1].date()}"
         f"（{len(daily)} 交易日）；宏观 = FRED {'/'.join(MACRO_SERIES)}"
         "（vintage 近似：月度滞后 1 个月、日度滞后 1 天；非真 ALFRED vintage，"
         "月度值含事后修订）",
         "NAPM/ISM PMI 已从 FRED 下架（授权收回）——库存周期用 INDPRO 同比 + "
         "MANEMP 6 个月变化替代口径，如实注明",
         "",
         "== 第一步 · 买卖点目录 ==",
         f"买点：复用 N4 拆股修正后目录 27 坑（真 {int((pits['label']=='true').sum())}"
         f"/假 {int((pits['label']=='false').sum())}"
         f"/中间态 {int((pits['label']=='mid').sum())}）",
         f"卖点（新建）：候选聚类去重后 {len(tops)} 顶——真 "
         f"{int((tops['label']=='true').sum())}/假 {int((tops['label']=='false').sum())}"
         f"/中间态 {int((tops['label']=='mid').sum())}"
         f"/窗口不足 {int((tops['label']=='truncated').sum())}",
         "",
         "== 真假顶清单 =="]
    for _, r in tops.iterrows():
        L.append(f"  {r['top_date'].date()} close {r['top_close']:8.2f} "
                 f"rise60 {r['rise_pct']:6.1f}% fwd60 max {r['fwd60_max_pct']:6.1f}% "
                 f"min {r['fwd60_min_pct']:6.1f}% -> {r['label']}")
    L += ["",
          f"分段（分界 {DISCOVERY_END.date()}，与 N3/N4 口径一致）：",
          f"  坑：发现段 真 {int((pits_disc['label']=='true').sum())}/假 "
          f"{int((pits_disc['label']=='false').sum())}；考场段 真 "
          f"{int((pits_exam['label']=='true').sum())}/假 {int((pits_exam['label']=='false').sum())}",
          f"  顶：发现段 真 {int((tops_disc['label']=='true').sum())}/假 "
          f"{int((tops_disc['label']=='false').sum())}；考场段 真 "
          f"{int((tops_exam['label']=='true').sum())}/假 {int((tops_exam['label']=='false').sum())}",
          "",
          "== 第二步 · 签名对比（仅发现段做检验；考场段不参与筛选，防泄漏） =="]

    def fmt_sig(sub: pd.DataFrame) -> list[str]:
        out = []
        for _, r in sub.iterrows():
            if pd.isna(r["p_nominal"]):
                out.append(f"  {r['feature']:22s} n={r['n_true']}/{r['n_false']} "
                           f"[{r['true_stat']}] vs [{r['false_stat']}] 样本不足，不检验")
                continue
            star = " *" if r["p_nominal"] < 0.10 else ""
            out.append(f"  {r['feature']:22s} n={r['n_true']}/{r['n_false']} "
                       f"真[{r['true_stat']}] vs 假[{r['false_stat']}] "
                       f"p={r['p_nominal']:.4f} Bonf={r['p_bonf']:.3f}{star}")
        return out

    L += ["-- 坑（买点）宏观相位：真 vs 假 --",
          *fmt_sig(sig[(sig["group"] == "pit") & (sig["layer"] == "macro")]),
          "",
          "-- 顶（卖点）宏观相位：真 vs 假 --",
          *fmt_sig(sig[(sig["group"] == "top") & (sig["layer"] == "macro")]),
          "",
          "-- 顶（卖点）微观对照层（N4 特征首次用于顶侧） --",
          *fmt_sig(sig[sig["layer"] == "micro_ref"]),
          "",
          "（坑侧微观层 N4 已测过 26 项且经 N5 判死，不重测，目录中仅作对照列）",
          ""]

    L.append("== 第三步 · 顺推 ==")
    rule_results = []
    for name, surv, disc, exam in (("坑（买点）", surv_pit, pits_disc, pits_exam),
                                   ("顶（卖点）", surv_top, tops_disc, tops_exam)):
        if not len(surv):
            L.append(f"{name}：宏观层无名义幸存签名（p<0.10）——宏观相位在此尺度无区分力，不给规则")
            continue
        legs = build_rule_legs(surv, disc)
        L.append(f"{name}：名义幸存 {len(surv)} 项 → 规则雏形（合取）：")
        for col, desc in legs:
            L.append(f"    - {desc}")
        r_in = eval_rule(disc, legs)
        r_out = eval_rule(exam, legs)
        L.append(f"    发现段（构建，in-sample）：真命中 {r_in['hit_true']}/{r_in['n_true']}，"
                 f"假误中 {r_in['hit_false']}/{r_in['n_false']}")
        L.append(f"    考场段（复验，out-of-sample）：真命中 {r_out['hit_true']}/{r_out['n_true']}，"
                 f"假误中 {r_out['hit_false']}/{r_out['n_false']}")
        for leg in legs:  # 逐腿诊断：考场段各腿单独通过率（定位是哪条腿杀死了规则）
            s_in, s_out = eval_rule(disc, [leg]), eval_rule(exam, [leg])
            L.append(f"      腿 {leg[0]:16s} 通过率 发现段 "
                     f"{s_in['hit_true'] + s_in['hit_false']}/{s_in['n_true'] + s_in['n_false']}"
                     f" → 考场段 {s_out['hit_true'] + s_out['hit_false']}"
                     f"/{s_out['n_true'] + s_out['n_false']}")
        rule_results.append((name, legs, r_in, r_out))

    # 最终判读（由数字自动生成，防止重跑后文字与结果脱节）
    L += ["", "== 三步法最终判读 =="]
    if not rule_results:
        L.append("宏观相位层在坑/顶两侧均无名义幸存签名——宏观相位在此尺度无区分力。")
    for name, legs, r_in, r_out in rule_results:
        fired_out = r_out["hit_true"] + r_out["hit_false"]
        n_out = r_out["n_true"] + r_out["n_false"]
        if n_out and fired_out == 0:
            L.append(f"{name}：规则在考场段 {n_out} 个转折点上零触发——发现段签名实为"
                     "**时代标签**（宏观慢变量把发现段特定宏观环境整体标成真转折背景，"
                     "而考场段宏观环境已整体迁移，规则条件不再出现）。前向无区分力。")
        elif n_out:
            prec_in = r_in["hit_true"] / max(r_in["hit_true"] + r_in["hit_false"], 1)
            prec_out = r_out["hit_true"] / max(fired_out, 1)
            L.append(f"{name}：考场段触发 {fired_out}/{n_out}，命中精度 "
                     f"{prec_out:.0%}（发现段 {prec_in:.0%}）——样本极少，仅方向性证据。")

    L += ["",
          f"多重比较：本轮新增 {n_tests} 项检验（宏观 {len(MACRO_ALL)} 变量 × 坑/顶 2 组 "
          f"+ 顶侧微观对照 {len(n4.FEATURES_ORDER)} 项中可检验者），Bonferroni ×{n_tests}；"
          "并入 strategy-lab 计数器（docs 由主线更新）",
          "",
          "方法备注：",
          "- 顶部判据为预登记（候选=局部高点且较前 60 日低点涨 >=25%；真顶=60 日内"
          "回撤 >=20% 且中途未创收盘新高；假顶=先创新高 >=10%；中间态剔除；波级去重镜像 N4）",
          "- 宏观变量为慢变量：同一宏观相位跨越多个转折点，真/假样本高度非独立，"
          "有效样本量远小于名义 n——p 值系统性偏乐观，只作方向性证据",
          "- 8 年窗口 ≈2 轮库存周期/1.5 轮利率周期：相位类别的枚举本身就不完整",
          "- vintage 近似局限：月度值为最终修订值+1 个月滞后，非当时实时可见值",
          "- 考场段样本极少，复验结果按计数如实报告，不做显著性宣称"]

    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
