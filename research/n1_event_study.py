#!/usr/bin/env python
"""N1 — 机读事实类情报的历史事件研究（哨兵情报线第一战）.

问题（用户"从古看今"）：高位者的法定披露/公开动作（Form 4 内部人买卖、8-K、
FOMC 决议、Musk 发帖）在**公开时刻之后**对 TSLA 是否有可预测的价格影响？
"当时拿到情报按机械规则操作"能否获利？

方法（零 LLM，纯机读时间戳 + 字符串匹配，防污染）：
1. 事件对齐：event_time_utc → **第一个严格晚于披露时刻的交易日开盘**（盘后/凌晨披露
   → 次日开盘；盘中披露 → 次日开盘，因为当日开盘先于披露，不可用——同为防前视）。
2. 事件研究：入场开盘 → +1/+5/+20 交易日收盘的收益分布，对比全样本随机同尺寸
   抽样基线（bootstrap 1000 次，双侧经验 p 值）+ 对全日均值的 t 统计。
   分段：train 2018→2023 年末 / valid 2024→2026。
3. 机械规则回测（仅对 train 段显著为正的类别）：披露后次日开盘买入（悲观滑点入场，
   src.common.execution），持有 N∈{5,20} 交易日收盘卖出，-20% 灾难止损（stop-market
   悲观结算），单仓位（持仓中新事件跳过）。train 观察 / valid 验证。

已知统计边界（先于结果写死）：
- 重叠窗口：事件密集类（Musk 推文日、内部人卖出日）+20 日窗口互相重叠，收益非独立，
  t 统计偏乐观；bootstrap 基线同样允许重叠抽样，口径对称，但都不修正自相关。
- insider_buy 全历史仅 11 笔（价格数据窗内更少）——样本 <10 的类别只描述不下结论。
- Musk 推文归档止于 2025-05，该类 valid 段覆盖不全。

用法： .venv/bin/python research/n1_event_study.py
输出： outputs/n1_event_study/{per_event_stats.csv, rule_backtest.csv, summary.txt}
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
from src.common.execution import CostModel, entry_fill, settle_bracket  # noqa: E402

DATA = ROOT / "data"
INTEL = DATA / "intel"
OUT = ROOT / "outputs" / "n1_event_study"

WINDOWS = [1, 5, 20]
N_BOOT = 1000
TRAIN_END = pd.Timestamp("2024-01-01", tz="UTC")  # train < 此刻 <= valid
TESLA_RE = r"tesla|tsla"
CLUSTER_DAYS = 30
CLUSTER_MIN_USD = 1_000_000
CAT_STOP = 0.20  # 灾难止损 20%
HOLD_NS = [5, 20]
P_SIG = 0.05
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- price scaffolding


def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """小时线 → 日线（ET 交易日），保留每日首根 bar 的 UTC 开盘时刻."""
    et_date = hourly.index.tz_convert(ET).date
    g = hourly.groupby(et_date)
    daily = pd.DataFrame(
        {
            "open_time_utc": g.apply(lambda d: d.index[0]),
            "Open": g["Open"].first(),
            "Close": g["Close"].last(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def entry_day_index(daily: pd.DataFrame, event_time: pd.Timestamp) -> int | None:
    """第一个开盘时刻严格晚于 event_time 的交易日序号；越界返回 None."""
    pos = daily["open_time_utc"].searchsorted(event_time, side="right")
    return int(pos) if pos < len(daily) else None


# ---------------------------------------------------------------- event classes


def load_events(name: str) -> pd.DataFrame:
    df = pd.read_csv(INTEL / f"{name}.csv")
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed")
    return df


def payloads(df: pd.DataFrame) -> list[dict]:
    return [json.loads(p) for p in df["payload"]]


def build_classes(daily: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, str]]:
    """返回 {类别名: 去重后的入场日序号列表} 与 {类别名: 备注}."""
    classes: dict[str, list[int]] = {}
    notes: dict[str, str] = {}

    def add(name: str, times: list[pd.Timestamp], note: str = "") -> None:
        idx = sorted({e for t in times if (e := entry_day_index(daily, t)) is not None})
        classes[name] = idx
        dropped = len(times) - len({t for t in times if entry_day_index(daily, t) is not None})
        notes[name] = note + (f"（{dropped} 事件在价格窗外丢弃）" if dropped else "")

    # Form 4
    f4 = load_events("edgar_form4")
    for typ, cname in [("insider_buy", "form4_insider_buy"), ("insider_sell", "form4_insider_sell")]:
        sub = f4[f4["type"] == typ]
        add(cname, list(sub["event_time_utc"]), f"申报数 {len(sub)}；")

    # 内部人买入集群：单笔 >=$1M，或 30 日内 >=2 个不同 CIK 买入（以满足条件的披露时刻为事件）
    buys = f4[f4["type"] == "insider_buy"].copy()
    pl = payloads(buys)
    buys["value_usd"] = [p["value_usd"] for p in pl]
    buys["owner_cik"] = [p["owner_cik"] for p in pl]
    buys = buys.sort_values("event_time_utc").reset_index(drop=True)
    cluster_times, reasons = [], []
    for i, r in buys.iterrows():
        big = r["value_usd"] >= CLUSTER_MIN_USD
        prior = buys[
            (buys["event_time_utc"] >= r["event_time_utc"] - pd.Timedelta(days=CLUSTER_DAYS))
            & (buys["event_time_utc"] <= r["event_time_utc"])
        ]
        multi = prior["owner_cik"].nunique() >= 2
        if big or multi:
            cluster_times.append(r["event_time_utc"])
            reasons.append("big" if big else "multi")
    add(
        "form4_buy_cluster",
        cluster_times,
        f"触发 {len(cluster_times)}（单笔≥$1M:{reasons.count('big')} / 30日多人:{reasons.count('multi')}）；",
    )

    # 8-K 按 item（>=10 张才成类）
    k8 = load_events("edgar_8k")
    item_lists = [p["items"] for p in payloads(k8)]
    all_items = pd.Series([i for lst in item_lists for i in lst])
    for item, cnt in all_items.value_counts().items():
        if cnt < 10:
            continue
        mask = [item in lst for lst in item_lists]
        add(f"8k_item_{item}", list(k8.loc[mask, "event_time_utc"]), f"申报数 {cnt}；")

    # FOMC
    fomc = load_events("fomc")
    add("fomc_statement", list(fomc["event_time_utc"]), f"决议数 {len(fomc)}；")

    # Musk 推文：提及 Tesla 的原创/回复（转发不算）→ 入场日级去重
    tw = load_events("musk_tweets")
    tw = tw[tw["type"].isin(["musk_post", "musk_reply"])]
    texts = pd.Series([json.loads(p)["text"] for p in tw["payload"]], index=tw.index)
    mention = texts.str.lower().str.contains(TESLA_RE, regex=True, na=False)
    add(
        "musk_tesla_mention",
        list(tw.loc[mention, "event_time_utc"]),
        f"提及帖 {int(mention.sum())}/{len(tw)}（归档止于 2025-05）；",
    )

    # Musk 发帖密度：ET 日发帖数 > 过去 365 天该数列的 90 分位（因果阈值，无前视）
    tw_all = load_events("musk_tweets")
    et_day = tw_all["event_time_utc"].dt.tz_convert(ET).dt.date
    cnt = tw_all.groupby(et_day).size()
    cnt.index = pd.to_datetime(cnt.index)
    cnt = cnt.asfreq("D", fill_value=0)
    thr = cnt.shift(1).rolling(365, min_periods=180).quantile(0.9)
    dense_days = cnt.index[(cnt > thr) & thr.notna()]
    # 事件时刻 = 当日 ET 结束（23:59），入场自然落到下一交易日开盘
    dense_times = [
        pd.Timestamp(d).tz_localize(ET) + pd.Timedelta(hours=23, minutes=59) for d in dense_days
    ]
    add(
        "musk_high_density",
        [t.tz_convert("UTC") for t in dense_times],
        f"高密度日 {len(dense_days)}（>过去365日90分位；归档止于 2025-05）；",
    )

    return classes, notes


# ---------------------------------------------------------------- event study


def window_returns(daily: pd.DataFrame, n: int) -> np.ndarray:
    """ret[e] = Close[e+n-1]/Open[e] - 1；末尾不足 n 日为 NaN."""
    open_ = daily["Open"].to_numpy()
    close = daily["Close"].to_numpy()
    ret = np.full(len(daily), np.nan)
    if n <= len(daily):
        ret[: len(daily) - n + 1] = close[n - 1 :] / open_[: len(daily) - n + 1] - 1
    return ret


def seg_mask(daily: pd.DataFrame, seg: str) -> np.ndarray:
    ts = pd.to_datetime(daily["open_time_utc"], utc=True)
    if seg == "train":
        return (ts < TRAIN_END).to_numpy()
    if seg == "valid":
        return (ts >= TRAIN_END).to_numpy()
    return np.ones(len(daily), bool)


def study(daily: pd.DataFrame, classes: dict[str, list[int]], notes: dict[str, str]) -> pd.DataFrame:
    rows = []
    rets = {n: window_returns(daily, n) for n in WINDOWS}
    for n in WINDOWS:
        valid_all = ~np.isnan(rets[n])
        for seg in ["full", "train", "valid"]:
            m = seg_mask(daily, seg) & valid_all
            pool = rets[n][m]
            pool_idx = np.flatnonzero(m)
            for cname, idx in classes.items():
                ev = [i for i in idx if m[i]]
                r = rets[n][ev]
                row = {
                    "class": cname,
                    "window_d": n,
                    "segment": seg,
                    "n_events": len(ev),
                    "note": notes[cname],
                }
                if len(ev) >= 2 and len(pool) > 10:
                    boot = np.array(
                        [rets[n][RNG.choice(pool_idx, size=len(ev))].mean() for _ in range(N_BOOT)]
                    )
                    frac_ge = (boot >= r.mean()).mean()
                    t = (r.mean() - pool.mean()) / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-12)
                    row.update(
                        mean_bp=r.mean() * 1e4,
                        median_bp=np.median(r) * 1e4,
                        std_bp=r.std(ddof=1) * 1e4,
                        baseline_mean_bp=pool.mean() * 1e4,
                        t_stat=t,
                        p_boot_two_sided=2 * min(frac_ge, 1 - frac_ge),
                        p_boot_pos=frac_ge,  # 单侧：随机基线 >= 事件均值 的比例（小=事件更好）
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- mechanical rule backtest


def rule_backtest(
    hourly: pd.DataFrame, daily: pd.DataFrame, entry_idx: list[int], hold_n: int
) -> pd.DataFrame:
    """披露后次日开盘买入、持 hold_n 交易日收盘卖、-20% 灾难止损、单仓位、悲观结算."""
    cost = CostModel()  # fee 1bp/side, slip 2bp
    trades = []
    busy_until = -1
    for e in entry_idx:
        if e <= busy_until:
            continue  # 持仓中，跳过
        exit_day = e + hold_n - 1
        if exit_day >= len(daily):
            continue
        w_start = daily["open_time_utc"].iloc[e]
        w_end_close = daily["open_time_utc"].iloc[exit_day]
        window = hourly[
            (hourly.index >= w_start)
            & (hourly.index.tz_convert(ET).date <= daily.index[exit_day].date())
        ]
        if window.empty:
            continue
        raw_open = float(window["Open"].iloc[0])
        entry_px = entry_fill(raw_open, 1, cost)
        res = settle_bracket(
            window, 1, entry_px, tp_px=raw_open * 1e9, sl_px=entry_px * (1 - CAT_STOP), cost=cost
        )
        trades.append(
            {
                "entry_time": w_start,
                "entry_px": entry_px,
                "exit_time": res.exit_time,
                "exit_px": res.exit_px,
                "hit": res.hit,
                "ret": res.ret,
                "hold_n": hold_n,
            }
        )
        busy_until = exit_day
        _ = w_end_close
    return pd.DataFrame(trades)


def bt_metrics(tr: pd.DataFrame, label: str) -> dict:
    if tr.empty:
        return {"leg": label, "n": 0}
    eq = (1 + tr["ret"]).cumprod()
    peak = eq.cummax()
    years = max(
        (tr["exit_time"].iloc[-1] - tr["entry_time"].iloc[0]).total_seconds() / 86400 / 365.25, 1e-9
    )
    return {
        "leg": label,
        "n": len(tr),
        "wr_pct": (tr["ret"] > 0).mean() * 100,
        "avg_bp": tr["ret"].mean() * 1e4,
        "median_bp": tr["ret"].median() * 1e4,
        "total_ret_pct": (eq.iloc[-1] - 1) * 100,
        "cagr_pct": (eq.iloc[-1] ** (1 / years) - 1) * 100,
        "closed_mdd_pct": ((eq / peak) - 1).min() * 100,
        "n_cat_stop": int((tr["hit"] == "sl").sum()),
        "max_hold_days": int((tr["exit_time"] - tr["entry_time"]).dt.days.max()),
    }


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hourly = load_bars(str(DATA / "TSLA_1h_alpaca.csv"))
    daily = build_daily(hourly)
    print(f"daily bars: {len(daily)} ({daily.index[0].date()} -> {daily.index[-1].date()})")

    classes, notes = build_classes(daily)
    for c, idx in classes.items():
        print(f"  {c}: {len(idx)} entry days  [{notes[c]}]")

    stats = study(daily, classes, notes)
    stats.to_csv(OUT / "per_event_stats.csv", index=False)

    # 多重比较记账：类别 × 窗口 × 分段 中实际算了 p 值的行数
    n_tested = int(stats["p_boot_two_sided"].notna().sum())

    # 规则回测：train 段显著为正（单侧 p<=0.05 且 n>=10）的类别
    sig = stats[
        (stats["segment"] == "train")
        & (stats["n_events"] >= 10)
        & (stats["p_boot_pos"] <= P_SIG)
        & (stats["mean_bp"] > stats["baseline_mean_bp"])
    ]
    sig_classes = sorted(sig["class"].unique())
    print(f"\ntrain 段显著为正的类别: {sig_classes or '无'}")

    bt_rows = []
    for cname in sig_classes:
        for hold in HOLD_NS:
            idx = classes[cname]
            for seg, lo, hi in [
                ("train", None, TRAIN_END),
                ("valid", TRAIN_END, None),
            ]:
                sel = [
                    i
                    for i in idx
                    if (lo is None or daily["open_time_utc"].iloc[i] >= lo)
                    and (hi is None or daily["open_time_utc"].iloc[i] < hi)
                ]
                tr = rule_backtest(hourly, daily, sel, hold)
                m = bt_metrics(tr, f"{cname}|hold{hold}|{seg}")
                m.update({"class": cname, "hold_n": hold, "segment": seg, "n_signals": len(sel)})
                bt_rows.append(m)
    bt = pd.DataFrame(bt_rows)
    bt.to_csv(OUT / "rule_backtest.csv", index=False)

    write_summary(stats, bt, classes, notes, n_tested, sig_classes, daily)
    print(f"\nwritten: {OUT}")


def fmt_stat_row(r) -> str:
    if pd.isna(r.get("mean_bp", np.nan)):
        return f"    +{r['window_d']:>2}日  n={r['n_events']:<4d} （样本不足，不计）"
    return (
        f"    +{r['window_d']:>2}日  n={r['n_events']:<4d} 均值 {r['mean_bp']:+7.1f}bp"
        f"（基线 {r['baseline_mean_bp']:+6.1f}bp） 中位 {r['median_bp']:+7.1f}bp"
        f"  t={r['t_stat']:+5.2f}  p_boot(双侧)={r['p_boot_two_sided']:.3f}"
    )


def write_summary(stats, bt, classes, notes, n_tested, sig_classes, daily) -> None:
    L = []
    L.append("N1 — 机读事实类情报的历史事件研究（2026-07-24）")
    L.append("=" * 72)
    L.append(f"价格窗：{daily.index[0].date()} → {daily.index[-1].date()}（{len(daily)} 交易日，1h→日线）")
    L.append(f"分段：train < 2024-01-01 <= valid；bootstrap {N_BOOT} 次；对齐 = 披露后第一个可用开盘")
    L.append("")
    L.append("事件类别与样本：")
    for c, idx in classes.items():
        L.append(f"  {c}: {len(idx)} 入场日  [{notes[c]}]")
    L.append("")
    L.append(f"多重比较记账：本轮共计算 {n_tested} 个 (类别×窗口×分段) 假设检验 p 值，")
    L.append(f"外加规则回测 {len(bt)} 个 (类别×持有期×分段) 配置——并入 strategy-lab 计数器。")
    L.append("")
    for cname in classes:
        L.append(f"—— {cname} ——")
        for seg in ["train", "valid", "full"]:
            sub = stats[(stats["class"] == cname) & (stats["segment"] == seg)]
            L.append(f"  [{seg}]")
            for _, r in sub.sort_values("window_d").iterrows():
                L.append(fmt_stat_row(r))
        L.append("")
    if len(bt):
        L.append("—— 机械规则回测（悲观结算 fee1bp/slip2bp，-20% 灾难止损，单仓位）——")
        for _, r in bt.iterrows():
            if r["n"] == 0:
                L.append(f"  {r['leg']}: 无成交（信号 {r['n_signals']}）")
                continue
            cagr = f"CAGR {r['cagr_pct']:+.1f}%" if r["n"] >= 5 else "CAGR —（n<5 无意义）"
            L.append(
                f"  {r['leg']}: n={r['n']}（信号 {r['n_signals']}） WR {r['wr_pct']:.1f}%"
                f" 均值 {r['avg_bp']:+.1f}bp 总收益 {r['total_ret_pct']:+.1f}%"
                f" {cagr} 已平仓MDD {r['closed_mdd_pct']:.1f}%"
                f" 灾难止损 {r['n_cat_stop']} 次 最长持仓 {r['max_hold_days']} 天"
            )
    else:
        L.append("—— 机械规则回测：train 段无显著为正类别，按预登记协议跳过 ——")
    L.append("")
    L.append(VERDICT)
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


VERDICT = """\
==================== 中文判读（2026-07-24，基于上表数字） ====================

一句话回答"当时拿到情报按机械规则操作能否获利"：**不能给出肯定回答**。
13 个事件类别 × 3 个窗口在训练段（2018→2023）没有任何一个在多重比较校正后
仍显著；唯一的名义显著者（8-K item 1.01）验证段只有 1 个样本，无法裁决。

逐信号判读：

1. Form 4 内部人买入（全历史仅 11 笔申报、价格窗内 9 个入场日）——
   样本 <10，按预登记规则只描述不下结论。描述：训练段 7 笔披露后 20 日
   平均 -1150bp（基线 +537bp）——2018-2019 Musk 的几次增持全部买在
   下跌途中，披露后继续跌；验证段 2 笔（2025-04 Gebbia $1M、2025-09
   Musk $10 亿）披露后 20 日 +1338bp。方向在两段相反，唯一诚实的结论是
   "TSLA 内部人买入太罕见，历史上无法构成可回测的信号"。学术文献里
   内部人买入的弱预测力（框架文档预警条目）在 TSLA 单票上不可复现——
   因为几乎没有买入。集群定义（≥2 人/30 日 或 单笔 ≥$1M）触发 9 次、
   价格窗内 7 次，同样太少。

2. Form 4 内部人卖出（252 个入场日，样本充足）——无信息。训练段
   +1/+5/+20 日全部与随机基线无法区分（p 0.44-0.71）。与文献一致：
   高管卖出理由太多（股权激励变现、缴税），不含方向信息。验证段
   +5 日 p=0.014 为孤立名义显著（+1/+20 日不显著、训练段不显著），
   多重比较下按噪声处理。

3. 8-K：唯一名义显著 = item 1.01（重大协议，训练段 17 例）披露后
   次日 +124bp、t=3.09、单侧 p≈0.03——但本轮共算 108 个 p 值，
   Bonferroni 校正后不显著；且 +5/+20 日窗口无延续，验证段仅 1 例。
   机械规则回测（次日开盘买入）：hold5 训练段 CAGR 7.9% 低于 8% 达标线；
   hold20 训练段 CAGR 24.6% 但 WR 57%、已平仓 MDD -25.8% 破 20% 线，
   且验证段 n=1 远低于"≥30 笔"达标线。**判：不达标，不进 shadow**。
   其余 item（业绩 2.02、RegFD 7.01、其他事件 8.01、高管变动 5.02）全不显著。

4. FOMC 决议（67 例）——次日方向为负（训练 -57bp、全样本 -48bp，
   p≈0.08-0.10），未过显著线；+5/+20 日无信号。"宏观贴现率敏感"
   在 TSLA 上不构成可交易的日级事件信号。

5. Musk 推文（归档止于 2025-05，此后缺口如实标注）——
   提及 Tesla 的日子占全部交易日 ~72%（1225/约1700），信号几乎常开，
   自然与基线无差（各窗口 p 0.17-0.65）；高密度发帖日（374 例）同样
   不显著，训练段 +20 日甚至偏负（+334bp vs 基线 +537bp）。
   结论：不做内容/语义区分的推文级信号没有信息量——这正是叙事类
   情报需要 LLM 前向标注（intel-framework 第三节）而非字符串匹配的原因。

方法论边界（诚实声明）：
- 事件密集类的 +20 日窗口互相重叠，t 统计偏乐观（本轮结论全为"不显著"，
  该偏差只会让结论更保守，不影响判读方向）。
- 8-K item 1.01 的次日效应值得留在观察清单：它是"公司签了重大协议"
  这一事实类信号，机制上说得通，但历史频率（~2.4 张/年）注定永远
  凑不齐 30 笔验证样本——只能作为未来复合信号的一个特征，
  不能作为独立策略。
- 本轮多重比较记账：108 个假设检验 + 4 个回测配置 = 112，并入
  strategy-lab 计数器（由主线更新 docs）。

对哨兵体系的结论："从古看今"这一步做完了——机读事实类情报在 TSLA 上
**没有找到能独立变现的历史信号**；真金白银类（内部人买入）败在样本
稀缺而非信号方向。下一步的价值假设只剩两条：(a) 把 8-K/Form 4 事实
特征并进 GBDT 复合模型（E8 管线）而非独立成策；(b) 叙事类（推文语义）
走前向 LLM 标注积累。以上均需另行登记假设后再动手。
"""


if __name__ == "__main__":
    main()
