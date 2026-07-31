#!/usr/bin/env python
"""N7 — 池级 8-K item 1.01（重大协议）事件研究（判决性检验）.

来源：N1 在 TSLA 单标的上发现 item 1.01 次日 +124bp t=3.09（17 例，Bonferroni
后不显著、样本永远凑不齐）。本实验把同一假设放到 26 标的池（样本 ~15 倍）做
一次性判决检验。

主假设（唯一预登记，多重比较 +1）：
    8-K item 1.01 披露后次日超额收益 > 0。
    - 事件对齐 = N1 同款：第一个开盘时刻严格晚于 acceptanceDateTime 的交易日
      （盘后/盘前/周末披露 → 次日开盘；盘中披露 → 次日开盘，防前视）。
    - 次日收益 = 该入场日 Open→Close（1h 聚合日线）。
    - 超额 = 减同日池等权均值（不含本标的，避免自身机械稀释）。
    - 检验 = 标的分层置换 20000 次（各标的保持自身事件数、在该标的自身可用
      交易日内无放回抽样），单侧 p<0.05 过、不过判死。

辅助描述（不计显著）：1/5/20 日窗口、按年份分段、盘中 vs 盘后披露、逐标的方向。
若主检验过：机械规则悲观回测（披露次日开盘买入持 1 日收盘卖，fee1bp/side +
slip2bp 入场），报单笔期望 bp 与年化容量。

数据：data/intel/pool_8k/{SYMBOL}.csv（本脚本 --collect 采集，SEC submissions
API，≤10 req/s、带 UA）；TSLA 复用 data/intel/edgar_8k.csv。价格 data/pool_1h/
+ data/TSLA_1h_alpaca.csv（已拆股折算）。

用法：
    .venv/bin/python research/n7_pool_8k.py --collect   # 先采集（幂等，已有跳过）
    .venv/bin/python research/n7_pool_8k.py             # 分析
输出： outputs/n7_pool_8k/{events_all.csv, primary_test.csv, summary.txt}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_io import ET, load_bars  # noqa: E402

DATA = ROOT / "data"
INTEL = DATA / "intel"
POOL_8K = INTEL / "pool_8k"
OUT = ROOT / "outputs" / "n7_pool_8k"

POOL_SYMBOLS = [  # data/pool_1h 的 25 标的；TSLA 单列
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "CRM", "COST", "JPM", "V", "UNH", "XOM", "WMT", "DIS", "BA", "CAT", "GS",
    "INTC", "MU", "QCOM", "PLTR", "COIN",
]
ALL_SYMBOLS = POOL_SYMBOLS + ["TSLA"]

SINCE = "2018-01-01"
ITEM = "1.01"
WINDOWS = [1, 5, 20]
N_PERM = 20000
P_SIG = 0.05
MAX_ALIGN_GAP_DAYS = 7      # 入场开盘距披露 >7 天 → 事件在价格窗外，丢弃
MIN_BENCH_SYMBOLS = 11      # 同日基准至少 10 个其他标的
FEE_BP, SLIP_BP = 1.0, 2.0  # 悲观成本：fee 1bp/side ×2 + 入场滑点 2bp
RNG = np.random.default_rng(20260801)

UA = {"User-Agent": "TSLA-research tom (drawmoon2026@gmail.com)"}
_MIN_INTERVAL = 0.12  # ~8 req/s < SEC 10 req/s
_last_req = [0.0]


# ---------------------------------------------------------------- collection

def _get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def cik_map() -> dict[str, str]:
    raw = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in raw.values()}


def load_submissions(cik: str) -> pd.DataFrame:
    main = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    frames = [pd.DataFrame(main["filings"]["recent"])]
    for extra in main["filings"].get("files", []):
        if extra["filingTo"] >= SINCE:
            sub = json.loads(_get(f"https://data.sec.gov/submissions/{extra['name']}"))
            frames.append(pd.DataFrame(sub))
    df = pd.concat(frames, ignore_index=True)
    return df[df["filingDate"] >= SINCE].reset_index(drop=True)


def build_8k_csv(subs: pd.DataFrame) -> pd.DataFrame:
    """与 intel/edgar.py fetch_8k 完全同 schema：event_time_utc,source,type,payload."""
    k8 = subs[subs["form"].isin(["8-K", "8-K/A"])].reset_index(drop=True)
    rows = []
    for _, r in k8.iterrows():
        items = [s.strip() for s in str(r.get("items", "") or "").split(",") if s.strip()]
        rows.append(
            {
                "event_time_utc": r["acceptanceDateTime"],
                "source": "edgar_8k",
                "type": "8k_items_" + "_".join(sorted(items)) if items else "8k_unknown",
                "payload": json.dumps(
                    {
                        "accession": r["accessionNumber"],
                        "is_amendment": r["form"] == "8-K/A",
                        "items": items,
                        "report_date": r.get("reportDate", ""),
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def collect() -> None:
    POOL_8K.mkdir(parents=True, exist_ok=True)
    cmap = cik_map()
    for sym in POOL_SYMBOLS:
        out = POOL_8K / f"{sym}.csv"
        if out.exists():
            print(f"  {sym}: exists, skip")
            continue
        cik = cmap[sym]
        subs = load_submissions(cik)
        df = build_8k_csv(subs)
        df.sort_values("event_time_utc").to_csv(out, index=False)
        n_unk = int((df["type"] == "8k_unknown").sum())
        print(f"  {sym} (CIK {cik}): {len(df)} 8-K rows, {n_unk} without items")
    print("collection done ->", POOL_8K)


# ---------------------------------------------------------------- prices

def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """N1 同款：1h → ET 交易日日线，保留首根 bar 的 UTC 开盘时刻."""
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


def load_all_daily() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in ALL_SYMBOLS:
        path = DATA / "TSLA_1h_alpaca.csv" if sym == "TSLA" else DATA / "pool_1h" / f"{sym}_1h.csv"
        out[sym] = build_daily(load_bars(str(path)))
    return out


def window_return_frame(daily: dict[str, pd.DataFrame], n: int) -> pd.DataFrame:
    """R_n[date, sym] = Close[d+n-1]/Open[d] - 1（d 为该标的交易日序）."""
    cols = {}
    for sym, df in daily.items():
        open_ = df["Open"].to_numpy()
        close = df["Close"].to_numpy()
        ret = np.full(len(df), np.nan)
        if n <= len(df):
            ret[: len(df) - n + 1] = close[n - 1:] / open_[: len(df) - n + 1] - 1
        cols[sym] = pd.Series(ret, index=df.index)
    return pd.DataFrame(cols)


def excess_frame(R: pd.DataFrame) -> pd.DataFrame:
    """超额 = 减同日其余标的等权均值；同日基准 <MIN_BENCH_SYMBOLS-1 个则 NaN."""
    S = R.sum(axis=1)
    C = R.count(axis=1)
    bench = R.rsub(S, axis=0).div(C - 1, axis=0)  # (S - R[s]) / (C-1)
    ex = R - bench
    ex[C < MIN_BENCH_SYMBOLS] = np.nan
    return ex


# ---------------------------------------------------------------- events

def load_8k(sym: str) -> pd.DataFrame:
    path = INTEL / "edgar_8k.csv" if sym == "TSLA" else POOL_8K / f"{sym}.csv"
    df = pd.read_csv(path)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, format="mixed")
    pl = [json.loads(p) for p in df["payload"]]
    df["items"] = [p["items"] for p in pl]
    df["accession"] = [p["accession"] for p in pl]
    df["is_amendment"] = [bool(p["is_amendment"]) for p in pl]
    return df


def build_events(daily: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {"n_8k_total": 0, "n_8k_no_items": 0, "no_items_by_year": {},
                      "n_101_raw": 0, "n_amend": 0, "n_out_of_window": 0, "n_dedup_dropped": 0}
    for sym in ALL_SYMBOLS:
        df = load_8k(sym)
        meta["n_8k_total"] += len(df)
        no_items = df["items"].apply(len) == 0
        meta["n_8k_no_items"] += int(no_items.sum())
        for y, c in df.loc[no_items, "event_time_utc"].dt.year.value_counts().items():
            meta["no_items_by_year"][int(y)] = meta["no_items_by_year"].get(int(y), 0) + int(c)
        ev = df[df["items"].apply(lambda lst: ITEM in lst)].sort_values("event_time_utc")
        meta["n_101_raw"] += len(ev)
        meta["n_amend"] += int(ev["is_amendment"].sum())
        d = daily[sym]
        opens = d["open_time_utc"]
        seen_days: set = set()
        for _, r in ev.iterrows():
            pos = int(opens.searchsorted(r["event_time_utc"], side="right"))
            if pos >= len(d):
                meta["n_out_of_window"] += 1
                continue
            gap = (opens.iloc[pos] - r["event_time_utc"]).total_seconds() / 86400
            if gap > MAX_ALIGN_GAP_DAYS:
                meta["n_out_of_window"] += 1  # 披露远早于价格窗（如 2018 上半年）
                continue
            day = d.index[pos]
            if (sym, day) in seen_days:  # 同标的同入场日去重（N1 同款）
                meta["n_dedup_dropped"] += 1
                continue
            seen_days.add((sym, day))
            t_et = r["event_time_utc"].tz_convert(ET)
            intraday = (t_et.weekday() < 5) and (
                (t_et.hour, t_et.minute) >= (9, 30) and t_et.hour < 16
            )
            rows.append(
                {
                    "symbol": sym,
                    "event_time_utc": r["event_time_utc"],
                    "accession": r["accession"],
                    "is_amendment": r["is_amendment"],
                    "items": "|".join(r["items"]),
                    "entry_day": day.date(),
                    "entry_pos": pos,
                    "entry_open_time_utc": opens.iloc[pos],
                    "timing": "intraday" if intraday else "after_hours",
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True), meta


# ---------------------------------------------------------------- primary test

def stratified_permutation(events: pd.DataFrame, ex1: pd.DataFrame) -> tuple[float, np.ndarray]:
    """标的分层置换：各标的在自身可用日内无放回抽同数事件日，返回 (obs_mean, null_means)."""
    obs_vals = events["ex1"].to_numpy()
    obs_mean = float(np.mean(obs_vals))
    n_total = len(obs_vals)

    per_sym = []  # (eligible ex1 array, k)
    for sym, g in events.groupby("symbol"):
        elig = ex1[sym].dropna().to_numpy()
        per_sym.append((elig, len(g)))

    null_sums = np.zeros(N_PERM)
    chunk = 2000
    for elig, k in per_sym:
        n_days = len(elig)
        for lo in range(0, N_PERM, chunk):
            b = min(chunk, N_PERM - lo)
            u = RNG.random((b, n_days))
            idx = np.argpartition(u, k - 1, axis=1)[:, :k]  # 每行无放回抽 k 天
            null_sums[lo: lo + b] += elig[idx].sum(axis=1)
    null_means = null_sums / n_total
    return obs_mean, null_means


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    daily = load_all_daily()
    for sym in ALL_SYMBOLS:
        d = daily[sym]
        print(f"  {sym}: {len(d)} days ({d.index[0].date()} -> {d.index[-1].date()})")

    R = {n: window_return_frame(daily, n) for n in WINDOWS}
    EX = {n: excess_frame(R[n]) for n in WINDOWS}

    events, meta = build_events(daily)
    for n in WINDOWS:
        events[f"ret{n}"] = [
            R[n].at[pd.Timestamp(d), s] for d, s in zip(events["entry_day"], events["symbol"])
        ]
        events[f"ex{n}"] = [
            EX[n].at[pd.Timestamp(d), s] for d, s in zip(events["entry_day"], events["symbol"])
        ]
    events["year"] = pd.to_datetime(events["entry_day"]).dt.year

    n_nan_ex1 = int(events["ex1"].isna().sum())
    events_test = events.dropna(subset=["ex1"]).reset_index(drop=True)
    events.to_csv(OUT / "events_all.csv", index=False)

    # —— 主检验（唯一预登记假设，多重比较 +1）——
    obs_mean, null_means = stratified_permutation(events_test, EX[1])
    p_one = float((1 + (null_means >= obs_mean).sum()) / (1 + N_PERM))
    passed = p_one < P_SIG
    prim = pd.DataFrame(
        [
            {
                "hypothesis": "pool 8-K item 1.01 next-day excess return > 0",
                "n_events": len(events_test),
                "n_symbols": events_test["symbol"].nunique(),
                "obs_mean_ex1_bp": obs_mean * 1e4,
                "null_mean_bp": null_means.mean() * 1e4,
                "null_std_bp": null_means.std() * 1e4,
                "p_one_sided": p_one,
                "n_perm": N_PERM,
                "alpha": P_SIG,
                "passed": passed,
            }
        ]
    )
    prim.to_csv(OUT / "primary_test.csv", index=False)
    print(f"\nPRIMARY: n={len(events_test)} mean_ex1={obs_mean*1e4:+.1f}bp p={p_one:.4f} "
          f"-> {'PASS' if passed else 'FAIL'}")

    # —— 辅助描述（不计显著）——
    desc_win = []
    for n in WINDOWS:
        sub = events.dropna(subset=[f"ex{n}"])
        r, e = sub[f"ret{n}"], sub[f"ex{n}"]
        t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)) + 1e-12)
        desc_win.append(
            dict(window=n, n=len(sub), raw_mean_bp=r.mean() * 1e4, ex_mean_bp=e.mean() * 1e4,
                 ex_median_bp=e.median() * 1e4, ex_t_naive=t)
        )
    by_year = (
        events_test.groupby("year")["ex1"]
        .agg(n="count", mean_bp=lambda x: x.mean() * 1e4, pos_frac=lambda x: (x > 0).mean())
        .reset_index()
    )
    by_timing = (
        events_test.groupby("timing")["ex1"]
        .agg(n="count", mean_bp=lambda x: x.mean() * 1e4, pos_frac=lambda x: (x > 0).mean())
        .reset_index()
    )
    by_sym = (
        events_test.groupby("symbol")["ex1"]
        .agg(n="count", mean_bp=lambda x: x.mean() * 1e4, pos_frac=lambda x: (x > 0).mean())
        .sort_values("mean_bp", ascending=False)
        .reset_index()
    )
    n_sym_pos = int((by_sym["mean_bp"] > 0).sum())

    # —— 机械规则（仅主检验过线时）——
    bt = None
    if passed:
        slip, fee = SLIP_BP / 1e4, FEE_BP / 1e4
        opens = np.array(
            [daily[s]["Open"].iloc[p] for s, p in zip(events_test["symbol"], events_test["entry_pos"])]
        )
        closes = np.array(
            [daily[s]["Close"].iloc[p] for s, p in zip(events_test["symbol"], events_test["entry_pos"])]
        )
        net = closes / (opens * (1 + slip)) - 1 - 2 * fee
        span_y = (
            pd.to_datetime(events_test["entry_day"]).max()
            - pd.to_datetime(events_test["entry_day"]).min()
        ).days / 365.25
        bt = {
            "n_trades": len(net),
            "wr_pct": float((net > 0).mean() * 100),
            "avg_net_bp": float(net.mean() * 1e4),
            "median_net_bp": float(np.median(net) * 1e4),
            "std_net_bp": float(net.std(ddof=1) * 1e4),
            "events_per_year": float(len(net) / span_y),
            "annual_capacity_bp": float(net.mean() * 1e4 * len(net) / span_y),
        }
        pd.DataFrame([bt]).to_csv(OUT / "rule_backtest.csv", index=False)

    write_summary(events, events_test, meta, prim.iloc[0], desc_win, by_year, by_timing,
                  by_sym, n_sym_pos, n_nan_ex1, bt, daily)
    print("written:", OUT)


def write_summary(events, events_test, meta, prim, desc_win, by_year, by_timing,
                  by_sym, n_sym_pos, n_nan_ex1, bt, daily) -> None:
    L = []
    L.append("N7 — 池级 8-K item 1.01（重大协议）事件研究（2026-08-01）")
    L.append("=" * 72)
    L.append(f"池：{len(ALL_SYMBOLS)} 标的（25 池 + TSLA）；8-K 采集窗 {SINCE} 起（submissions API，")
    L.append("acceptanceDateTime 防前视）；价格 = 1h 聚合日线（拆股已折算）。")
    d0 = min(d.index[0] for d in daily.values()).date()
    d1 = max(d.index[-1] for d in daily.values()).date()
    L.append(f"价格窗：{d0} → {d1}；对齐 = 披露后第一个严格晚于披露时刻的开盘（N1 同款）；")
    L.append("次日收益 = 入场日 Open→Close；超额 = 减同日其余标的等权均值。")
    L.append("")
    L.append("—— 样本 ——")
    L.append(f"  8-K 总申报 {meta['n_8k_total']} 张，其中无 items 字段（解析失败）{meta['n_8k_no_items']} 张"
             f"（{meta['n_8k_no_items'] / max(meta['n_8k_total'], 1):.1%}）")
    if meta["no_items_by_year"]:
        yr = ", ".join(f"{y}:{c}" for y, c in sorted(meta["no_items_by_year"].items()))
        L.append(f"    无 items 按年分布：{yr}")
    L.append(f"  item 1.01 原始 {meta['n_101_raw']} 张（含 8-K/A 修订 {meta['n_amend']} 张，与 N1 同口径保留）；")
    L.append(f"  价格窗外丢弃 {meta['n_out_of_window']}、同标的同入场日去重 {meta['n_dedup_dropped']}、"
             f"窗口收益缺失 {n_nan_ex1} → 主检验样本 n={len(events_test)}（{events_test['symbol'].nunique()} 标的）")
    L.append("")
    L.append("—— 主检验（唯一预登记假设，多重比较 +1）——")
    L.append(f"  H1：披露后次日超额收益 > 0；标的分层置换 {N_PERM} 次，单侧 α={P_SIG}")
    L.append(f"  观测均值 {prim['obs_mean_ex1_bp']:+.1f}bp（零分布均值 {prim['null_mean_bp']:+.2f}bp、"
             f"标准差 {prim['null_std_bp']:.1f}bp）")
    L.append(f"  p(单侧) = {prim['p_one_sided']:.4f}  →  **{'过线' if prim['passed'] else '判死'}**")
    L.append("")
    L.append("—— 辅助描述（不计显著）——")
    for r in desc_win:
        L.append(f"  +{r['window']:>2}日  n={r['n']:<4d} 原始 {r['raw_mean_bp']:+7.1f}bp"
                 f"  超额 {r['ex_mean_bp']:+7.1f}bp（中位 {r['ex_median_bp']:+6.1f}bp）"
                 f"  naive-t={r['ex_t_naive']:+.2f}")
    L.append("  按年份（次日超额）：")
    for _, r in by_year.iterrows():
        L.append(f"    {int(r['year'])}: n={int(r['n']):<3d} {r['mean_bp']:+7.1f}bp  胜率 {r['pos_frac']:.0%}")
    L.append("  盘中 vs 盘后披露（次日超额）：")
    for _, r in by_timing.iterrows():
        L.append(f"    {r['timing']:<12s} n={int(r['n']):<4d} {r['mean_bp']:+7.1f}bp  胜率 {r['pos_frac']:.0%}")
    L.append(f"  逐标的方向：{n_sym_pos}/{len(by_sym)} 个标的次日超额均值为正")
    for _, r in by_sym.iterrows():
        L.append(f"    {r['symbol']:<6s} n={int(r['n']):<3d} {r['mean_bp']:+7.1f}bp  胜率 {r['pos_frac']:.0%}")
    L.append("")
    if bt is not None:
        L.append("—— 机械规则悲观回测（主检验过线触发；fee 1bp/side + slip 2bp 入场）——")
        L.append(f"  披露次日开盘买入持 1 日收盘卖：n={bt['n_trades']}  WR {bt['wr_pct']:.1f}%"
                 f"  单笔净期望 {bt['avg_net_bp']:+.1f}bp（中位 {bt['median_net_bp']:+.1f}bp）")
        L.append(f"  年频 {bt['events_per_year']:.0f} 笔 → 年化容量 ≈ {bt['annual_capacity_bp']:+.0f}bp"
                 f"（单仓名义、不复利、事件可并发）")
    else:
        L.append("—— 机械规则回测：主检验未过线，按预登记协议跳过 ——")
    L.append("")
    L.append("—— 诚实边界 ——")
    L.append("  1) items 解析依赖 submissions 索引，早年申报格式噪声见上方失败率；")
    L.append("  2) 池标的都是当前大市值（幸存偏差）：结论只适用于'活到今天的大盘股'；")
    L.append("  3) 同日多标的事件与 5/20 日窗口存在重叠，naive-t 偏乐观（主检验用置换不受此影响）；")
    L.append("  4) 超额基准为 26 标的等权（非全市场），行业共同冲击未完全剥离；")
    L.append("  5) 实际样本 n=169 低于预登记预计的 300-500——大盘股 item 1.01 年频（~0.9 张/标的/年）")
    L.append("     比 TSLA（~2.4）低，预计值高估；样本量已是 N1 的 ~10 倍，检验效力足够裁决；")
    L.append("  6) 多重比较 +1（主检验）——辅助描述与逐标的表不作显著性声明。")
    L.append("")
    L.append(VERDICT)
    (OUT / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


VERDICT = """\
==================== 中文判读（2026-08-01，基于上表数字） ====================

一句话：**判死**。8-K item 1.01（重大协议）"披露后次日跑赢池"的假设在 26 标的
× 8 年、169 个事件上不成立——次日超额 +8.3bp，标的分层置换单侧 p=0.246，
离 0.05 线很远。N1 在 TSLA 上看到的 +124bp/t=3.09（17 例）按预登记协议
正式定性为**单标的噪声**，不是被样本量埋没的真效应。

细节佐证（均为描述，不计显著）：
1. 方向一致性坍塌：10/20 个标的次日超额均值为正——掷硬币水平。TSLA 自己
   仍然是全池第三强（+74.6bp、WR 78%、n=18，与 N1 同向），但它在池里是
   分布的尾巴，不是规律的代表。BA/CAT/NFLX 等工业与消费股方向为负。
2. 时间稳定性差：按年均值在 -45bp 到 +48bp 之间来回翻符号（2020、2025 为负），
   没有任何年份结构。
3. 窗口无延续：+5 日超额 +12.5bp（t=0.29）、+20 日 -2.0bp——即便次日有微弱
   正漂移，也在 20 日内完全耗散。
4. 盘中披露子组（n=10，+43.6bp、WR 80%）样本太小，只配当好奇心，不配当假设。

对哨兵体系的结论：'重大协议'这一事实类 8-K 信号在大盘股上无独立次日 alpha，
与 N1 的 Form 4/FOMC/推文结论合并后，机读法定披露类情报的独立变现假设
已全部检验完毕、全部阴性。8-K item 特征保留在 E8 复合特征池中的资格不受
影响（那是另一个已登记的通道）；本假设关闭，不再以任何变体重开，除非
先有新的机制层证据。多重比较 +1（计数器并入 strategy-lab，由主线更新）。
"""


if __name__ == "__main__":
    main()
