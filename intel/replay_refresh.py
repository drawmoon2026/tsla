"""推演引擎每日续演 — 把「静态存档回放」升级为「每日续演到最新完整交易日」.

每次运行（幂等；输入没变化时快速退出，适合挂在仪表盘 5 分钟生成链上）：

1. 数据续接：data/TSLA_5m_rolling.csv（首次从冻结源 data/TSLA_5m_3y.csv 起步），
   经 Alpaca API 增量补 5m RTH bar 到最新完整交易日（盘中不收当日残段；
   取数失败降级用现有滚动数据，原因如实写入 meta，不装新鲜）。
   P0-2 拆股防护：续接前对旧尾时刻做重叠收盘核对（差 >5% 中止）+ 首根新 bar
   隔夜跳变核对（>25% 中止），中止时向 events 发 price_splice_alert 告警事件，
   等待人工重建 rolling CSV——拆股后新旧口径静默拼接会造成假崩盘/S2 假触发。

2. 交易续算（冻结管线零改动，参数全部读 models/e8a/meta.json）：
   - 留出段 = outputs/e8a_replay 冻结存档（research/e8a_trades_export.py 产物，
     缺失自动重建），先复核逐笔数字与 frontier_shift / e11 存档一致再原样并入；
   - 前向段 = 对滚动数据整段重跑冻结管线：src/v_stats 网格并集去重检测 →
     research/ml_common.build_dataset 特征 → e8_pooled 波动率标准化
     （rv20_med_train 用 meta.json 冻结值，不重算）→ model.joblib 打分 →
     冻结门控 → E9 悲观结算（research/e8a_trades_export.detail_trade 同一函数），
     只取入场在冻结数据边界（data/TSLA_5m_3y.csv 末 bar）之后的新交易，
     标记 segment="forward"（2026-07-24 冻结日之后属真前向）；
     S2（E11 冻结口径）拦截笔照常结算、blocked=1 不入权益；
   - 管线漂移防线：重算的留出段事件概率必须与 models/e8a/holdout_ref.csv
     逐事件一致（容差 1e-6）、留出段 62/54 笔统计与存档一致——任一漂移即中止，
     绝不带错续演。

3. 探测器逐日状态合并（统一 daily.csv）：
   - 历史推演段：outputs/n3h_deduction/daily_states.csv（2023-07 → Musk 归档尽头，
     det_src=n3h；文件内 blind 行 det_src=blind）；
   - 归档尽头 → 前向值班上线之间的空窗：det_src=blind（失明如实标注）；
   - 2026-07-24 起：data/intel/sentinel.sqlite detector_state 表
     （det_src=live；标定期 CALIBRATING 如实标注，不出信号）。

4. 输出统一推演数据集 outputs/replay_current/{trades.csv, daily.csv, meta.json}
   （meta 含算法版本串、数据截至日、分段口径与全部核对结果），
   供 intel/dashboard.py 全系统推演页读取；旧 outputs/e8a_replay 保持不动，
   页面在本数据集缺失时自动降级回它。

已知口径边界（如实声明）：
- 最新一日收盘前 ~2 小时内确认的事件，其训练标签在当日数据内无法结算
  （build_dataset label_unresolved 丢行），会在下一交易日数据到位后自动补入——
  前向段逐日重算，不会永久漏单，只滞后一个数据日；
- 前向段起点 = 冻结数据边界次日（2026-07-23），冻结声明日为 2026-07-24：
  段内首个交易日严格说是「冻结前最后一日」，meta 与页面均如实标注。

用法：
    .venv/bin/python -m intel.replay_refresh            # 增量续演（幂等）
    .venv/bin/python -m intel.replay_refresh --force    # 忽略输入签名强制重算
    .venv/bin/python -m intel.replay_refresh --offline  # 跳过取数（只用现有数据）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ET = ZoneInfo("America/New_York")

SEED_CSV = ROOT / "data" / "TSLA_5m_3y.csv"          # 冻结数据源（起步/边界）
ROLLING_CSV = ROOT / "data" / "TSLA_5m_rolling.csv"  # 滚动数据（本脚本维护）
FROZEN_DIR = ROOT / "outputs" / "e8a_replay"         # 留出段冻结存档
STATES_CSV = ROOT / "outputs" / "n3h_deduction" / "daily_states.csv"
MODEL_DIR = ROOT / "models" / "e8a"
DB_PATH = ROOT / "data" / "intel" / "sentinel.sqlite"
OUT = ROOT / "outputs" / "replay_current"

HOLDOUT_START = date(2025, 10, 1)
FREEZE_DAY = date(2026, 7, 24)

ALGO_VERSION = ("E8-A+S2 冻结 2026-07-24（models/e8a）× N3-H 探测器冻结 2026-07-24"
                " · replay_refresh r1")


# ---- 拆股/口径断裂防护（P0-2）：续接前重叠核对 + 隔夜跳变阈值 -----------------
SPLICE_OVERLAP_TOL_PCT = 5.0   # 重叠 bar（同一时刻）收盘差异超此值 → 中止续接
SPLICE_GAP_TOL_PCT = 25.0      # 首根新 bar 相对旧尾的隔夜跳变超此值 → 中止续接


# ------------------------------------------------------------- small helpers

def _last_line(path: Path) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        return f.read().splitlines()[-1].decode()


def _last_ts(path: Path) -> pd.Timestamp:
    return pd.Timestamp(_last_line(path).split(",")[0])


def _last_close(path: Path) -> float:
    # 列序固定：Datetime,Open,High,Low,Close,Volume
    return float(_last_line(path).split(",")[4])


def _price_splice_alert(kind: str, title: str, payload: dict) -> None:
    """向 events 表发 price_splice_alert 告警事件（仪表盘情报流可见）；失败只打印."""
    try:
        from intel import store

        conn = store.connect()
        try:
            store.upsert_source(conn, {
                "source_id": "replay_refresh",
                "name": "推演引擎数据续接守卫（价格线拆股/口径断裂防护）",
                "tier": "T1",
                "method": "derived",
                "poll_interval_s": 0,
                "cost": "free",
                "weight_source": 0.5,
                "notes": "非采集渠道：rolling CSV 续接前的重叠核对与隔夜跳变防护（P0-2）",
            })
            store.insert_events(conn, "replay_refresh", [{
                "dedupe_key": f"price_splice_{kind}_{datetime.now(timezone.utc).date()}",
                "event_time_utc": store.utcnow_iso(),
                "symbol": "TSLA",
                "type": "price_splice_alert",
                "title": title,
                "payload": payload,
            }])
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[rolling] 告警事件写入失败（不影响中止决定）：{type(e).__name__}: {e}")


def _detector_rows() -> list[dict]:
    """detector_state 表（前向值班）；表/库缺失 → []（不炸）."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT state_date, state, baseline_days, risk_off_until, "
                "updated_utc FROM detector_state ORDER BY state_date"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _states_file_rows() -> list[dict]:
    """n3h daily_states.csv（历史推演）；缺失 → []."""
    if not STATES_CSV.exists():
        return []
    import csv as _csv

    out = []
    with STATES_CSV.open(newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            try:
                out.append({
                    "date": date.fromisoformat(r["date"].strip()),
                    "state": (r.get("state") or "").strip(),
                    "blind": (r.get("tweet_data_blind") or "").strip() == "True",
                    "trig": (r.get("trigger") or "").strip() == "True",
                })
            except ValueError:
                continue
    return out


# ------------------------------------------------------ 1. 数据续接（Alpaca）

def refresh_rolling(now: datetime, offline: bool) -> dict:
    """把滚动 CSV 补到最新完整交易日；返回取数实况（永不抛异常）."""
    info: dict = {"ok": True, "error": None, "new_bars": 0, "new_days": [],
                  "seeded": False, "skipped": bool(offline)}
    if not ROLLING_CSV.exists():
        if not SEED_CSV.exists():
            info.update(ok=False, error=f"起步数据缺失：{SEED_CSV}")
            return info
        shutil.copyfile(SEED_CSV, ROLLING_CSV)
        info["seeded"] = True
        print(f"[rolling] 起步：复制 {SEED_CSV.name} -> {ROLLING_CSV.name}")
    if offline:
        return info
    try:
        from src.alpaca_data import fetch_bars, filter_rth, load_keys

        last_ts = _last_ts(ROLLING_CSV)
        last_close = _last_close(ROLLING_CSV)
        start = last_ts + pd.Timedelta(minutes=5)
        end = pd.Timestamp(now) - pd.Timedelta(minutes=17)  # free tier: >15min 前
        if start >= end:
            return info
        key, secret = load_keys()
        # P0-2 拆股防护：取数起点前移，覆盖旧尾时刻做重叠核对
        fetch_start = last_ts - pd.Timedelta(days=5)
        df = fetch_bars("TSLA", fetch_start, end, "5Min", key, secret)
        df = filter_rth(df)
        # P0-B 数据口径披露：实际使用的 feed 记入 meta（sip 被 403 拒会静默回落 iex，
        # 此处把回落结果落盘，页面据此显示口径）
        info["feed"] = df.attrs.get("feed")
        # ① 重叠核对：同一时刻（旧 CSV 末 bar）的收盘必须一致（拆股后 Alpaca
        #    adjustment="split" 会重写历史为拆后口径，与 CSV 旧段断裂）
        overlap = df[df.index == last_ts]
        if len(overlap):
            new_close = float(overlap["Close"].iloc[0])
            diff_pct = abs(new_close / last_close - 1) * 100
            info["splice_check"] = {"overlap_bar": True,
                                    "csv_close": last_close, "api_close": new_close,
                                    "diff_pct": round(diff_pct, 3)}
            if diff_pct > SPLICE_OVERLAP_TOL_PCT:
                msg = (f"价格口径断裂（疑似拆股）：旧尾 {last_ts} 收盘 CSV={last_close:.4f} "
                       f"vs API={new_close:.4f}（差 {diff_pct:.1f}% > "
                       f"{SPLICE_OVERLAP_TOL_PCT:.0f}%）——中止续接，等待人工重建 rolling CSV")
                print(f"[rolling] {msg}")
                _price_splice_alert("overlap", "推演数据续接中止：" + msg,
                                    {"bar_ts": str(last_ts), "csv_close": last_close,
                                     "api_close": new_close, "diff_pct": diff_pct,
                                     "tol_pct": SPLICE_OVERLAP_TOL_PCT})
                info.update(ok=False, error=msg)
                return info
        else:
            info["splice_check"] = {"overlap_bar": False,
                                    "note": "API 未回旧尾时刻 bar，重叠核对跳过，"
                                            "仅靠隔夜跳变防线"}
        df = df[df.index > last_ts]
        # 盘中运行：当日残段不收（只续到最新完整交易日）。当日完整性截止时刻
        # 按统一日历收盘 +5min（半日市 13:05 即可收当日；原固定 16:05）
        from intel.market_calendar import close_time_et
        now_et = now.astimezone(ET)
        done_t = (datetime.combine(now_et.date(), close_time_et(now_et.date()))
                  + timedelta(minutes=5)).time()
        if now_et.time() < done_t:
            et_dates = df.index.tz_convert("America/New_York").date
            df = df[np.asarray(et_dates) != now_et.date()]
        # ② 隔夜跳变防线：首根新 bar 相对旧尾收盘跳变超阈值 → 中止（拆股日
        #    最小因子 2 产生 ~50% 跳变；TSLA 真实隔夜极值远小于 25%）
        if len(df):
            first_open = float(df["Open"].iloc[0])
            gap_pct = abs(first_open / last_close - 1) * 100
            if gap_pct > SPLICE_GAP_TOL_PCT:
                msg = (f"隔夜跳变异常（疑似拆股）：旧尾收盘 {last_close:.4f} → "
                       f"首根新 bar {df.index[0]} 开盘 {first_open:.4f}"
                       f"（跳变 {gap_pct:.1f}% > {SPLICE_GAP_TOL_PCT:.0f}%）"
                       "——中止续接，等待人工重建 rolling CSV")
                print(f"[rolling] {msg}")
                _price_splice_alert("gap", "推演数据续接中止：" + msg,
                                    {"last_close": last_close, "first_open": first_open,
                                     "first_bar_ts": str(df.index[0]),
                                     "gap_pct": gap_pct, "tol_pct": SPLICE_GAP_TOL_PCT})
                info.update(ok=False, error=msg)
                return info
        if len(df):
            df.to_csv(ROLLING_CSV, mode="a", header=False,
                      date_format="%Y-%m-%dT%H:%M:%S%z")
            days = sorted({str(d) for d in df.index.tz_convert("America/New_York").date})
            info.update(new_bars=int(len(df)), new_days=days)
            print(f"[rolling] +{len(df)} bars（{days[0]} → {days[-1]}）")
        else:
            print("[rolling] 无新完整交易日数据")
    except SystemExit as e:          # alpaca_data 用 SystemExit 报「无数据/无钥匙」
        msg = str(e)
        if "no bars" in msg.lower():
            print("[rolling] Alpaca 区间内无 bar（非交易日）")
        else:
            info.update(ok=False, error=msg)
    except Exception as e:  # noqa: BLE001 — 取数失败降级用现有数据
        info.update(ok=False, error=f"{type(e).__name__}: {e}")
    if info["error"]:
        print(f"[rolling] 取数失败（降级用现有滚动数据）：{info['error']}")
    return info


# ------------------------------------------------- 2. 冻结管线：事件→打分→结算

def frozen_segment() -> tuple[pd.DataFrame, dict]:
    """留出段冻结存档（缺失自动重建）+ 逐笔数字复核（漂移即中止）."""
    if not (FROZEN_DIR / "summary.json").exists() \
            or not (FROZEN_DIR / "trades.csv").exists():
        print("[frozen] outputs/e8a_replay 缺失——重建冻结存档 ...")
        from research.e8a_trades_export import main as export_main
        export_main()
    ftr = pd.read_csv(FROZEN_DIR / "trades.csv")
    fsum = json.loads((FROZEN_DIR / "summary.json").read_text(encoding="utf-8"))

    from research.e8a_trades_export import EXP_ALL, EXP_S2

    r = ftr["ret"].to_numpy()
    kept = ftr[~ftr["blocked"]]
    rk = kept["ret"].to_numpy()
    got_all = {"n": len(ftr), "wr": float((r > 0).mean()),
               "avg_bp": float(r.mean() * 1e4), "total": float(np.prod(1 + r) - 1)}
    got_s2 = {"n": len(kept), "wr": float((rk > 0).mean()),
              "avg_bp": float(rk.mean() * 1e4), "total": float(np.prod(1 + rk) - 1)}
    for tag, got, exp in (("全 62 笔", got_all, EXP_ALL), ("S2 后 54 笔", got_s2, EXP_S2)):
        ok = (got["n"] == exp["n"] and abs(got["wr"] - exp["wr"]) < 1e-9
              and abs(got["avg_bp"] - exp["avg_bp"]) < 0.05
              and abs(got["total"] - exp["total"]) < 1e-6)
        print(f"[frozen] {tag}: n={got['n']} WR={got['wr']:.2%} "
              f"avg={got['avg_bp']:+.2f}bp total={got['total']:+.2%} "
              + ("与存档一致" if ok else "与存档不一致"))
        if not ok:
            raise SystemExit(f"[frozen] 留出段（{tag}）数字漂移——中止续演")
    return ftr, fsum


def score_all_events(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """冻结检测/特征/推断管线整段重跑：返回逐事件 prob 表 + 管线信息."""
    import joblib

    from research.ml_common import build_dataset
    from src.v_stats import run_grid_for_tf

    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    vd = meta["v_detection"]
    print("[pipeline] v_stats 网格检测（36 组合并集去重，冻结参数）...")
    _, ev, _ = run_grid_for_tf(
        bars, "5m", vd["grid"]["x"], vd["grid"]["y"], vd["grid"]["t"],
        vd["lookback"], vd["lookforward"], vd["max_drop_bars"])
    ev = ev.sort_values("trough_confirm_t").reset_index(drop=True)
    print(f"[pipeline] 物理事件 {len(ev)}；构建特征（ml_common.build_dataset）...")
    ds, info = build_dataset(ev, bars)

    # e8_pooled 波动率标准化 —— rv20_med_train 用冻结值，绝不重算
    rv = ds["rv20"].replace(0, np.nan)
    ds["drop_pct_n"] = ds["drop_pct"] / rv
    ds["drop_speed_n"] = ds["drop_speed"] / rv
    ds["open_to_now_ret_n"] = ds["open_to_now_ret"] / rv
    ds["prev_day_ret_n"] = ds["prev_day_ret"] / rv
    ds["rv20_rel"] = ds["rv20"] / float(meta["tsla_rv20_med_train"])
    feats = meta["features"]
    keep = ds[[f for f in feats if f != "symbol"]].notna().all(axis=1)
    ds = ds[keep].reset_index(drop=True)
    ds["symbol"] = pd.Categorical(["TSLA"] * len(ds),
                                  categories=meta["symbol_categories"])

    model = joblib.load(MODEL_DIR / "model.joblib")
    ds["prob"] = model.predict_proba(ds[feats])[:, 1]
    ds["epos"] = bars.index.get_indexer(pd.DatetimeIndex(ds["entry_t"]))
    assert (ds["epos"] >= 0).all()
    info["n_scored"] = len(ds)
    return ds, {"meta": meta, "build_info": info}


def check_prob_parity(ds: pd.DataFrame, n_frozen: int) -> dict:
    """漂移防线：留出段事件概率必须与 holdout_ref.csv 逐事件一致."""
    ref = pd.read_csv(MODEL_DIR / "holdout_ref.csv")
    ref["entry_t"] = pd.to_datetime(ref["entry_t"], utc=True)
    # holdout_ref 只覆盖留出段（et_day >= 2025-10-01），比对范围与之对齐
    old = ds[(ds["epos"] < n_frozen) & (ds["et_day"] >= HOLDOUT_START)]
    new_by_t: dict = {}
    for t, g in old.groupby("entry_t"):
        new_by_t[t] = sorted(g["prob"].tolist())
    missing, max_diff = 0, 0.0
    for t, g in ref.groupby("entry_t"):
        want = sorted(g["prob"].tolist())
        have = new_by_t.get(t)
        if have is None or len(have) < len(want):
            missing += 1
            continue
        # 同 entry 事件数可能因边界补结算而增多；冻结事件必须逐个可对上
        for w in want:
            diff = min(abs(w - h) for h in have)
            max_diff = max(max_diff, diff)
    n_new_only = sum(1 for t in new_by_t if t not in set(ref["entry_t"]))
    out = {"n_ref_events": int(len(ref)), "n_recomputed_frozen_range": int(len(old)),
           "max_prob_diff": float(max_diff), "missing_ref_entries": int(missing),
           "boundary_new_events": int(n_new_only)}
    print(f"[parity] holdout_ref {len(ref)} 事件复算：max|Δprob|={max_diff:.2e} "
          f"缺失={missing} 边界新增（不入冻结段）={n_new_only}")
    if missing or max_diff > 1e-6:
        raise SystemExit("[parity] 冻结管线复算与 holdout_ref 漂移——中止续演")
    return out


def forward_trades(bars, bs, ds: pd.DataFrame, n_frozen: int,
                   ftr: pd.DataFrame, gate: float,
                   s2_by_day: dict) -> pd.DataFrame:
    """冻结边界之后的新交易（同一非重叠流续接留出段，E9 悲观结算）."""
    from research.e8a_trades_export import detail_trade
    from src.common.data_io import ET as ET_STR

    # 流续接点：重放留出段最后一笔，拿到其出场 bar（并再验一次结算一致）
    last = ftr.iloc[-1]
    d_last = detail_trade(bs, int(last["epos"]))
    assert abs(d_last["ret"] - float(last["ret"])) < 1e-9, \
        "留出段末笔在滚动数据上结算漂移"
    busy = int(d_last["exit_pos"])
    eq = float(ftr["cum_eq_s2"].dropna().iloc[-1]) if ftr["cum_eq_s2"].notna().any() else 1.0
    seq = int(ftr["seq"].max())

    cand = (ds[(ds["epos"] >= n_frozen) & (ds["prob"] >= gate)]
            .groupby("epos", as_index=True)["prob"].max().sort_index())
    rows = []
    for epos, prob in cand.items():
        epos = int(epos)
        if epos <= busy:
            continue
        d = detail_trade(bs, epos)
        busy = int(d["exit_pos"])
        ts_e = bars.index[epos].tz_convert(ET_STR)
        ts_x = d["exit_t"].tz_convert(ET_STR)
        et_day = ts_e.date()
        blocked = bool(s2_by_day.get(et_day, False))
        seq += 1
        if not blocked:
            eq *= 1 + d["ret"]
        rows.append({
            "seq": seq, "epos": epos,
            "entry_t_utc": bars.index[epos].isoformat(),
            "entry_et": ts_e.strftime("%Y-%m-%d %H:%M"),
            "et_day": str(et_day),
            "prob": round(float(prob), 6),
            "entry_open": float(bs.opens[epos]),
            "entry_fill": round(d["entry_px"], 4),
            "exit_et": ts_x.strftime("%Y-%m-%d %H:%M"),
            "exit_px": round(d["exit_px"], 4),
            "exit_type": d["exit_type"],
            "ret": round(d["ret"], 8),
            "blocked": blocked,
            "cum_eq_s2": round(eq, 8) if not blocked else np.nan,
            "segment": "forward",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------- 3. 探测器逐日状态合并

def detector_daily(all_days: list[date]) -> tuple[dict, dict]:
    """date -> {state, src, trig}；src ∈ n3h|blind|live。附 meta 用来源说明."""
    hist = _states_file_rows()
    live = _detector_rows()
    m: dict[date, dict] = {}
    for r in hist:
        m[r["date"]] = {"state": r["state"].lower(),
                        "src": "blind" if r["blind"] else "n3h",
                        "trig": r["trig"]}
    live_first = None
    for r in live:
        d = date.fromisoformat(r["state_date"])
        live_first = live_first or d
        m[d] = {"state": r["state"].lower(), "src": "live",
                "trig": False, "baseline_days": r.get("baseline_days")}
    # 覆盖不到的交易日（归档尽头→值班上线之间、值班漏日）一律如实标失明
    for d in all_days:
        if d not in m:
            m[d] = {"state": "", "src": "blind", "trig": False}
    n3h_days = [r["date"] for r in hist if not r["blind"]]
    blind_days = [r["date"] for r in hist if r["blind"]]
    src_note = {
        "n3h": (f"{min(n3h_days)} → {max(n3h_days)} 历史推演"
                "（outputs/n3h_deduction/daily_states.csv）") if n3h_days else "无",
        "blind": (f"{min(blind_days)} 起 Musk 归档尽头失明段"
                  "（无放风腿数据，无触发 ≠ 安全）") if blind_days else "无",
        "live": (f"{live_first} 起 detector_state 前向值班"
                 "（标定期 CALIBRATING 如实标注）") if live_first else
                "尚无前向值班行（detector_state 表空）",
    }
    cur = live[-1] if live else None
    det_meta = {
        "rule": "N3-H 冻结 2026-07-24（intel/detector.py，参数不许改）",
        "current_state": cur["state"] if cur else None,
        "current_state_date": cur["state_date"] if cur else None,
        "baseline_days": cur["baseline_days"] if cur else None,
        "sources": src_note,
    }
    return m, det_meta


# ------------------------------------------------------------------ assemble

def build(force: bool, offline: bool) -> None:
    now = datetime.now(timezone.utc)
    fetch = refresh_rolling(now, offline)
    if not ROLLING_CSV.exists():
        raise SystemExit(f"滚动数据不可用：{fetch['error']}")

    det_rows = _detector_rows()
    sig = {
        "algo_version": ALGO_VERSION,
        "rolling_last": str(_last_ts(ROLLING_CSV)),
        "rolling_size": ROLLING_CSV.stat().st_size,
        "detector_n": len(det_rows),
        "detector_last": det_rows[-1]["state_date"] if det_rows else None,
        "detector_updated": det_rows[-1]["updated_utc"] if det_rows else None,
        "states_csv": ([int(STATES_CSV.stat().st_mtime), STATES_CSV.stat().st_size]
                       if STATES_CSV.exists() else None),
        "frozen_mtime": (int((FROZEN_DIR / "summary.json").stat().st_mtime)
                         if (FROZEN_DIR / "summary.json").exists() else None),
    }
    old_meta = None
    if (OUT / "meta.json").exists():
        try:
            old_meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_meta = None
    if not force and old_meta and old_meta.get("inputs_sig") == sig:
        print(f"[replay_refresh] 输入无变化（数据截至 "
              f"{old_meta['data']['data_through']}）——已是最新，跳过重算")
        if not (OUT / "sandbox.json").exists():   # 沙盘缺失单独补（不整链重算）
            _refresh_sandbox()
        return

    from research.e9_frontier_search import BarSet
    from src.common.data_io import ET as ET_STR, load_bars

    print(f"[replay_refresh] 加载滚动数据 {ROLLING_CSV.name} ...")
    bars = load_bars(str(ROLLING_CSV))
    bs = BarSet(bars)
    et_days_arr = bars.index.tz_convert(ET_STR).date
    data_through = et_days_arr[-1]

    # 冻结数据边界（TSLA_5m_3y.csv 末 bar；文件缺失则退回上次 meta 记录）
    if SEED_CSV.exists():
        seed_last = _last_ts(SEED_CSV)
    elif old_meta and old_meta.get("frozen_boundary_utc"):
        seed_last = pd.Timestamp(old_meta["frozen_boundary_utc"])
    else:
        raise SystemExit("冻结边界不可知：TSLA_5m_3y.csv 与历史 meta 均缺失")
    n_frozen = int(bars.index.get_loc(seed_last)) + 1
    frozen_end_day = et_days_arr[n_frozen - 1]

    # 留出段冻结存档 + 复核
    ftr, fsum = frozen_segment()
    # 滚动数据与冻结段对齐防线：62 笔入场时刻在滚动索引里的位置必须等于存档 epos
    chk = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(ftr["entry_t_utc"])))
    assert (chk == ftr["epos"].to_numpy()).all(), "滚动数据与冻结数据边界错位"

    # 冻结管线整段重跑 + 概率漂移防线
    ds, pinfo = score_all_events(bars)
    parity = check_prob_parity(ds, n_frozen)
    meta_m = pinfo["meta"]
    gate = float(meta_m["gate"]["threshold"])

    # S2 逐日状态（E11 冻结口径，与 e8a_trades_export 同式）
    dc = bars["Close"].groupby(et_days_arr).last()
    s2_state = (dc / dc.rolling(252).max() - 1.0 < -0.20).shift(1).fillna(False)
    s2_by_day = dict(zip(dc.index, s2_state.astype(bool)))

    # 前向段交易续算
    fwd = forward_trades(bars, bs, ds, n_frozen, ftr, gate, s2_by_day)
    n_fwd_kept = int((~fwd["blocked"]).sum()) if len(fwd) else 0
    n_fwd_blk = int(fwd["blocked"].sum()) if len(fwd) else 0

    # 合并 trades.csv（留出段原样 + segment 列）
    ftr_out = ftr.copy()
    ftr_out["segment"] = "holdout"
    trades = pd.concat([ftr_out, fwd], ignore_index=True) if len(fwd) else ftr_out

    # 统一 daily.csv：全滚动区间 + S2 + 分段 + 探测器状态
    all_days = list(dc.index)
    det_map, det_meta = detector_daily(all_days)
    drows = []
    for d in all_days:
        seg = ("pre" if d < HOLDOUT_START
               else "holdout" if d <= frozen_end_day else "forward")
        dm = det_map.get(d, {"state": "", "src": "blind", "trig": False})
        drows.append({"et_day": str(d), "close": round(float(dc[d]), 4),
                      "s2_off": bool(s2_by_day[d]), "seg": seg,
                      "det_state": dm["state"], "det_src": dm["src"],
                      "det_trig": bool(dm.get("trig", False))})
    daily = pd.DataFrame(drows)

    # 前向段统计
    fwd_days = [d for d in all_days if d > frozen_end_day]
    fk = fwd[~fwd["blocked"]] if len(fwd) else fwd
    fwd_stats = {
        "start": str(fwd_days[0]) if fwd_days else None,
        "end": str(data_through) if fwd_days else None,
        "trading_days": len(fwd_days),
        "n_trades": n_fwd_kept, "n_blocked": n_fwd_blk,
        "wr": (float((fk["ret"] > 0).mean()) if n_fwd_kept else None),
        "avg_bp": (float(fk["ret"].mean() * 1e4) if n_fwd_kept else None),
        "total": (float(np.prod(1 + fk["ret"]) - 1) if n_fwd_kept else 0.0),
        "s2_off_days": int(sum(bool(s2_by_day[d]) for d in fwd_days)),
        "s2_off_all": bool(fwd_days) and all(bool(s2_by_day[d]) for d in fwd_days),
        "post_freeze_start": str(FREEZE_DAY),
        "note": ("前向段自冻结数据边界次日起；2026-07-24 冻结日之后为真样本外。"
                 "S2 关闸日不开新仓：0 新交易在关闸期属正确结果，不是数据缺失。"),
    }
    kept_h = ftr[~ftr["blocked"]]
    hold_stats = {
        "start": str(HOLDOUT_START), "end": str(frozen_end_day),
        "n_trades": int(len(kept_h)), "n_blocked": int(ftr["blocked"].sum()),
        "wr": float((kept_h["ret"] > 0).mean()),
        "avg_bp": float(kept_h["ret"].mean() * 1e4),
        "total": float(np.prod(1 + kept_h["ret"]) - 1),
        "verified": "与 frontier_shift / e11 switch_table 存档核对一致",
    }

    meta_out = {
        "version": ALGO_VERSION,
        "generated_utc": now.isoformat(timespec="seconds"),
        "inputs_sig": sig,
        "frozen_boundary_utc": str(seed_last),
        "data": {
            "bars": str(ROLLING_CSV.relative_to(ROOT)),
            "data_through": str(data_through),
            "n_bars": int(len(bars)), "n_days": len(all_days),
            "fetch": fetch,
        },
        "algo": {
            "strategy": meta_m["strategy"], "frozen_at": meta_m["frozen_at"],
            "gate": gate, "geometry": meta_m["geometry"],
            "s2_switch": meta_m["s2_switch"],
            "model": "models/e8a/model.joblib（LightGBM leave-TSLA-out，冻结）",
        },
        "detector": det_meta,
        "segments": {"holdout": hold_stats, "forward": fwd_stats},
        "holdout_summary": fsum,
        "checks": {
            "frozen_trades": "留出段 62/54 笔与存档核对一致",
            **parity,
            "pipeline_drops": pinfo["build_info"]["drops"],
        },
        "notes": [
            "冻结参数零改动：检测网格/特征/模型/门控/几何/S2 全部读 models/e8a 冻结件。",
            "最新一日尾段（<2h）确认的事件因标签未结算暂缺，下一数据日自动补入。",
            "探测器 daily 来源分段见 detector.sources；未覆盖交易日一律标 blind。",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT / "trades.csv", index=False)
    daily.to_csv(OUT / "daily.csv", index=False)
    (OUT / "meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    s2_now = "S2 停用中" if s2_by_day.get(data_through) else "S2 开闸"
    print(f"[replay_refresh] 数据截至 {data_through}（{s2_now}）· "
          f"留出段 {hold_stats['n_trades']}+{hold_stats['n_blocked']}拦 · "
          f"前向段 {n_fwd_kept} 笔成交 + {n_fwd_blk} 笔 S2 拦截"
          f"（{fwd_stats['trading_days']} 交易日）")
    print(f"written: {OUT}/trades.csv, daily.csv, meta.json")
    _refresh_sandbox(bars=bars, bs=bs)


def _refresh_sandbox(bars=None, bs=None) -> None:
    """what-if 沙盘预结算（intel/sandbox_export）——失败不拖垮续演链，如实打印."""
    try:
        from intel.sandbox_export import export_sandbox
        export_sandbox(bars=bars, bs=bs)
    except SystemExit as e:            # 导出侧核对失败（漂移中止）——如实转告
        print(f"[replay_refresh] 沙盘导出中止：{e}")
    except Exception as e:  # noqa: BLE001
        print(f"[replay_refresh] 沙盘导出失败（页面将降级为生成指引）："
              f"{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="推演引擎每日续演（冻结管线零改动）")
    ap.add_argument("--force", action="store_true", help="忽略输入签名强制重算")
    ap.add_argument("--offline", action="store_true", help="跳过 Alpaca 取数")
    args = ap.parse_args()
    build(force=args.force, offline=args.offline)


if __name__ == "__main__":
    main()
