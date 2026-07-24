"""N3 因果探测器 — 哨兵常驻值班组件（前向虚拟推演，不碰真钱）.

冻结规则（N3-H 2026-07-24 冻结版，本文件只做前向实现，参数不许改）：
- 触发 = Musk 密集发帖日（act = 次交易日）× 回看 20 交易日内有空头利益
  up-jump 发布（change_pct >= +10%，发布时刻早于密集日结束）
- risk-off 持续 = F20：触发日起 20 个交易日，重叠触发顺延
- 定位 = 窄谱风险过滤器（只防"空头知情型下跌"），不是全天候减仓开关

标定问题（docs/strategy-lab.md N3 条目预先声明）：
前向放风腿来自 x_nitter RSS（只含 post+RT，无 reply；每账号每次仅最近 20 帖），
与历史 Sprinklr 归档（含 reply）口径不同——历史密集参考值 65 帖/日不可直接用。
实现 = 分位数映射标定（这是标定实现，不是改规则）：
- 从 musk_tweets.csv 全档（2018-01-03 → 2025-05-08，2683 个自然日，0 补齐）
  算出 65 帖/日的经验分位数 P(count<=65) = 0.8789，写死为 DENSE_QUANTILE；
- 前 CALIB_BDAYS=20 个交易日为标定期：只累积 nitter 口径的日计数基线
  （逐日存 detector_state.musk_window_json），状态输出 CALIBRATING，不出信号；
- 期满后阈值 = nitter 基线（扩张窗，含标定期后继续累积）的同分位数，每日重算。

已知口径边界（如实声明，不修饰）：
- nitter RT 的时间是原帖时间；实例宕机期间日计数会低估（poll_log 可查）；
- 交易日用 numpy busday（周一至周五）近似，不剔美股假日，F20 因此可能偏移 ~1 日；
- 首个标定日覆盖的自然日可能因轮询未满一天而低估——都是基线噪声，标定期本身
  就是为吸收这类口径差而设。

每次哨兵轮询后运行（run_sentinel 注册在全部采集器之后）：
- 逐交易日在 detector_state 写/更新一行状态（CALIBRATING / RISK_ON / RISK_OFF）
- 状态切换时向 events 表发一条 source=detector 事件（title 含状态与两腿数值，
  仪表盘情报流自动显示）
- RISK_OFF 触发记一条假想减仓单、解除记恢复单（detector_trades，含 yfinance
  TSLA 价格快照），周报判分用

用法：
    .venv/bin/python -m intel.detector --once     # 单跑一轮
    .venv/bin/python -m intel.detector_report     # 值班报告
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from intel import store
from intel.collectors.base import Collector, cli

ET = ZoneInfo("America/New_York")

# ---- 冻结规则参数（N3-H，不许改） -----------------------------------------
SHORT_JUMP_PCT = 10.0    # up-jump：空头利益双周 change_pct >= +10%
LOOKBACK_BDAYS = 20      # 密集日回看窗（交易日）
PERSIST_BDAYS = 20       # F20：触发日起 20 个交易日 risk-off，重叠触发顺延

# ---- 标定常量（分位映射实现，非规则参数） ----------------------------------
DENSE_REF_COUNT = 65     # 历史 Sprinklr 口径密集参考值（帖/日）
DENSE_QUANTILE = 0.8789  # P(日发帖数<=65)，musk_tweets.csv 2018-01-03→2025-05-08
                         # 全档 2683 自然日（0 补齐）经验 CDF，2026-07-24 算出写死
CALIB_BDAYS = 20         # 标定期：前 20 个交易日只累积基线，不出信号

MUSK_TYPES = ("x_musk_post", "x_musk_rt")
COST_LINE = -0.0006      # 判分成本线：往返 ~6bp（与 N3-H 日记同口径）

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detector_state (
    state_date          TEXT PRIMARY KEY,   -- ET 交易日 YYYY-MM-DD
    state               TEXT NOT NULL CHECK (state IN ('CALIBRATING','RISK_ON','RISK_OFF')),
    musk_count          REAL,               -- 信号窗内最大日发帖数（nitter 口径）
    musk_count_day      TEXT,               -- 该计数对应的自然日
    musk_window_json    TEXT,               -- 本行覆盖的自然日计数 {date: count}，基线即由此累积
    dense_thr           REAL,               -- 当日生效密集阈值（标定期 NULL）
    baseline_days       INTEGER NOT NULL,   -- 已累积基线交易日数（含当日）
    short_settlement    TEXT,               -- 最新已发布空头结算日
    short_chg_pct       REAL,               -- 最新已发布空头 change_pct
    short_upjump_recent INTEGER NOT NULL DEFAULT 0,  -- 回看20交易日内有无 up-jump 发布
    triggered           INTEGER NOT NULL DEFAULT 0,  -- 本日是否命中触发（含顺延触发）
    risk_off_until      TEXT,               -- RISK_OFF 到期交易日（含）
    updated_utc         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detector_trades (
    trade_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_time_utc TEXT NOT NULL,
    state_date     TEXT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('REDUCE','RESTORE')),
    price          REAL,               -- yfinance TSLA 快照；获取失败 NULL 如实记
    price_time_utc TEXT,
    note           TEXT
);
"""


# ------------------------------------------------------------------ helpers


def prev_bday(d: date) -> date:
    return np.busday_offset(d, -1, roll="forward").astype("datetime64[D]").astype(date)


def next_bday_after(d: date) -> date:
    """严格晚于 d 的第一个交易日（d 可为周末）——act 日近似."""
    return np.busday_offset(d, 1, roll="backward").astype("datetime64[D]").astype(date)


def _et_date(iso_utc: str) -> date:
    return datetime.fromisoformat(iso_utc).astimezone(ET).date()


def musk_day_counts(conn, lo: date, hi: date) -> dict[str, int]:
    """[lo, hi] 闭区间自然日的 Musk 日发帖数（nitter 口径 post+RT，0 补齐）."""
    counts = {str(lo + timedelta(days=k)): 0 for k in range((hi - lo).days + 1)}
    q = ",".join("?" * len(MUSK_TYPES))
    for (t,) in conn.execute(
        f"SELECT event_time_utc FROM events WHERE source_id='x_nitter' AND type IN ({q})",
        MUSK_TYPES,
    ):
        d = str(_et_date(t))
        if d in counts:
            counts[d] += 1
    return counts


def short_releases(conn, now_iso: str) -> list[dict]:
    """已发布（event_time <= now）的空头利益发布，按发布时刻升序."""
    out = []
    for r in conn.execute(
        """SELECT event_time_utc, payload_json FROM events
           WHERE source_id='finra_short' AND event_time_utc <= ?
           ORDER BY event_time_utc""",
        (now_iso,),
    ):
        p = json.loads(r["payload_json"] or "{}")
        chg = p.get("change_pct")
        if chg is None and p.get("prev_short_interest"):
            chg = (p["short_interest"] / p["prev_short_interest"] - 1) * 100
        if chg is None:
            continue
        out.append({
            "pub_time_utc": r["event_time_utc"],
            "settlement": p.get("settlement_date"),
            "chg_pct": float(chg),
            "short_interest": p.get("short_interest"),
            "days_to_cover": p.get("days_to_cover"),
        })
    return out


def recent_upjumps(shorts: list[dict], today: date) -> list[dict]:
    """act（发布次交易日）落在回看 20 交易日内的 up-jump 发布."""
    out = []
    for s in shorts:
        if s["chg_pct"] < SHORT_JUMP_PCT:
            continue
        act = next_bday_after(_et_date(s["pub_time_utc"]))
        if act <= today and int(np.busday_count(act, today)) <= LOOKBACK_BDAYS:
            out.append(s)
    return out


def baseline_from_rows(conn, before_date: str | None = None) -> dict[str, int]:
    """把历史 detector_state 行累积的日计数合并成基线分布（date -> count）."""
    sql = "SELECT musk_window_json FROM detector_state"
    args: tuple = ()
    if before_date:
        sql += " WHERE state_date < ?"
        args = (before_date,)
    base: dict[str, int] = {}
    for (j,) in conn.execute(sql, args):
        if j:
            base.update(json.loads(j))
    return base


def tsla_snapshot() -> tuple[float | None, str | None, str | None]:
    """(price, time_utc, error) —— yfinance 当前价快照，失败如实返回 None."""
    try:
        import yfinance as yf

        tk = yf.Ticker("TSLA")
        price = None
        try:
            price = float(tk.fast_info["last_price"])
        except Exception:  # noqa: BLE001
            h = tk.history(period="1d")
            if len(h):
                price = float(h["Close"].iloc[-1])
        if price and price > 0:
            return price, store.utcnow_iso(), None
        return None, None, "no price data"
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ 状态机


def evaluate(conn, now: datetime | None = None) -> dict:
    """跑一轮状态机：写 detector_state 当日行，切换时发事件/记假想单.

    返回 {state, switched, n_new_events, note}。非交易日不评估。
    """
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="seconds")
    today = now.astimezone(ET).date()
    if not bool(np.is_busday(today)):
        return {"state": None, "switched": False, "n_new_events": 0,
                "note": f"{today} 非交易日（busday 近似），跳过"}

    conn.executescript(_SCHEMA)

    # -- 前情：今日已有行则续写（盘中多次轮询），否则接最近一行
    row_today = conn.execute(
        "SELECT * FROM detector_state WHERE state_date=?", (str(today),)
    ).fetchone()
    row_prev = conn.execute(
        "SELECT * FROM detector_state WHERE state_date<? ORDER BY state_date DESC LIMIT 1",
        (str(today),),
    ).fetchone()
    ref = row_today or row_prev
    prev_state = ref["state"] if ref else None
    prev_until = ref["risk_off_until"] if ref else None
    n_rows_before = conn.execute(
        "SELECT COUNT(*) FROM detector_state WHERE state_date<?", (str(today),)
    ).fetchone()[0]
    baseline_days = n_rows_before + 1          # 含今日
    active = n_rows_before >= CALIB_BDAYS      # 第 21 个交易日起出信号

    # -- 放风腿：信号窗 = 上一交易日（含）至昨日（含）的自然日，act 均为今日
    win_lo, win_hi = prev_bday(today), today - timedelta(days=1)
    window = musk_day_counts(conn, win_lo, win_hi) if win_hi >= win_lo else {}
    musk_day, musk_cnt = (max(window.items(), key=lambda kv: kv[1])
                          if window else (None, None))

    # -- 空头腿
    shorts = short_releases(conn, now_iso)
    latest = shorts[-1] if shorts else None
    upjumps = recent_upjumps(shorts, today)

    # -- 状态决策
    thr = None
    triggered = False
    until = prev_until
    if not active:
        state = "CALIBRATING"
        until = None
    else:
        base = baseline_from_rows(conn, before_date=str(today))  # 不含今日窗（因果）
        vals = np.array(list(base.values()), float)
        thr = float(np.quantile(vals, DENSE_QUANTILE)) if len(vals) else None
        if thr is not None:
            for d, c in window.items():
                if c <= thr:
                    continue
                dense_end = datetime.combine(
                    date.fromisoformat(d), datetime.max.time().replace(microsecond=0)
                ).replace(tzinfo=ET)
                if any(datetime.fromisoformat(u["pub_time_utc"]) < dense_end
                       for u in upjumps):
                    triggered = True
                    musk_day, musk_cnt = d, c   # 展示触发那一天的读数
                    break
        if triggered:
            new_until = str(np.busday_offset(today, PERSIST_BDAYS - 1)
                            .astype("datetime64[D]").astype(date))
            until = max(until or "", new_until)  # 重叠触发顺延
            state = "RISK_OFF"
        elif until and until >= str(today):
            state = "RISK_OFF"                   # F20 延续
        else:
            state, until = "RISK_ON", None

    # -- 写当日状态行（同日重跑覆盖更新）
    conn.execute(
        """INSERT OR REPLACE INTO detector_state
           (state_date, state, musk_count, musk_count_day, musk_window_json,
            dense_thr, baseline_days, short_settlement, short_chg_pct,
            short_upjump_recent, triggered, risk_off_until, updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (str(today), state, musk_cnt, musk_day, json.dumps(window),
         thr, baseline_days, latest["settlement"] if latest else None,
         latest["chg_pct"] if latest else None, int(bool(upjumps)),
         int(triggered), until, now_iso),
    )
    conn.commit()

    # -- 状态切换：发 detector 事件 + 假想单
    switched = prev_state != state
    n_new = 0
    if switched:
        musk_s = (f"Musk {musk_day} 发帖 {musk_cnt}（nitter 口径）" if musk_cnt is not None
                  else "Musk 无窗口数据")
        if thr is not None:
            musk_s += f"，阈值 {thr:.1f}"
        elif state == "CALIBRATING":
            musk_s += f"，基线标定 {baseline_days}/{CALIB_BDAYS} 交易日"
        short_s = (f"空头 {latest['chg_pct']:+.1f}%（结算 {latest['settlement']}）"
                   if latest else "空头无已发布数据")
        if state == "RISK_OFF":
            title = (f"探测器 RISK_OFF 触发：{musk_s} × {short_s}，"
                     f"回看{LOOKBACK_BDAYS}日内 up-jump {len(upjumps)} 份 → F20 至 {until}")
        elif prev_state == "RISK_OFF":
            title = f"探测器 RISK_OFF 解除 → RISK_ON：F20 到期；{musk_s}；{short_s}"
        elif state == "RISK_ON" and prev_state == "CALIBRATING":
            title = f"探测器标定完成 → RISK_ON：{musk_s}；{short_s}"
        else:
            title = f"探测器上线 CALIBRATING：{musk_s}；{short_s}"
        n_new = store.insert_events(conn, "detector", [{
            "dedupe_key": f"{today}_{prev_state}->{state}",
            "event_time_utc": now_iso,
            "symbol": "TSLA",
            "type": "detector_state",
            "title": title,
            "payload": {
                "state": state, "prev_state": prev_state,
                "musk_count": musk_cnt, "musk_count_day": musk_day,
                "dense_thr": thr, "baseline_days": baseline_days,
                "short_chg_pct": latest["chg_pct"] if latest else None,
                "short_settlement": latest["settlement"] if latest else None,
                "n_upjumps_recent": len(upjumps), "risk_off_until": until,
                "rule": f"N3-H frozen: upjump>=+{SHORT_JUMP_PCT:.0f}% x dense, F{PERSIST_BDAYS}",
            },
        }])
        action = ("REDUCE" if state == "RISK_OFF"
                  else "RESTORE" if prev_state == "RISK_OFF" else None)
        if action:
            price, ptime, err = tsla_snapshot()
            note = ("假想减仓：" if action == "REDUCE" else "假想恢复：") + title
            if err:
                note += f"（价格快照失败：{err}）"
            conn.execute(
                """INSERT INTO detector_trades
                   (trade_time_utc, state_date, action, price, price_time_utc, note)
                   VALUES (?,?,?,?,?,?)""",
                (now_iso, str(today), action, price, ptime, note),
            )
            conn.commit()

    return {"state": state, "switched": switched, "n_new_events": n_new,
            "note": f"{state} baseline={baseline_days}/{CALIB_BDAYS}"
                    + (f" thr={thr:.1f}" if thr is not None else "")
                    + (f" until={until}" if state == "RISK_OFF" else "")}


# ------------------------------------------------------------------ 组件封装


class CausalDetector(Collector):
    """哨兵值班组件：非采集渠道，每轮在全部采集器之后运行一次状态机."""

    SOURCE = {
        "source_id": "detector",
        "name": "N3 因果探测器（空头up-jump×Musk密集，F20）",
        "tier": "T1",
        "method": "derived",
        "poll_interval_s": 0,   # 不节流：每轮哨兵都评估
        "cost": "free",
        "weight_source": 0.5,
        "notes": "衍生信号非采集渠道；冻结规则 N3-H；nitter 口径标定期 20 交易日，"
                 "期满前状态 CALIBRATING",
    }

    def run_once(self, verbose: bool = True) -> dict:
        sid = self.SOURCE["source_id"]
        conn = store.connect()
        store.upsert_source(conn, self.SOURCE)
        t0 = time.time()
        try:
            res = evaluate(conn)
            dur = int((time.time() - t0) * 1000)
            store.record_poll(conn, sid, ok=True, n_seen=1,
                              n_new=res["n_new_events"], duration_ms=dur)
            stats = {"source_id": sid, "ok": True, "n_seen": 1,
                     "n_new": res["n_new_events"], "duration_ms": dur, "error": None}
            if verbose:
                print(f"[{sid}] ok  {res['note']}"
                      + ("  ->state event" if res["n_new_events"] else "")
                      + f" ({dur}ms)")
        except Exception as e:  # noqa: BLE001
            dur = int((time.time() - t0) * 1000)
            err = f"{type(e).__name__}: {e}"
            store.record_poll(conn, sid, ok=False, duration_ms=dur, error=err)
            stats = {"source_id": sid, "ok": False, "n_seen": 0, "n_new": 0,
                     "duration_ms": dur, "error": err}
            if verbose:
                traceback.print_exc()
        finally:
            conn.close()
        return stats


if __name__ == "__main__":
    cli(CausalDetector)
