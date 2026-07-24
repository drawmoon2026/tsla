"""N3 因果探测器 — 值班报告 CLI.

板块：当前状态 / 两腿最新读数 / 标定进度 / 历史状态切换 / 假想单判分。
判分口径与 N3-H 推演日记同源：减仓→恢复期间 B&H 收益 < -6bp（往返成本线）
记"对"；未平仓单用当前价（yfinance）算浮动判分，取价失败则如实标注。

用法：
    .venv/bin/python -m intel.detector_report            # 全报告
    .venv/bin/python -m intel.detector_report --no-price # 不联网取现价
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from intel import store
from intel.detector import (
    CALIB_BDAYS, COST_LINE, DENSE_QUANTILE, DENSE_REF_COUNT, LOOKBACK_BDAYS,
    PERSIST_BDAYS, SHORT_JUMP_PCT, baseline_from_rows, tsla_snapshot,
)


def _has(conn, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-price", action="store_true", help="跳过 yfinance 现价（离线判分）")
    args = ap.parse_args()

    conn = store.connect()
    L: list[str] = []
    L.append("N3 因果探测器 · 值班报告")
    L.append("=" * 72)
    L.append(f"冻结规则：空头 up-jump >= +{SHORT_JUMP_PCT:.0f}% × Musk 密集日"
             f"（回看 {LOOKBACK_BDAYS} 交易日），risk-off F{PERSIST_BDAYS} 顺延")
    L.append(f"标定：nitter 基线分位映射，q={DENSE_QUANTILE}"
             f"（= 历史 Sprinklr 口径 {DENSE_REF_COUNT} 帖/日的经验分位数）")
    L.append("")

    if not _has(conn, "detector_state"):
        L.append("detector_state 表不存在——探测器尚未运行过（先跑 intel.run_sentinel --once）")
        print("\n".join(L))
        return

    cur = conn.execute(
        "SELECT * FROM detector_state ORDER BY state_date DESC LIMIT 1").fetchone()

    # ---------------- 当前状态 ----------------
    L.append("—— 当前状态 ——")
    if cur is None:
        L.append("（无状态记录）")
    else:
        L.append(f"{cur['state_date']}  状态 = {cur['state']}"
                 + (f"（RISK_OFF 至 {cur['risk_off_until']}）" if cur["risk_off_until"] else "")
                 + (f"  本日触发" if cur["triggered"] else "")
                 + f"  （更新于 {cur['updated_utc']}）")
        L.append("")

        # ---------------- 两腿最新读数 ----------------
        L.append("—— 两腿最新读数 ——")
        mc = cur["musk_count"]
        L.append(f"放风腿：Musk {cur['musk_count_day'] or 'n/a'} 发帖 "
                 + (f"{mc:.0f}" if mc is not None else "n/a")
                 + "（nitter 口径 post+RT）"
                 + (f"，密集阈值 {cur['dense_thr']:.1f}" if cur["dense_thr"] is not None
                    else "，阈值未生效（标定期）"))
        L.append(f"空头腿：change_pct "
                 + (f"{cur['short_chg_pct']:+.1f}%" if cur["short_chg_pct"] is not None else "n/a")
                 + f"（结算 {cur['short_settlement'] or 'n/a'}）；"
                 f"回看 {LOOKBACK_BDAYS} 交易日内 up-jump："
                 + ("有" if cur["short_upjump_recent"] else "无"))
        L.append("")

        # ---------------- 标定进度 ----------------
        L.append("—— 标定进度 ——")
        base = baseline_from_rows(conn)  # 含今日窗，纯展示
        vals = np.array(list(base.values()), float)
        n_bd = cur["baseline_days"]
        if cur["state"] == "CALIBRATING":
            L.append(f"标定中：{n_bd}/{CALIB_BDAYS} 交易日；已累积自然日样本 {len(vals)} 天")
        else:
            L.append(f"标定完成（基线扩张窗持续累积）：交易日 {n_bd}，自然日样本 {len(vals)} 天")
        if len(vals):
            L.append(f"nitter 日计数分布：均值 {vals.mean():.1f}  中位数 {np.median(vals):.0f}"
                     f"  最大 {vals.max():.0f}；q{DENSE_QUANTILE} 阈值"
                     + ("预览" if cur["state"] == "CALIBRATING" else "")
                     + f" = {np.quantile(vals, DENSE_QUANTILE):.1f}")
        L.append("")

    # ---------------- 历史状态切换 ----------------
    L.append("—— 历史状态切换（events: source=detector）——")
    rows = conn.execute(
        """SELECT event_time_utc, title FROM events
           WHERE source_id='detector' ORDER BY event_time_utc""").fetchall()
    if rows:
        for r in rows:
            L.append(f"  {r['event_time_utc']}  {r['title']}")
    else:
        L.append("  （无）")
    L.append("")

    # ---------------- 假想单判分 ----------------
    L.append("—— 假想单判分（REDUCE→RESTORE 配对；成本线 -6bp）——")
    trades = (conn.execute(
        "SELECT * FROM detector_trades ORDER BY trade_time_utc").fetchall()
        if _has(conn, "detector_trades") else [])
    if not trades:
        L.append("  （无假想单——尚未出现 RISK_OFF 触发）")
    else:
        cur_price = None
        if not args.no_price:
            cur_price, _, err = tsla_snapshot()
            if err:
                L.append(f"  [现价获取失败：{err}——未平仓单不判分]")
        open_reduce = None
        n_right = n_wrong = 0
        for t in trades:
            px = f"{t['price']:.2f}" if t["price"] is not None else "n/a"
            L.append(f"  {t['trade_time_utc']}  {t['action']:<7} @ {px}  ({t['state_date']})")
            if t["action"] == "REDUCE":
                open_reduce = t
            elif t["action"] == "RESTORE" and open_reduce is not None:
                if open_reduce["price"] is not None and t["price"] is not None:
                    ret = t["price"] / open_reduce["price"] - 1
                    ok = ret < COST_LINE
                    n_right, n_wrong = n_right + ok, n_wrong + (not ok)
                    L.append(f"    → 判分：risk-off 期间 B&H {ret:+.2%} —— "
                             f"{'对（避损成立）' if ok else '错（空仓错过收益，含成本）'}")
                else:
                    L.append("    → 判分：缺价格快照，无法判分（如实记）")
                open_reduce = None
        if open_reduce is not None:
            if cur_price and open_reduce["price"]:
                ret = cur_price / open_reduce["price"] - 1
                L.append(f"    → 未平仓：现价 {cur_price:.2f}，浮动 {ret:+.2%}"
                         f"（{'暂对' if ret < COST_LINE else '暂错'}，以 RESTORE 时为准）")
            else:
                L.append("    → 未平仓：无现价，暂不判分")
        if n_right + n_wrong:
            L.append(f"  战绩：{n_right} 对 / {n_wrong} 错（已平仓配对）")
    conn.close()
    print("\n".join(L))


if __name__ == "__main__":
    main()
