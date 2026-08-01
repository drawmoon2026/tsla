"""TSLA 期权链日度快照采集器（T1，布局痕迹层——知情资金的足迹）.

通道：yfinance（Yahoo Finance 期权链，免费，延迟报价）。历史不可得——快照从
今天起攒（前向积累，与 shadow 同节奏）。

存储：
- 新表 options_chain（与 events 同库 sentinel.sqlite）：每快照日 × 每合约一行，
  含 snapshot_date（ET 交易日期）/expiration/right/strike/volume/open_interest/iv。
  主键 (snapshot_date, contract_symbol)，同日重跑 REPLACE（保留当日最新）。
- events 表并入一条日度摘要事件（type=options_snapshot，dedupe 按 snapshot_date）：
  总 volume、P/C ratio（volume 与 OI 两口径）、对上一快照日的 OI 突变 top 行权价。

口径（诚实声明）：event_time_utc = 快照抓取时刻（我们观察到链的时刻，非行情时刻）；
Yahoo 的 volume 是当日盘中累计、OI 是前一交易日收盘后的清算所数据——OI 突变的
"事件日"天然滞后一天，做特征时以 observed_time_utc 为准即可防前视。

用法： .venv/bin/python -m intel.collectors.options_snapshot --once
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from intel import store
from intel.collectors.base import Collector, cli

TOP_N_OI_CHANGE = 5

_CHAIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS options_chain (
    snapshot_date     TEXT NOT NULL,     -- ET 日期 YYYY-MM-DD
    snapshot_time_utc TEXT NOT NULL,
    contract_symbol   TEXT NOT NULL,
    expiration        TEXT NOT NULL,
    opt_right         TEXT NOT NULL CHECK (opt_right IN ('C','P')),
    strike            REAL NOT NULL,
    last_price        REAL,
    bid               REAL,
    ask               REAL,
    volume            INTEGER,
    open_interest     INTEGER,
    iv                REAL,
    in_the_money      INTEGER,
    PRIMARY KEY (snapshot_date, contract_symbol)
);
CREATE INDEX IF NOT EXISTS idx_chain_exp ON options_chain (snapshot_date, expiration);
"""


def _f(x):
    try:
        v = float(x)
        return None if v != v else v  # NaN -> None
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return int(v) if v is not None else None


class OptionsSnapshotCollector(Collector):
    SOURCE = {
        "source_id": "options_snapshot",
        "name": "TSLA full option chain daily snapshot (yfinance)",
        "tier": "T0",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.7,
        "notes": "历史不可得，快照从 2026-07-24 起攒；OI 为前一交易日清算数据，"
                 "volume 为抓取时刻盘中累计；明细进 options_chain 表；"
                 "警告：Yahoo OI 字段有抽风期（整链 0，见 oi_quality 旗标）",
    }

    def fetch(self):
        import yfinance as yf

        from intel import prices

        # P1-4：与价格上下文共享 yfinance 退避状态（data/intel/price_cache.json）
        # ——限流期不再对同一 IP 追打整条期权链，快速失败（poll_log 红灯可见）
        remaining = prices.yf_backoff_remaining()
        if remaining > 0:
            raise RuntimeError(
                f"yfinance 共享退避中（还剩 {remaining:.0f}s）——跳过期权链抓取")
        t = yf.Ticker("TSLA")
        now_utc = datetime.now(ZoneInfo("UTC"))
        snap_date = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        rows = []
        try:
            for exp in t.options:
                ch = t.option_chain(exp)
                for right, df in [("C", ch.calls), ("P", ch.puts)]:
                    for r in df.itertuples(index=False):
                        rows.append(
                            {
                                "snapshot_date": snap_date,
                                "snapshot_time_utc": now_utc.isoformat(timespec="seconds"),
                                "contract_symbol": r.contractSymbol,
                                "expiration": exp,
                                "opt_right": right,
                                "strike": _f(r.strike),
                                "last_price": _f(r.lastPrice),
                                "bid": _f(r.bid),
                                "ask": _f(r.ask),
                                "volume": _i(r.volume),
                                "open_interest": _i(r.openInterest),
                                "iv": _f(r.impliedVolatility),
                                "in_the_money": int(bool(r.inTheMoney)),
                            }
                        )
        except Exception as e:  # noqa: BLE001 —— 失败推进共享退避计数，再如实上抛
            prices.yf_failure(f"options_chain: {type(e).__name__}: {e}")
            raise
        if not rows:
            prices.yf_failure("options_chain: 空链（疑似限流）")
            raise RuntimeError("期权链为空（疑似限流），已计入共享退避")
        return {"snapshot_date": snap_date, "time_utc": now_utc.isoformat(timespec="seconds"),
                "rows": rows}

    def normalize(self, raw) -> list[dict]:
        rows, snap_date = raw["rows"], raw["snapshot_date"]
        if not rows:
            return []
        conn = store.connect()
        try:
            conn.executescript(_CHAIN_SCHEMA)
            conn.executemany(
                """INSERT OR REPLACE INTO options_chain
                   (snapshot_date, snapshot_time_utc, contract_symbol, expiration,
                    opt_right, strike, last_price, bid, ask, volume, open_interest,
                    iv, in_the_money)
                   VALUES (:snapshot_date, :snapshot_time_utc, :contract_symbol,
                           :expiration, :opt_right, :strike, :last_price, :bid, :ask,
                           :volume, :open_interest, :iv, :in_the_money)""",
                rows,
            )
            conn.commit()
            summary = self._summarize(conn, snap_date, rows)
        finally:
            conn.close()
        return [
            {
                "dedupe_key": f"opt_{snap_date}",
                "event_time_utc": raw["time_utc"],
                "symbol": "TSLA",
                "type": "options_snapshot",
                "title": (f"TSLA options {snap_date}: vol {summary['total_volume']:,}"
                          f" P/C(vol) {summary['pc_ratio_volume']}"),
                "url": "https://finance.yahoo.com/quote/TSLA/options",
                "payload": summary,
            }
        ]

    @staticmethod
    def _summarize(conn, snap_date: str, rows: list[dict]) -> dict:
        cvol = sum(r["volume"] or 0 for r in rows if r["opt_right"] == "C")
        pvol = sum(r["volume"] or 0 for r in rows if r["opt_right"] == "P")
        coi = sum(r["open_interest"] or 0 for r in rows if r["opt_right"] == "C")
        poi = sum(r["open_interest"] or 0 for r in rows if r["opt_right"] == "P")
        # OI 突变：对上一快照日同合约的 OI 差，按 |diff| 取 top 行权价
        prev = conn.execute(
            "SELECT MAX(snapshot_date) FROM options_chain WHERE snapshot_date < ?",
            (snap_date,),
        ).fetchone()[0]
        oi_jumps = []
        if prev:
            cur = conn.execute(
                """SELECT a.expiration, a.opt_right, a.strike,
                          a.open_interest - COALESCE(b.open_interest, 0) AS d_oi,
                          a.open_interest AS oi_now
                   FROM options_chain a
                   LEFT JOIN options_chain b
                     ON b.snapshot_date = ? AND b.contract_symbol = a.contract_symbol
                   WHERE a.snapshot_date = ? AND a.open_interest IS NOT NULL
                   ORDER BY ABS(a.open_interest - COALESCE(b.open_interest, 0)) DESC
                   LIMIT ?""",
                (prev, snap_date, TOP_N_OI_CHANGE),
            )
            oi_jumps = [
                {"expiration": r["expiration"], "right": r["opt_right"],
                 "strike": r["strike"], "d_oi": r["d_oi"], "oi": r["oi_now"]}
                for r in cur
            ]
        return {
            "snapshot_date": snap_date,
            "n_contracts": len(rows),
            "n_expirations": len({r["expiration"] for r in rows}),
            "total_volume": cvol + pvol,
            "call_volume": cvol,
            "put_volume": pvol,
            "pc_ratio_volume": round(pvol / cvol, 3) if cvol else None,
            "total_oi": coi + poi,
            "call_oi": coi,
            "put_oi": poi,
            "pc_ratio_oi": round(poi / coi, 3) if coi else None,
            # 数据质量旗标：yfinance 的 OI 有历史性抽风期（整链返回 0，2026-07-24 实测）；
            # OI 总量 < 日 volume 的 10% 视为可疑，OI 类特征该日不可用
            "oi_quality": "suspect_zero" if (coi + poi) < 0.1 * (cvol + pvol) else "ok",
            "prev_snapshot_date": prev,
            "top_oi_changes": oi_jumps,
        }


if __name__ == "__main__":
    cli(OptionsSnapshotCollector)
