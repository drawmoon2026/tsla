"""TSLA 财报日历采集器（T2，日历预告类，yfinance 尽力而为）.

来源 yfinance Ticker.calendar（Yahoo Finance 财报日程；新版返回 dict、旧版返回
DataFrame，两者都兼容），失败降级 get_earnings_dates()。Yahoo 给的常是"预计窗口"
（一到两个日期），未官宣前会漂移——payload 带 confirmed=False 声明口径。

event_time_utc = 财报日 16:30 ET（TSLA 惯例盘后发布）转 UTC。与 fed_fomc 同属
"日历预告"类：event_time 可以在未来，时延视图中 lag 为负属正常（提前知道日期
≠ 前视，财报内容仍未公开）。dedupe 按预计日期——日期漂移时会各留一条记录，
仪表盘日历取"未来最近一场"即可。

用法： .venv/bin/python -m intel.collectors.earnings_cal --once
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from intel.collectors.base import Collector, cli

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _to_date(x) -> date | None:
    """date / datetime / pandas.Timestamp / 'YYYY-MM-DD' → date（失败 None）。"""
    try:
        if hasattr(x, "date") and not isinstance(x, date):
            return x.date()  # datetime / Timestamp
        if isinstance(x, date):
            return x
        return date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


def _extract_earnings_dates(cal) -> list[date]:
    """Ticker.calendar 的两种形态（dict / DataFrame）都尽力解析。"""
    if cal is None:
        return []
    raw: list = []
    try:
        if isinstance(cal, dict):
            v = cal.get("Earnings Date") or cal.get("earnings_date") or []
            raw = list(v) if isinstance(v, (list, tuple)) else [v]
        elif hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            raw = list(cal.loc["Earnings Date"])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for x in raw:
        d = _to_date(x)
        if d is not None:
            out.append(d)
    return sorted(set(out))


class EarningsCalCollector(Collector):
    SOURCE = {
        "source_id": "earnings_cal",
        "name": "TSLA earnings date calendar (yfinance)",
        "tier": "T2",
        "method": "api",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.7,
        "notes": "日历预告类：event_time=财报日 16:30 ET（盘后惯例），可在未来；"
                 "Yahoo 未官宣前为预计窗口、日期可能漂移（payload.confirmed=False）",
    }

    def fetch(self):
        import yfinance as yf

        t = yf.Ticker("TSLA")
        dates: list[date] = []
        errors: list[str] = []
        try:
            dates = _extract_earnings_dates(t.calendar)
        except Exception as e:  # noqa: BLE001
            errors.append(f"calendar: {type(e).__name__}: {e}")
        if not dates:  # 降级通道：历史+未来财报表，取未来行
            try:
                df = t.get_earnings_dates(limit=12)
                if df is not None and len(df):
                    today = datetime.now(ET).date()
                    dates = sorted({
                        d for d in (_to_date(idx) for idx in df.index)
                        if d is not None and d >= today
                    })
            except Exception as e:  # noqa: BLE001
                errors.append(f"get_earnings_dates: {type(e).__name__}: {e}")
        if not dates:
            raise RuntimeError(
                "yfinance 未返回财报日期（" + ("; ".join(errors) or "两通道皆空") + "）"
            )
        return dates

    def normalize(self, raw: list[date]) -> list[dict]:
        if not raw:
            return []
        # Yahoo 常给预计窗口（首末两天）——按窗口首日记一条事件，payload 带全窗口
        d0 = raw[0]
        window = [str(d) for d in raw]
        et_dt = datetime(d0.year, d0.month, d0.day, 16, 30, tzinfo=ET)
        label = str(d0) + (f" ~ {raw[-1]}" if len(raw) > 1 else "")
        return [{
            "dedupe_key": f"earnings_{d0}",
            "event_time_utc": et_dt.astimezone(UTC).isoformat(timespec="seconds"),
            "symbol": "TSLA",
            "type": "earnings_date",
            "title": f"TSLA 财报日（Yahoo 预计 {label}，盘后）",
            "url": "https://finance.yahoo.com/quote/TSLA",
            "payload": {"dates": window, "confirmed": False,
                        "note": "Yahoo 预计窗口，官宣前可能漂移；16:30 ET 为盘后惯例假设"},
        }]


if __name__ == "__main__":
    cli(EarningsCalCollector)
