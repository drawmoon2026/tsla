"""FOMC 日历采集器：联储官网会议日期表解析（T2）.

来源 https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm（年度静态页，
覆盖前 5 年 + 未来 1-2 年的全部会议日期）。event_time_utc = 会议最后一天 14:00 ET
（决议声明的常规发布时刻）转 UTC。这是"日历预告"类事件：event_time 可以在未来，
时延视图中该渠道 lag 为负属正常（提前知道 ≠ 前视，声明内容仍未公开）。
决议声明的精确发布流水另见旧模块 intel/fomc.py（ne-press.json，CSV 管线）。

用法： .venv/bin/python -m intel.collectors.fed --once
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from intel.collectors.base import Collector, cli, http_get

URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ET = ZoneInfo("America/New_York")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

_RE_YEAR = re.compile(r"(\d{4}) FOMC Meetings")
_RE_MONTH = re.compile(r'fomc-meeting__month[^>]*>(?:<strong>)?([^<]+)')
_RE_DATE = re.compile(r'fomc-meeting__date[^>]*>([^<]+)')


def _parse_meetings(html: str) -> list[dict]:
    """按年份面板切段，段内配对 month/date div。"""
    out = []
    marks = [(m.start(), int(m.group(1))) for m in _RE_YEAR.finditer(html)]
    for i, (pos, year) in enumerate(marks):
        seg = html[pos: marks[i + 1][0] if i + 1 < len(marks) else len(html)]
        months = [m.strip() for m in _RE_MONTH.findall(seg)]
        dates = [d.strip() for d in _RE_DATE.findall(seg)]
        for month_s, date_s in zip(months, dates):
            has_sep = "*" in date_s  # 带 SEP（经济预测摘要）的会议
            nums = re.findall(r"\d+", date_s)
            if not nums:
                continue
            last_day = int(nums[-1])
            # 跨月标签（如 "October/November" + "31-1"）：会议末日属后一个月
            month_parts = [p.strip() for p in month_s.replace("/", " ").split()
                           if p.strip() in _MONTHS]
            if not month_parts:
                continue
            month = _MONTHS[month_parts[-1]]
            try:
                dt_et = datetime(year, month, last_day, 14, 0, tzinfo=ET)
            except ValueError:
                continue
            out.append({
                "year": year, "month_label": month_s, "date_label": date_s,
                "statement_et": dt_et, "has_sep": has_sep,
                "unscheduled": "unscheduled" in date_s.lower(),
            })
    return out


class FedCollector(Collector):
    # P1-7：联储官网日历页恒有会议列表（整年场次），解析出 0 场 = 疑似改版正则失配；
    # 日频轮询，连续 2 天零产出即告警（存量 FOMC 事件可掩盖哑火一年以上）
    ZERO_SEEN_ALERT_N = 2

    SOURCE = {
        "source_id": "fed_fomc",
        "name": "FOMC meeting calendar (federalreserve.gov)",
        "tier": "T2",
        "method": "scrape",
        "poll_interval_s": 86400,
        "cost": "free",
        "weight_source": 0.7,
        "notes": "日历预告类：event_time=会议末日 14:00 ET（决议常规发布时刻），可在未来",
    }

    def fetch(self):
        return http_get(URL).text

    def normalize(self, raw) -> list[dict]:
        events = []
        for m in _parse_meetings(raw):
            key = f"{m['year']}-{m['month_label']}-{m['date_label']}"
            events.append({
                "dedupe_key": key,
                "event_time_utc": m["statement_et"].astimezone(
                    ZoneInfo("UTC")).isoformat(timespec="seconds"),
                "symbol": None,
                "type": "fomc_meeting" + ("_sep" if m["has_sep"] else ""),
                "title": f"FOMC meeting {m['month_label']} {m['date_label']} {m['year']}",
                "url": URL,
                "payload": {
                    "year": m["year"], "month": m["month_label"],
                    "dates": m["date_label"], "has_sep": m["has_sep"],
                    "unscheduled": m["unscheduled"],
                },
            })
        return events


if __name__ == "__main__":
    cli(FedCollector)
