"""价格上下文 — 仪表盘生成时的日线增量更新 + 现价快照 + S2 开关读数.

职责（roadmap #1/#2 的数据层）：
- 每次仪表盘生成时拉 yfinance TSLA 日线（增量补到今天）与现价快照；
- 成功则写 JSON 缓存（data/intel/price_daily_cache.json），失败降级读缓存，
  再失败如实报错（页面显示"取价失败"与 STALE 徽章，不装新鲜）；
- 与本地 CSV 日线合并成一条连续日收盘序列（CSV 为主，yfinance 只补尾部）；
- 计算 S2 开关读数：距 252 交易日滚动高点回撤 %（E11 冻结口径：
  drawdown from 252d rolling high > 20% → 停用买入策略）。

只依赖 yfinance（探测器价格快照同款依赖），失败不炸调用方。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from intel.store import DB_PATH, utcnow_iso

ET = ZoneInfo("America/New_York")
CACHE_PATH = DB_PATH.parent / "price_daily_cache.json"

S2_WINDOW = 252        # E11 冻结口径：252 交易日滚动高点
S2_LINE_PCT = -20.0    # 回撤超过 -20% → S2 触发


def _fetch_yf() -> dict:
    """拉 yfinance 日线 + 现价；任何失败抛异常由调用方降级."""
    import yfinance as yf

    tk = yf.Ticker("TSLA")
    h = tk.history(period="2y", interval="1d", auto_adjust=False)
    daily = {
        str(idx.date()): float(c)
        for idx, c in h["Close"].items()
        if c == c  # 滤 NaN
    }
    if not daily:
        raise RuntimeError("yfinance 日线为空")
    live = None
    try:
        live = float(tk.fast_info["last_price"])
    except Exception:  # noqa: BLE001
        live = daily[max(daily)]
    if not live or live <= 0:
        live = daily[max(daily)]
    return {
        "fetched_utc": utcnow_iso(),
        "daily": daily,
        "live_price": live,
        "live_time_utc": utcnow_iso(),
    }


def _read_cache() -> dict | None:
    try:
        d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return d if d.get("daily") else None
    except Exception:  # noqa: BLE001
        return None


def _write_cache(data: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(data, separators=(",", ":")),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # 缓存写失败不影响本次渲染


def s2_reading(dates: list[date], closes: list[float]) -> dict | None:
    """S2 开关读数：距 252 交易日滚动高点回撤（含最新一根，实时可算）."""
    if len(closes) < 30:  # 数据太短没有意义
        return None
    w_dates = dates[-S2_WINDOW:]
    w_closes = closes[-S2_WINDOW:]
    hi_i = max(range(len(w_closes)), key=lambda i: w_closes[i])
    high, high_date = w_closes[hi_i], w_dates[hi_i]
    last = closes[-1]
    dd_pct = (last / high - 1.0) * 100.0
    return {
        "drawdown_pct": dd_pct,
        "high": high,
        "high_date": high_date,
        "ref_price": last,
        "ref_date": dates[-1],
        "triggered": dd_pct < S2_LINE_PCT,
        "margin_pp": dd_pct - S2_LINE_PCT,   # 距 -20% 线余量（正=尚有余量）
        "window_n": len(w_closes),
        "line_pct": S2_LINE_PCT,
    }


def get_price_context(
    csv_dates: list[date], csv_closes: list[float],
    now: datetime | None = None,
) -> dict:
    """合并 CSV 日线 + yfinance 增量 + 现价快照，附 S2 读数.

    返回 dict（键全在，值可 None）：
      dates/closes         合并后的日收盘序列（含今日临时收盘，若有现价）
      live_price/live_time_utc  现价快照；error 非 None 时二者可能取自缓存
      chg_pct/chg_date     最近一根日线的涨跌与其日期（若为今日即"今日涨跌"）
      appended_dates       相对 CSV 新补的日期数（含临时今日点）
      from_cache           本次取价失败、用的是上次缓存
      error                取价失败原因（成功为 None）
      s2                   s2_reading() 结果
      price_asof           序列最后一根的日期
    """
    now = now or datetime.now(timezone.utc)
    today_et = now.astimezone(ET).date()

    error = None
    from_cache = False
    data = None
    try:
        data = _fetch_yf()
        _write_cache(data)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        data = _read_cache()
        from_cache = data is not None

    merged: dict[date, float] = dict(zip(csv_dates, csv_closes))
    n_before = len(merged)
    csv_last = csv_dates[-1] if csv_dates else None
    live_price = live_time = None
    if data:
        for ds, c in sorted(data["daily"].items()):
            try:
                d = date.fromisoformat(ds)
            except ValueError:
                continue
            if csv_last is None or d > csv_last:
                merged[d] = c
        live_price = data.get("live_price")
        live_time = data.get("live_time_utc")

    dates = sorted(merged)
    closes = [merged[d] for d in dates]

    # 现价晚于最后一根日线（盘中）→ 以今日临时收盘并入序列（S2/走势同源）
    if (live_price and dates and today_et > dates[-1]
            and abs(live_price / closes[-1] - 1.0) > 5e-4):
        dates.append(today_et)
        closes.append(float(live_price))

    chg_pct = chg_date = None
    if len(closes) >= 2:
        chg_pct = (closes[-1] / closes[-2] - 1.0) * 100.0
        chg_date = dates[-1]

    return {
        "dates": dates,
        "closes": closes,
        "live_price": live_price,
        "live_time_utc": live_time,
        "chg_pct": chg_pct,
        "chg_date": chg_date,
        "appended_dates": len(dates) - n_before,
        "from_cache": from_cache,
        "error": error,
        "s2": s2_reading(dates, closes),
        "price_asof": dates[-1] if dates else None,
    }


if __name__ == "__main__":
    ctx = get_price_context([], [])
    s2 = ctx["s2"]
    print(f"live={ctx['live_price']} asof={ctx['price_asof']} "
          f"chg={ctx['chg_pct']} err={ctx['error']} cache={ctx['from_cache']}")
    if s2:
        print(f"S2: dd={s2['drawdown_pct']:.2f}% high={s2['high']:.2f} "
              f"({s2['high_date']}) triggered={s2['triggered']}")
