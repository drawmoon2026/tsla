"""价格上下文 — 仪表盘生成时的日线增量更新 + 现价快照 + S2 开关读数.

职责（roadmap #1/#2 的数据层）：
- 每次仪表盘生成时拉 yfinance TSLA 日线（增量补到今天）与现价快照；
- 进程外共享缓存 data/intel/price_cache.json（P1-4 限流治理）：
    * 5 分钟内的成功结果直接复用，不打 API——同一 5 分钟链条上的
      仪表盘 / 判分器（OHLC 同源）/ 期权快照退避判断共用这一份；
    * 取数失败记入退避状态（连续失败指数退避，上限 1 小时）：退避窗内
      不再打 API，直接用旧缓存降级（页面显示 STALE 徽章，不装新鲜）；
- 与本地 CSV 日线合并成一条连续日收盘序列（CSV 为主，yfinance 只补尾部）；
  P0-2 拆股防护：合并前核对 CSV 末日收盘与 yfinance 同日收盘，差异 >5%
  视为口径断裂（疑似拆股，yfinance 拆股日会整体重写为拆后口径）——放弃 CSV
  段，整体改用 yfinance 序列并在返回值里如实标注（splice_mismatch）；
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
CACHE_PATH = DB_PATH.parent / "price_cache.json"          # 共享缓存（P1-4）
LEGACY_CACHE_PATH = DB_PATH.parent / "price_daily_cache.json"  # 旧缓存，只读兜底

S2_WINDOW = 252        # E11 冻结口径：252 交易日滚动高点
S2_LINE_PCT = -20.0    # 回撤超过 -20% → S2 触发

CACHE_TTL_S = 300          # 5 分钟内直接复用缓存，不打 API
BACKOFF_BASE_S = 300       # 首次失败退避 5 分钟
BACKOFF_MAX_S = 3600       # 退避上限 1 小时
SPLICE_TOL_PCT = 5.0       # CSV 末日 vs yfinance 同日收盘差异容忍（P0-2）


def _fetch_yf() -> dict:
    """拉 yfinance 日线（含 OHLC）+ 现价；任何失败抛异常由调用方降级."""
    import yfinance as yf

    tk = yf.Ticker("TSLA")
    h = tk.history(period="2y", interval="1d", auto_adjust=False)
    daily: dict[str, float] = {}
    ohlc: dict[str, list[float]] = {}
    for idx, row in h.iterrows():
        o, c = float(row["Open"]), float(row["Close"])
        if c == c and c > 0:
            ds = str(idx.date())
            daily[ds] = c
            if o == o and o > 0:
                ohlc[ds] = [o, c]
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
        "ohlc": ohlc,
        "live_price": live,
        "live_time_utc": utcnow_iso(),
    }


def _read_cache() -> dict | None:
    for p in (CACHE_PATH, LEGACY_CACHE_PATH):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("daily"):
                return d
        except Exception:  # noqa: BLE001
            continue
    return None


def _write_cache(data: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(data, separators=(",", ":")),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # 缓存写失败不影响本次渲染


def _age_s(iso: str | None, now: datetime) -> float | None:
    try:
        t = datetime.fromisoformat(iso or "")
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (now - t).total_seconds()
    except ValueError:
        return None


def yf_backoff_remaining(now: datetime | None = None) -> float:
    """yfinance 失败退避剩余秒数（0 = 可以打 API）——期权快照等共享此状态."""
    now = now or datetime.now(timezone.utc)
    cache = _read_cache() or {}
    fail = cache.get("fail") or {}
    n = int(fail.get("count") or 0)
    if n <= 0:
        return 0.0
    age = _age_s(fail.get("last_utc"), now)
    if age is None:
        return 0.0
    backoff = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (n - 1))
    return max(0.0, backoff - age)


def yf_failure(err: str, now: datetime | None = None) -> None:
    """登记一次 yfinance 失败（推进共享退避计数）——期权快照采集器也调用."""
    now = now or datetime.now(timezone.utc)
    cache = _read_cache() or {}
    fail = cache.get("fail") or {}
    cache["fail"] = {"count": int(fail.get("count") or 0) + 1,
                     "last_utc": now.isoformat(timespec="seconds"),
                     "error": str(err)[:500]}
    _write_cache(cache)


def yf_ok() -> None:
    """清除共享退避状态（任一模块成功取数即调用）."""
    cache = _read_cache() or {}
    if cache.get("fail"):
        cache.pop("fail", None)
        _write_cache(cache)


def fetch_daily(now: datetime | None = None) -> tuple[dict | None, str | None, str]:
    """共享取数入口：返回 (data, error, src)，src ∈ api|ttl|stale|none.

    - ttl   ：缓存 5 分钟内新鲜，直接复用（不打 API，error=None）
    - api   ：真打了 yfinance 且成功（缓存已更新，退避清零）
    - stale ：本次没拿到新数据（失败或退避中），用旧缓存降级
    - none  ：连缓存都没有
    """
    now = now or datetime.now(timezone.utc)
    cache = _read_cache()
    age = _age_s((cache or {}).get("fetched_utc"), now)
    if cache and age is not None and age < CACHE_TTL_S:
        return cache, None, "ttl"
    remaining = yf_backoff_remaining(now)
    if remaining > 0:
        fail = (cache or {}).get("fail") or {}
        err = (f"yfinance 退避中（还剩 {remaining:.0f}s，连续失败 "
               f"{fail.get('count')} 次）：{fail.get('error', '')}")
        return (cache, err, "stale") if cache else (None, err, "none")
    try:
        data = _fetch_yf()
        fresh = dict(data)
        _write_cache(fresh)   # 成功即覆盖（不带 fail 键 = 退避清零）
        return fresh, None, "api"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        yf_failure(err, now)
        return (cache, err, "stale") if cache else (None, err, "none")


def get_ohlc(now: datetime | None = None
             ) -> tuple[dict[date, tuple[float, float]], str | None, str | None]:
    """{date: (open, close)}（共享缓存/退避同源）——判分器（N8）取价入口.

    返回 (ohlc, error, fetched_utc)：fetched_utc 为这份数据实际抓取时刻
    （ISO，缓存/降级时是旧值）——判分器用它做收盘价把关（P1-1）：
    仅当数据抓取于当日收盘定稿之后才允许判当日。
    """
    data, err, _src = fetch_daily(now)
    out: dict[date, tuple[float, float]] = {}
    for k, v in ((data or {}).get("ohlc") or {}).items():
        try:
            out[date.fromisoformat(k)] = (float(v[0]), float(v[1]))
        except (ValueError, TypeError, IndexError):
            continue
    return out, err, (data or {}).get("fetched_utc")


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
      from_cache           本次取价失败/退避，用的是上次缓存（STALE 徽章）
      cache_hit            缓存 5 分钟内新鲜直接复用（正常，不算降级）
      error                取价失败原因（成功为 None）
      splice_mismatch      P0-2：CSV 末日 vs yfinance 同日收盘断裂详情
                           （非 None 时序列已整体改用 yfinance，CSV 段弃用）
      s2                   s2_reading() 结果
      price_asof           序列最后一根的日期
    """
    now = now or datetime.now(timezone.utc)
    today_et = now.astimezone(ET).date()

    data, error, src = fetch_daily(now)
    from_cache = src == "stale"

    merged: dict[date, float] = dict(zip(csv_dates, csv_closes))
    n_before = len(merged)
    csv_last = csv_dates[-1] if csv_dates else None

    # P0-2 拆股防护：CSV 末日与 yfinance 同日收盘核对，断裂即弃用 CSV 段
    splice_mismatch = None
    if data and csv_last is not None and str(csv_last) in data["daily"]:
        yf_close = data["daily"][str(csv_last)]
        diff_pct = abs(yf_close / csv_closes[-1] - 1.0) * 100.0
        if diff_pct > SPLICE_TOL_PCT:
            splice_mismatch = {
                "date": str(csv_last),
                "csv_close": csv_closes[-1],
                "yf_close": yf_close,
                "diff_pct": round(diff_pct, 3),
                "tol_pct": SPLICE_TOL_PCT,
                "action": "疑似拆股/口径断裂——CSV 段弃用，整体改用 yfinance 序列",
            }
            merged = {}
            csv_last = None

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
        "cache_hit": src == "ttl",
        "error": error,
        "splice_mismatch": splice_mismatch,
        "s2": s2_reading(dates, closes),
        "price_asof": dates[-1] if dates else None,
    }


if __name__ == "__main__":
    ctx = get_price_context([], [])
    s2 = ctx["s2"]
    print(f"live={ctx['live_price']} asof={ctx['price_asof']} "
          f"chg={ctx['chg_pct']} err={ctx['error']} cache={ctx['from_cache']} "
          f"ttl_hit={ctx['cache_hit']}")
    if s2:
        print(f"S2: dd={s2['drawdown_pct']:.2f}% high={s2['high']:.2f} "
              f"({s2['high_date']}) triggered={s2['triggered']}")
