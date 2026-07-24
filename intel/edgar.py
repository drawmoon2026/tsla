"""SEC EDGAR 采集：TSLA Form 4（内部人交易）与 8-K（重大事件）.

口径（防前视）：
- event_time_utc = EDGAR acceptanceDateTime（SEC 收到并接收申报的时刻，UTC）。
  这是公众理论上最早能看到该申报的时刻；交易日（transactionDate）只进 payload。
- Form 4 逐笔交易展开后按"申报(accession) × 交易代码(code)"聚合成事件行：
  同一张 Form 4 里的多笔同 code 交易合并（股数/金额求和，价格加权平均）。
- 交易代码：P=公开市场买入, S=公开市场卖出, A=授予, M=期权行权, F=税务代扣,
  G=赠与, C=转换, J/K/…=其他。事件研究只把 P 当"真金白银买入"、S 当"卖出"。

用法： .venv/bin/python -m intel.edgar   （项目根目录运行）
"""

from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

CIK = "0001318605"  # Tesla, Inc.
UA = {"User-Agent": "TSLA-research tom (drawmoon2026@gmail.com)"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "intel"
SINCE = "2018-01-01"
_MIN_INTERVAL = 0.12  # ~8 req/s，低于 SEC 10 req/s 限速
_last_req = [0.0]


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


def load_submissions() -> pd.DataFrame:
    """TSLA 全量申报列表（recent 段 + 归档段），2018-01-01 起."""
    main = json.loads(_get(f"https://data.sec.gov/submissions/CIK{CIK}.json"))
    frames = [pd.DataFrame(main["filings"]["recent"])]
    for extra in main["filings"].get("files", []):
        if extra["filingTo"] >= SINCE:
            sub = json.loads(_get(f"https://data.sec.gov/submissions/{extra['name']}"))
            frames.append(pd.DataFrame(sub))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["filingDate"] >= SINCE].reset_index(drop=True)
    return df


# ---------------------------------------------------------------- Form 4


def _txt(node, path, default=""):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else default


def parse_form4(xml_bytes: bytes) -> dict | None:
    """解析 ownershipDocument XML → {owners, transactions}；非 TSLA 返回 None."""
    root = ET.fromstring(xml_bytes)
    if _txt(root, "issuer/issuerTradingSymbol").upper() != "TSLA":
        return None
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        owners.append(
            {
                "name": _txt(ro, "reportingOwnerId/rptOwnerName"),
                "cik": _txt(ro, "reportingOwnerId/rptOwnerCik"),
                "is_officer": _txt(rel, "isOfficer", "0") in ("1", "true"),
                "is_director": _txt(rel, "isDirector", "0") in ("1", "true"),
                "is_ten_pct": _txt(rel, "isTenPercentOwner", "0") in ("1", "true"),
                "officer_title": _txt(rel, "officerTitle"),
            }
        )
    txns = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _txt(tx, "transactionCoding/transactionCode")
        shares = _txt(tx, "transactionAmounts/transactionShares/value", "0")
        price = _txt(tx, "transactionAmounts/transactionPricePerShare/value", "0")
        ad = _txt(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        date = _txt(tx, "transactionDate/value")
        try:
            shares_f, price_f = float(shares), float(price)
        except ValueError:
            continue
        txns.append(
            {"code": code, "acquired_disposed": ad, "shares": shares_f,
             "price": price_f, "value_usd": shares_f * price_f, "trade_date": date}
        )
    return {"owners": owners, "transactions": txns}


def fetch_form4(subs: pd.DataFrame) -> pd.DataFrame:
    f4 = subs[subs["form"].isin(["4", "4/A"])].reset_index(drop=True)
    print(f"Form 4 filings to fetch: {len(f4)}")
    rows, skipped = [], 0
    for i, r in f4.iterrows():
        acc = r["accessionNumber"].replace("-", "")
        doc = r["primaryDocument"].split("/")[-1]
        if not doc.endswith(".xml"):
            skipped += 1
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc}/{doc}"
        try:
            parsed = parse_form4(_get(url))
        except Exception as e:
            print(f"  fail {r['accessionNumber']}: {e}")
            skipped += 1
            continue
        if parsed is None or not parsed["transactions"]:
            skipped += 1
            continue
        owners = parsed["owners"]
        role = (
            "ten_pct_officer" if any(o["is_ten_pct"] for o in owners) and any(o["is_officer"] for o in owners)
            else "officer" if any(o["is_officer"] for o in owners)
            else "director" if any(o["is_director"] for o in owners)
            else "ten_pct" if any(o["is_ten_pct"] for o in owners)
            else "other"
        )
        # 按交易代码聚合同一张申报内的多笔交易
        tdf = pd.DataFrame(parsed["transactions"])
        for code, g in tdf.groupby("code"):
            shares, value = g["shares"].sum(), g["value_usd"].sum()
            rows.append(
                {
                    "event_time_utc": r["acceptanceDateTime"],
                    "source": "edgar_form4",
                    "type": {"P": "insider_buy", "S": "insider_sell"}.get(code, f"code_{code}"),
                    "payload": json.dumps(
                        {
                            "accession": r["accessionNumber"],
                            "is_amendment": r["form"] == "4/A",
                            "owner": owners[0]["name"] if owners else "",
                            "owner_cik": owners[0]["cik"] if owners else "",
                            "role": role,
                            "officer_title": owners[0]["officer_title"] if owners else "",
                            "code": code,
                            "shares": round(shares, 2),
                            "vwap": round(value / shares, 4) if shares else 0.0,
                            "value_usd": round(value, 2),
                            "trade_dates": sorted(g["trade_date"].unique().tolist()),
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(f4)} done")
    print(f"Form 4: {len(rows)} event rows, {skipped} filings skipped (no txn / not TSLA / fetch fail)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 8-K


def fetch_8k(subs: pd.DataFrame) -> pd.DataFrame:
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
    print(f"8-K: {len(rows)} filings")
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subs = load_submissions()
    print(f"submissions since {SINCE}: {len(subs)}")
    df8k = fetch_8k(subs)
    df8k.sort_values("event_time_utc").to_csv(DATA_DIR / "edgar_8k.csv", index=False)
    df4 = fetch_form4(subs)
    df4.sort_values("event_time_utc").to_csv(DATA_DIR / "edgar_form4.csv", index=False)
    print("written:", DATA_DIR / "edgar_form4.csv", DATA_DIR / "edgar_8k.csv")


if __name__ == "__main__":
    main()
