"""棋谱预案 — 预案先于事件的二叉决策树页面生成器.

思想：树不预测走哪条分支，只保证每条分支都有应手（E10 已证明择时预测
跑输持有）。页面把系统真实阈值换算成具体价位，涨/跌两路各推 3-4 层，
每个分支点标注两类动作：

  「系统规则」  冻结实验产物，附实验编号（E11 / N3-H / E8-A / N6 / E10）
  「个人参数」  data/intel/position.json 里的用户仓位参数——金额/股数按
                account_value_usd（默认 100000，示例值·请改真实）换算成具体
                数字；数据背书的默认值标实验出处（应急借款 ≤ 权益 50% = E13
                实测），其余默认值标"未验证·待用户批准"（黄色）；未设置的
                参数保留提示但降为次要样式（附 E5/E10/E13 证据旁注）
  「历史频率」  8 年日线 first-passage 统计（滚动起点采样）：每层分支标注
                60 交易日内"先到下档 / 先到上档 / 都没到"的历史发生频率
                ——历史频率 ≠ 概率预测（页脚同文声明）
  「E13 参考」  代表组合 8 年 4 次干预点（日期/深度/金额）作为参考卡贴在
                对应深度的下跌节点旁；树顶另有"懒人基准"（30% DCA + 70%
                现金的 E13 实测合成 XIRR）作机会成本锚

数据源（全部只读，任一缺失降级显示、不炸）：
  - intel.prices.get_price_context   现价 / 252 日滚动高 / 回撤（E11 冻结口径）
  - data/intel/sentinel.sqlite       detector_state 最新行（N3 前向状态机）
  - outputs/shadow_status.json       e8a 影子引擎的 S2 读数（交叉对照）
  - models/e8a/meta.json             E8-A 门槛与交易几何（冻结参数，展示用）
  - data/intel/position.json         用户仓位参数（缺失时生成默认文件）

口径声明（页面页脚同文）：
  - S2 开关按 E11 冻结口径：距 252 交易日滚动高点回撤 < -20% 停用 E8-A 入场；
    实盘引擎以昨日日收盘评估（shift(1)），盘中破线不算、收盘确认次日生效。
  - "-10% / -20%" 档位按生成时刻现价换算，现价变动后档位价随之变。

用法：
    .venv/bin/python -m intel.playbook                # 生成 data/intel/playbook.html
    .venv/bin/python -m intel.playbook --out x.html   # 自定义输出路径

调度（暂不接线，说明留档）：
  本模块设计为由 com.tsla.dashboard 的 launchd 周期任务在 intel.dashboard
  之后串带生成——即把生成脚本里的命令扩为
      .venv/bin/python -m intel.dashboard && .venv/bin/python -m intel.playbook
  plist 本身归仪表盘域管理，本次不改（另一代理施工中）；主线稍后接线。
  在那之前手动跑上面的命令即可，页面自带数据龄徽章，不装新鲜。
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from intel.store import DB_PATH

try:  # 现价 / 252 日高 / 回撤——与仪表盘同源（E11 冻结口径）
    from intel.prices import S2_LINE_PCT, get_price_context
except Exception:  # noqa: BLE001
    get_price_context = None  # type: ignore[assignment]
    S2_LINE_PCT = -20.0

try:  # 探测器冻结参数同源引用；失败退回写死值（容错，不炸页面）
    from intel.detector import CALIB_BDAYS, LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT
except Exception:  # noqa: BLE001
    CALIB_BDAYS, LOOKBACK_BDAYS, PERSIST_BDAYS, SHORT_JUMP_PCT = 20, 20, 20, 10.0

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSITION_PATH = PROJECT_ROOT / "data" / "intel" / "position.json"
OUT_PATH = PROJECT_ROOT / "data" / "intel" / "playbook.html"
SHADOW_STATUS = PROJECT_ROOT / "outputs" / "shadow_status.json"
E8A_META = PROJECT_ROOT / "models" / "e8a" / "meta.json"
BARS_CSV = PROJECT_ROOT / "data" / "TSLA_1h_alpaca.csv"

S2_RATIO = 1.0 + S2_LINE_PCT / 100.0   # 0.8：S2 线 = 252 日高 × 0.8

UNVERIFIED = "未验证·待用户批准"

# 默认仓位参数：个人线全部以"距 252 日滚动高点的百分比"表达（与 S2 同口径，
# 负数=回撤，如 -20 即 252 日高 × 0.8）。金额基数 = account_value_usd（示例值，
# 用户须改为真实）。数据背书的默认值标注实验出处；未经用户批准的一律黄标。
# 改完本文件重跑 `python -m intel.playbook` 即生效。
DEFAULT_POSITION: dict = {
    "_说明": {
        "account_value_usd": "账户总权益（美元）——页面所有金额/股数换算的基数；"
                             "默认 100000 为示例值·请改为真实",
        "monthly_inflow_usd": "每月新入金（美元）；默认 4167 = E13 口径（≈¥30k/7.2，"
                              "每月首个交易日）",
        "position_pct": "当前仓位占可投资金的百分比（0-100）",
        "cost_basis": "持仓成本价（美元/股）；null=未填，页面不显示盈亏",
        "add_budget_pct": "S2 解除后可动用的加仓预算，占权益百分比；"
                          "只在 S2 解除分支生效，逐笔跟 E8-A 信号分批",
        "emergency_borrow_cap_pct": "应急过桥借款上限，占借款时点权益的百分比。"
            "E13 实测背书：借款按权益比例、上限 ≤50%——固定金额授信=小账户隐形"
            " 3-5 倍杠杆（39 次危险事件 100% 来自固定 3 个月工资档，峰值负债/权益"
            " 4.9）；比例制 ≤50% 全程零触 30% 维持线（缓冲 ≥0.65）",
        "trim_line_pct": "老仓位减仓线：距 252 日高的百分比（如 -10 → 高×0.90）；"
            "默认 null=不设。E10/E13 证据：择时减仓历史上跑输持有；"
            "若坚持要设，E5 显示从高点回撤 10-15% 是常见选择但未验证",
        "max_pain_pct": "最大容忍线：距 252 日高的百分比（如 -50 → 高×0.50）；"
            "默认 null=不设。E10/E13 证据同上（E5 的 -30% 强平对照 54 例全部恶化"
            "收益）；设线属于个人风险偏好而非统计优势",
        "_status": "各参数的验证状态；用户确认某参数后把状态改为'已批准'",
    },
    "account_value_usd": 100000,
    "monthly_inflow_usd": 4167,
    "position_pct": 30,
    "cost_basis": None,
    "add_budget_pct": 20,
    "emergency_borrow_cap_pct": 50,
    "trim_line_pct": None,
    "max_pain_pct": None,
    "_status": {
        "account_value_usd": "示例值·请改为真实",
        "monthly_inflow_usd": "示例值·E13 口径",
        "position_pct": "已确认",
        "cost_basis": UNVERIFIED,
        "add_budget_pct": UNVERIFIED,
        "emergency_borrow_cap_pct": "E13 实测背书·待用户批准",
        "trim_line_pct": UNVERIFIED,
        "max_pain_pct": UNVERIFIED,
    },
}


def esc(s: object) -> str:
    return html_mod.escape(str(s), quote=True)


def add_bdays(d: date, n: int) -> date:
    """d 之后第 n 个工作日（周一至周五近似，不剔假日——与探测器同口径）。"""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


# ---------------------------------------------------------------- data pulls


def load_position() -> dict:
    """读 position.json；不存在则写默认文件。缺键补默认，坏文件如实报错。"""
    if not POSITION_PATH.exists():
        POSITION_PATH.parent.mkdir(parents=True, exist_ok=True)
        POSITION_PATH.write_text(
            json.dumps(DEFAULT_POSITION, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    pos = json.loads(POSITION_PATH.read_text(encoding="utf-8"))
    for k, v in DEFAULT_POSITION.items():
        pos.setdefault(k, v)
    for nested in ("_status", "_说明"):
        pos.setdefault(nested, {})
        for k, v in DEFAULT_POSITION[nested].items():
            pos[nested].setdefault(k, v)
    return pos


def load_daily_closes() -> tuple[list[date], list[float]]:
    """ET 交易日日线收盘（与仪表盘同源：src.common.data_io）；失败返回空。"""
    try:
        import sys
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.common.data_io import load_bars
        df = load_bars(str(BARS_CSV))
        et = df.index.tz_convert("America/New_York")
        s = df["Close"].groupby(et.date).last()
        return list(s.index), [float(v) for v in s.values]
    except Exception:  # noqa: BLE001
        return [], []


def load_detector_row() -> dict | None:
    """detector_state 最新一行（只读）；表缺失/无行 → None。"""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            r = conn.execute(
                "SELECT * FROM detector_state ORDER BY state_date DESC LIMIT 1"
            ).fetchone()
            return dict(r) if r else None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None


def load_shadow_s2() -> dict | None:
    """shadow_status.json 里 e8a 的 S2 读数（引擎口径交叉对照）。"""
    try:
        d = json.loads(SHADOW_STATUS.read_text(encoding="utf-8"))
        e8a = d.get("strategies", {}).get("e8a", {})
        s2 = e8a.get("s2")
        if s2:
            s2 = dict(s2)
            s2["updated_at"] = d.get("updated_at")
            s2["halted"] = e8a.get("halted")
        return s2
    except Exception:  # noqa: BLE001
        return None


def load_e8a_meta() -> dict:
    """E8-A 冻结门槛与几何（展示用）；读不到给冻结时点写死值。"""
    try:
        m = json.loads(E8A_META.read_text(encoding="utf-8"))
        return {"gate": m["gate"]["threshold"],
                "tp": m["geometry"]["tp_pct"] * 100,
                "sl": m["geometry"]["sl_pct"] * 100,
                "timeout": m["geometry"]["timeout_bars"]}
    except Exception:  # noqa: BLE001
        return {"gate": 0.4325, "tp": 0.5, "sl": 2.0, "timeout": 48}


# ------------------------------------------------------ 历史频率（非预测）

FP_HORIZON = 60      # first-passage 视界：60 个交易日
FP_BAND_PP = 3.0     # 起点采样带宽：回撤中心 ±3 个百分点

E13_GRID = PROJECT_ROOT / "outputs" / "e13_salary_ladder" / "grid_results.csv"

# E13 干预参考案例（代表组合 s15_tp20_6mo_L50，权益 50% 比例额度）：
# 8 年窗口 W2 仅 4 次出手（首日建仓补足那笔是机械结果、不列），3 年 W1 对照并列。
# 深度口径 = 距梯子锚点跌幅（非距 252 日高），金额为当时借款额（年化 6.5% 过桥）。
E13_CASES = {
    "shallow": [("2018-08-17", "-16.5%", "$4,790", "@21.14", "借后缓冲 0.67"),
                ("2023-08-11", "-15.1%", "$3,691", "@239.55", "3 年窗口对照")],
    "deep":    [("2018-09-07", "-31.4%", "$4,947", "@17.37", "借后缓冲 0.67"),
                ("2023-10-30", "-27.8%", "$7,020", "@203.62", "3 年窗口对照")],
    "abyss":   [("2019-05-31", "-51.1%", "$16,331", "@12.38", "单次最大·额度打满"),
                ("2024-04-19", "-47.8%", "$17,495", "@147.11", "当时现金池仅 $6.9k")],
}


def rolling_dd_series(closes: list[float]) -> list[float]:
    """距 252 日滚动高的回撤 %（含当日、窗口不足时用可得窗口——同 s2_reading）。"""
    out, n = [], len(closes)
    for i in range(n):
        hi = max(closes[max(0, i - 251):i + 1])
        out.append((closes[i] / hi - 1) * 100)
    return out


def first_passage(closes: list[float], dd: list[float], center_dd: float,
                  up_move: float | None, down_move: float | None) -> dict | None:
    """三分 first-passage 频率：从 dd≈center 的滚动起点出发，FP_HORIZON 个
    交易日内先触上行目标 / 先触下行目标 / 两者都没到 的历史频率。

    口径：起点 = 所有 dd 在 center±FP_BAND_PP 内的交易日（高位上下文
    center≥-band 时取 dd≥-band），滚动采样、样本高度重叠；目标 = 起点收盘
    ×(1+move)，逐日收盘首次穿越判定（同日先查下行）；起点需有完整视界。
    这是历史发生频率，不是概率预测。
    """
    n = len(closes)
    if n < FP_HORIZON + 50:
        return None
    n_up = n_dn = n_none = 0
    for i in range(n - FP_HORIZON):
        if center_dd >= -FP_BAND_PP:
            if dd[i] < -FP_BAND_PP:
                continue
        elif abs(dd[i] - center_dd) > FP_BAND_PP:
            continue
        p0 = closes[i]
        upt = p0 * (1 + up_move / 100) if up_move is not None else None
        dnt = p0 * (1 + down_move / 100) if down_move is not None else None
        hit = None
        for j in range(i + 1, i + 1 + FP_HORIZON):
            if dnt is not None and closes[j] <= dnt:
                hit = "dn"; break
            if upt is not None and closes[j] >= upt:
                hit = "up"; break
        if hit == "up":
            n_up += 1
        elif hit == "dn":
            n_dn += 1
        else:
            n_none += 1
    tot = n_up + n_dn + n_none
    if tot < 30:  # 样本太少不显示，避免装统计
        return None
    return {"n": tot, "up": n_up / tot * 100, "dn": n_dn / tot * 100,
            "none": n_none / tot * 100}


def _xirr(flows: list[tuple[date, float]]) -> float:
    """现金流 XIRR（二分求解，E13 同口径）。"""
    d0 = flows[0][0]
    def npv(r: float) -> float:
        return sum(a / (1 + r) ** ((d - d0).days / 365.25) for d, a in flows)
    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_e13_bench(dates: list[date]) -> dict:
    """懒人基准：30% 月供买入持有 + 70% 现金 4%（不再平衡）的 E13 实测合成 XIRR.

    终值取 outputs/e13_salary_ladder/grid_results.csv 的对照行（DCA_hold /
    cash_4pct）按 30/70 合成，月度现金流（每月首个交易日 $4,167）与 E13 同口径
    求 XIRR。读不到 CSV 或日线时退回 E13 冻结时点算出的常数（口径相同）。
    """
    frozen = {"w2": 21.9, "w1": 9.0, "dca_w2": 41.3, "dca_w1": 19.7,
              "src": "冻结常数（grid_results.csv 不可读）"}
    try:
        import csv as _csv
        fv: dict[tuple[str, str], float] = {}
        with E13_GRID.open(encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r["strategy"] in ("DCA_hold", "cash_4pct"):
                    fv[(r["window"], r["strategy"])] = float(r["final_equity"])
        out = {"src": "grid_results.csv 对照行 30/70 合成"}
        for key, start in (("w2", date(2018, 7, 23)), ("w1", date(2023, 7, 1))):
            dep, seen = [], set()
            for d in dates:
                if d >= start and (d.year, d.month) not in seen:
                    seen.add((d.year, d.month)); dep.append(d)
            blend = (0.3 * fv[(key.upper(), "DCA_hold")]
                     + 0.7 * fv[(key.upper(), "cash_4pct")])
            flows = [(d, -4167.0) for d in dep] + [(dates[-1], blend)]
            out[key] = _xirr(flows) * 100
            # DCA-hold 全仓对照同口径求出（与 summary.txt 数字互证）
            out["dca_" + key] = _xirr(
                [(d, -4167.0) for d in dep]
                + [(dates[-1], fv[(key.upper(), "DCA_hold")])]) * 100
        return out
    except Exception:  # noqa: BLE001
        return frozen


# ---------------------------------------------------------------- 树模型


def act_sys(tag: str, text: str) -> dict:
    return {"kind": "sys", "tag": tag, "text": text}


def act_me(param: str, text: str) -> dict:
    return {"kind": "me", "tag": param, "text": text}


def act_unset(param: str, text: str) -> dict:
    return {"kind": "unset", "tag": param, "text": text}


def act_note(text: str) -> dict:
    return {"kind": "note", "tag": "", "text": text}


def act_freq(text: str) -> dict:
    return {"kind": "freq", "tag": "非预测", "text": text}


def act_e13(text: str) -> dict:
    return {"kind": "e13", "tag": "参考·非推荐", "text": text}


def node(direction: str, no: str, cond: str, tag: str = "",
         acts: list | None = None, children: list | None = None,
         val: str = "") -> dict:
    return {"dir": direction, "no": no, "cond": cond, "tag": tag, "val": val,
            "acts": acts or [], "children": children or []}


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def _price_pct(target: float, live: float) -> str:
    return f"{(target / live - 1) * 100:+.1f}%"


def build_tree(live: float, s2: dict, pos: dict, det: dict | None,
               meta: dict, closes: list[float] | None = None,
               ) -> tuple[list[dict], list[str], dict | None]:
    """构建二叉预案树。返回（根分支列表[涨,跌], 备注行, 根节点频率动作）。

    分支价位全部由系统真实阈值换算：
      S2 解除/触发线 = 252 日滚动高 × 0.8（E11 冻结：dd < -20% 停用）
      下行档位      = 现价 × 0.9 / × 0.8
      个人线        = 252 日高 × (1 + pct/100)（与 S2 同口径）
    金额/股数按 position.json 的 account_value_usd 换算（示例值须改真实）；
    每个分支节点标注该分支价位下的持仓市值重估与历史 first-passage 频率。
    """
    H = s2["high"]
    release = H * S2_RATIO            # S2 解除/触发线
    d10, d20 = live * 0.9, live * 0.8
    dd_of = lambda p: (p / H - 1) * 100  # noqa: E731
    dd_now = float(s2["drawdown_pct"])
    s2_on = bool(s2["triggered"])

    ab = pos.get("add_budget_pct")
    trim = pos.get("trim_line_pct")
    pain = pos.get("max_pain_pct")
    bc = pos.get("emergency_borrow_cap_pct")
    trim_price = H * (1 + trim / 100) if trim is not None else None
    pain_price = H * (1 + pain / 100) if pain is not None else None

    # ---- 金额换算基数（account_value 为示例值时页面已在局面卡声明）
    av = float(pos.get("account_value_usd") or 100000)
    pos_val = av * float(pos.get("position_pct") or 0) / 100
    sh = pos_val / live if live else 0.0     # 现持股数（估算，随分支价重估市值）

    def mval(p: float) -> str:
        return f"持仓重估 ≈ {_usd(sh * p)}（{sh:.0f} 股 × {p:.2f}）"

    # ---- 历史频率（first-passage，滚动起点，非预测）
    ddser = rolling_dd_series(closes) if closes else []

    def freq_act_at(center_dd: float, base: float,
                    up_price: float, up_label: str,
                    dn_price: float | None, dn_label: str) -> dict:
        if not closes:
            return act_note("日线样本不可用——历史频率缺席（如实声明）")
        up_mv = (up_price / base - 1) * 100
        dn_mv = (dn_price / base - 1) * 100 if dn_price is not None else None
        fp = first_passage(closes, ddser, center_dd, up_mv, dn_mv)
        if fp is None:
            return act_note(f"dd≈{center_dd:+.0f}% 附近历史起点不足 30 个，"
                            "频率不显示（不装统计）")
        head = (f"从 dd≈{center_dd:+.0f}%±{FP_BAND_PP:.0f}pp 的 <b>{fp['n']}</b> "
                f"个滚动起点出发，{FP_HORIZON} 交易日内：")
        if dn_price is not None:
            body = (f"先到 <b>{dn_price:.2f}</b>（{dn_label} {dn_mv:+.1f}%）"
                    f"<b>{fp['dn']:.0f}%</b> ｜ 先到 <b>{up_price:.2f}</b>"
                    f"（{up_label} {up_mv:+.1f}%）<b>{fp['up']:.0f}%</b> ｜ "
                    f"两者都没到 <b>{fp['none']:.0f}%</b>")
        else:
            body = (f"到达 <b>{up_price:.2f}</b>（{up_label} {up_mv:+.1f}%）"
                    f"<b>{fp['up']:.0f}%</b> ｜ 没到 <b>{fp['none']:.0f}%</b>"
                    "（下行线未设，无三分口径）")
        return act_freq(head + body)

    def e13_card(depth_key: str, intro: str) -> dict:
        rows = "；".join(
            f"{d} 距锚 <b>{dep}</b> 借 <b>{amt}</b> {px}（{note}）"
            for d, dep, amt, px, note in E13_CASES[depth_key])
        return act_e13(
            f"{intro}：{rows}。口径：深度=距梯子锚点跌幅（非距 252 日高）；"
            "代表组合 s15_tp20_6mo_L50（权益 50% 比例额度），8 年窗口仅 4 次"
            "出手、借后缓冲 0.67-0.88、全程零触 30% 维持线")

    # ---- 个人参数动作（含金额换算；未设时降级提示 + 证据旁注）
    borrow_act = (act_me("emergency_borrow_cap_pct",
        f"若此深度要动用应急资金（E13 干预点形态）：借款上限 = 借款时点权益 "
        f"× {bc:g}% —— 按当前权益 {_usd(av)} 即 ≤<b>{_usd(av * bc / 100)}</b>。"
        "E13 实测：应急借款按权益比例、上限 ≤50%（固定金额授信 = 小账户隐形 "
        "3-5 倍杠杆，39 次危险事件 100% 来自固定 3 个月工资档；比例制全程零"
        "触线、缓冲 ≥0.65）。年化 6.5% 过桥、回款优先还清；E13 总判决未变：借款"
        "只是风控下限，不是收益来源") if bc is not None else
        act_unset("emergency_borrow_cap_pct",
                  "应急借款上限未设——建议按 E13 实测设为权益 ≤50%（比例制，"
                  "非固定金额）→ position.json emergency_borrow_cap_pct"))

    trim_unset = act_unset("trim_line_pct",
        "减仓线未设（默认不设 = 持有）——E10/E13 证据：择时减仓历史上跑输持有；"
        "若坚持要设，E5 显示高点回撤 10-15% 是常见选择但未验证 → trim_line_pct")
    pain_unset = act_unset("max_pain_pct",
        "容忍线未设（默认不设）——E5 的 -30% 强平对照 54 例全部恶化收益；"
        "设线属个人风险偏好而非统计优势 → max_pain_pct")

    # -- 探测器子分支（触发/解除），标定期内注明不参与
    det_state = (det or {}).get("state")
    det_note = None
    if det_state == "CALIBRATING":
        bd = int((det or {}).get("baseline_days") or 0)
        try:
            eta = add_bdays(date.fromisoformat(det["state_date"]),
                            max(0, CALIB_BDAYS - bd))
        except Exception:  # noqa: BLE001
            eta = None
        eta_s = eta.strftime("%m-%d") if eta else "?"
        det_note = (f"标定期 {bd}/{CALIB_BDAYS} 交易日，预计 {eta_s} 期满——"
                    f"{eta_s} 前探测器不参与，本子分支期满后才生效")
    elif det is None:
        det_note = "detector_state 无数据——探测器子分支暂不可用（如实声明）"

    det_branch = node("det", "②-A",
        "探测器子分支：空头 up-jump ≥ +%.0f%% × Musk 密集（回看 %d 交易日）"
        % (SHORT_JUMP_PCT, LOOKBACK_BDAYS),
        tag=("标定中" if det_state == "CALIBRATING" else
             "RISK_OFF" if det_state == "RISK_OFF" else ""),
        acts=[act_note(det_note)] if det_note else [],
        children=[
            node("down", "②-A-1", "触发 RISK_OFF", tag=f"F{PERSIST_BDAYS}",
                acts=[
                    act_sys("N3-H · N6",
                        f"记假想减仓单（前向虚拟推演，不碰真钱）；F{PERSIST_BDAYS}："
                        f"触发日起 {PERSIST_BDAYS} 个交易日 risk-off，重叠触发顺延。"
                        "拆股防护（N6）：空头 change_pct ≥ +50% 不自动触发、转人工复核。"
                        "窄谱声明：只防“空头知情型下跌”，对空头回补型/宏观型失明"),
                    act_unset("跟随假想单",
                        "真仓是否跟随假想单减仓：个人层未定义此参数"
                        "——此分支真仓无预案，待批准后补"),
                ]),
            node("up", "②-A-2", "RISK_OFF 解除 / 未触发（RISK_ON）",
                acts=[act_sys("N3-H",
                    "维持/恢复——注意 RISK_ON ≠ 安全：历史上 2024-12→2025-04 "
                    "的 -53.8% 整段无信号（盲区常驻声明），广谱防线仍看 S2")]),
        ])

    gate_s = (f"GBDT 门槛 {meta['gate']:.4f}，几何 tp +{meta['tp']:.1f}% / "
              f"sl -{meta['sl']:.1f}% / 超时 {meta['timeout']}×5m，当日平仓")

    trim_act = (act_me("trim_line_pct",
        f"减仓线 <b>{trim_price:.2f}</b>（252 日高 {trim:+.0f}%，距现价 "
        f"{_price_pct(trim_price, live)}）：收盘站上即按事先计划减老仓位——届时持仓"
        f"市值 ≈ {_usd(sh * trim_price)}（{sh:.0f} 股），减多少待批准时一并写入")
        if trim is not None else trim_unset)
    pain_act = (act_me("max_pain_pct",
        f"最大容忍线 <b>{pain_price:.2f}</b>（252 日高 {pain:+.0f}%，距现价 "
        f"{_price_pct(pain_price, live)}）：跌破即执行事先批准的减仓/清仓纪律——"
        f"若清仓即卖 {sh:.0f} 股 ≈ {_usd(sh * pain_price)} @线价")
        if pain is not None else pain_unset)

    if ab is not None:
        ab_usd = av * ab / 100
        add_act = act_me("add_budget_pct",
            f"解除确认后启用加仓预算 {ab:g}% = <b>{_usd(ab_usd)}</b> ≈ "
            f"<b>{ab_usd / release:.0f} 股</b> @{release:.2f}——分 3-4 笔"
            f"（每笔 ≤{_usd(ab_usd / 3)} ≈ {ab_usd / 3 / release:.0f} 股），"
            "每笔以 E8-A 信号为准逐笔执行，不追价、不一次性打满")
    else:
        add_act = act_unset("add_budget_pct",
            "加仓预算未设置——此分支无个人预案 → add_budget_pct")

    if s2_on:
        # ---- 当前 S2 已触发：上行主线=解除线，下行主线=-10%/-20% 档位
        up = node("up", "①",
            f"收盘站上 <b>{release:.2f}</b>"
            f"（S2 解除线 = 252 日高 {H:.2f} × 0.8，较现价 {_price_pct(release, live)}）",
            tag="S2 解除", val=mval(release),
            acts=[
                act_sys("E11 · E8-A",
                    "E8-A 恢复入场资格（当前 shadow 白跑、未上钱，裁决期 ≥8 周）；"
                    f"信号逐笔出：{gate_s}。S2 以昨日日收盘评估（shift(1)）——"
                    "盘中破线不算，收盘确认次日生效"
                    "（E18 全历史校验为负，该策略线证据等级已降级——见 strategy-lab）"),
                add_act,
                freq_act_at(S2_LINE_PCT, release, H, "收复 252 日高",
                            release * 0.9, "再回落 10%"),
            ],
            children=[
                node("up", "①-A",
                    f"继续上行收复 252 日高 <b>{H:.2f}</b>"
                    f"（较现价 {_price_pct(H, live)}）→ 创新高",
                    tag="S2 线上移", val=mval(H),
                    acts=[
                        act_sys("E11 · E10",
                            "S2 停用线随新高跟踪上移：每创新高 H，停用线 = H×0.8"
                            f"（例：新高 500.00 → 线 400.00；当前线 {release:.2f} 即"
                            f"由高点 {H:.2f} 而来）。系统在高位不做减仓预测"
                            "——E10 已证日线择时跑输持有"),
                        trim_act,
                        freq_act_at(0.0, H, H * 1.1, "续涨 10% 持续新高",
                                    H * 0.8, "自新高回撤 20%"),
                    ],
                    children=[
                        node("up", "①-A-1", "持续创新高",
                            acts=[act_sys("E11",
                                "停用线持续 = 最新高 ×0.8 跟踪抬升；加仓预算未用完"
                                "的部分继续只跟 E8-A 信号，不因创新高追加")]),
                        node("down", "①-A-2",
                            "自新高 H 回撤 20%（收盘跌破 H×0.8）", tag="S2 再触发",
                            val=mval(H * 0.8),
                            acts=[
                                act_sys("E11",
                                    "E8-A 停用、剩余加仓预算冻结；已建仓位按持有"
                                    "处理——系统无择时卖出规则（E10）"),
                                pain_act,
                            ]),
                    ]),
                node("down", "①-B",
                    f"解除后收盘跌回 <b>{release:.2f}</b> 之下", tag="S2 复触发",
                    val=mval(release),
                    acts=[
                        act_sys("E11",
                            "E8-A 再停用、剩余加仓预算冻结；已加部分不因复触发"
                            "卖出（E10：不做来回择时）"),
                        pain_act,
                    ]),
            ])
        down = node("down", "②",
            f"跌破 <b>{d10:.2f}</b>（现价 -10%，距 252 日高 {dd_of(d10):.1f}%）",
            val=mval(d10),
            acts=[
                act_sys("E11 · E10",
                    "S2 已在触发态——更深回撤不改变开关状态，系统无新增动作；"
                    "系统不做下跌中的择时卖出（E10：预测跑输持有）"),
                borrow_act,
                e13_card("shallow", "历史上走到这类浅坑时，E13 模型建议的出手"),
                pain_act if pain is not None else pain_unset,
                freq_act_at(dd_of(d10), d10, release, "收复解除线",
                            d20, "续跌至 -20% 档"),
            ],
            children=[
                det_branch,
                node("down", "②-B",
                    f"跌破 <b>{d20:.2f}</b>（现价 -20%，距 252 日高 {dd_of(d20):.1f}%）",
                    val=mval(d20),
                    acts=[
                        act_sys("N4 · E10",
                            "如实声明：系统在此深度没有加码/抄底规则——坑底签名"
                            "研究已判死（N4/N5），不猜坑底"),
                        borrow_act,
                        e13_card("deep", "历史上走到这类深坑时，E13 模型建议的出手"),
                        pain_act,
                        freq_act_at(dd_of(d20), d20, release, "收复解除线",
                                    pain_price, "触最大容忍线"),
                    ],
                    children=[
                        node("up", "②-B-1",
                            f"自低位反弹、收盘站回 <b>{release:.2f}</b>",
                            val=mval(release),
                            acts=[act_note("转入分支 ①（S2 解除路径），预案同 ①")]),
                        node("down", "②-B-2",
                            (f"继续跌破最大容忍线 <b>{pain_price:.2f}</b>"
                             f"（252 日高 {pain:+.0f}%）" if pain is not None else
                             "继续深跌（最大容忍线未设）"),
                            val=(mval(pain_price) if pain is not None else ""),
                            acts=[
                                pain_act,
                                e13_card("abyss",
                                         "历史最深处（-48%/-51%）E13 模型的出手"),
                            ]),
                    ]),
            ])
    else:
        # ---- S2 未触发：上行主线=新高跟踪，下行主线=S2 触发线
        up = node("up", "①",
            f"上行创 252 日新高（收盘超过 <b>{H:.2f}</b>，较现价 {_price_pct(H, live)}）",
            tag="S2 线上移", val=mval(H),
            acts=[
                act_sys("E11 · E8-A",
                    f"E8-A 维持入场资格（shadow 白跑），信号逐笔出：{gate_s}；"
                    "S2 停用线随新高跟踪上移：每创新高 H，停用线 = H×0.8"
                    "（E18 全历史校验为负，该策略线证据等级已降级——见 strategy-lab）"),
                trim_act,
                freq_act_at(0.0, H, H * 1.1, "续涨 10% 持续新高",
                            H * 0.8, "回撤 20% 触发 S2"),
            ],
            children=[
                node("up", "①-A", "持续创新高",
                    acts=[act_sys("E11", "停用线持续 = 最新高 ×0.8 跟踪抬升"),
                          trim_act]),
                node("down", "①-B",
                    f"回落、收盘跌破 <b>{release:.2f}</b>（= {H:.2f} × 0.8）",
                    tag="S2 触发", val=mval(release),
                    acts=[act_sys("E11",
                        "E8-A 停用入场；已建仓位按持有处理（系统无择时卖出规则，"
                        "E10）"), pain_act]),
            ])
        down = node("down", "②",
            f"下行收盘跌破 <b>{release:.2f}</b>"
            f"（S2 触发线 = 252 日高 {H:.2f} × 0.8，较现价 {_price_pct(release, live)}）",
            tag="S2 触发", val=mval(release),
            acts=[
                act_sys("E11",
                    "E8-A 停用入场（以昨日日收盘评估，shift(1)）；系统不做下跌中"
                    "的择时卖出（E10）"),
                borrow_act,
                e13_card("shallow", "历史上走到这类浅坑时，E13 模型建议的出手"),
                pain_act,
                freq_act_at(S2_LINE_PCT, release, H, "收复 252 日高",
                            d20, "续跌至现价 -20% 档"),
            ],
            children=[
                det_branch,
                node("down", "②-B",
                    f"跌破 <b>{d20:.2f}</b>（现价 -20%，距 252 日高 {dd_of(d20):.1f}%）",
                    val=mval(d20),
                    acts=[
                        act_sys("N4 · E10",
                            "如实声明：系统在此深度没有加码/抄底规则（N4/N5 判死）"),
                        borrow_act,
                        e13_card("deep", "历史上走到这类深坑时，E13 模型建议的出手"),
                        pain_act,
                        freq_act_at(dd_of(d20), d20, release, "收复 S2 线",
                                    pain_price, "触最大容忍线"),
                    ],
                    children=[
                        node("up", "②-B-1",
                            f"反弹、收盘站回 <b>{release:.2f}</b>",
                            val=mval(release),
                            acts=[act_note("S2 解除，转回上行路径 ①")]),
                        node("down", "②-B-2",
                            (f"继续跌破最大容忍线 <b>{pain_price:.2f}</b>"
                             f"（252 日高 {pain:+.0f}%）" if pain is not None else
                             "继续深跌（最大容忍线未设）"),
                            val=(mval(pain_price) if pain is not None else ""),
                            acts=[
                                pain_act,
                                e13_card("abyss",
                                         "历史最深处（-48%/-51%）E13 模型的出手"),
                            ]),
                    ]),
            ])

    # 根节点三分频率：现价上下文出发，涨支 vs 跌支谁先到
    if s2_on:
        root_freq = freq_act_at(dd_now, live, release, "S2 解除线",
                                d10, "现价 -10% 档")
    else:
        root_freq = freq_act_at(dd_now, live, H, "252 日新高",
                                release, "S2 触发线")

    notes = [
        f"档位换算基准：现价 {live:.2f}（生成时刻快照）、252 日滚动高 {H:.2f}"
        f"（{s2['high_date']}）。现价变动后 -10%/-20% 档位价随之变，以收盘确认为准。",
        f"金额换算基数：账户权益 {_usd(av)}（position.json account_value_usd，"
        f"示例值·请改为真实）× 仓位 {pos.get('position_pct', 0):g}% = 持仓 "
        f"{_usd(pos_val)} ≈ {sh:.0f} 股 @{live:.2f}；分支节点的'持仓重估'= 股数不变"
        "×分支价，未计新加仓。",
        f"历史频率口径：TSLA 2018-2026 单标的日线（{len(closes) if closes else 0} "
        f"个交易日）first-passage 统计——起点=回撤中心 ±{FP_BAND_PP:.0f}pp 内全部"
        f"交易日（滚动采样、样本高度重叠、非独立），视界 {FP_HORIZON} 交易日、"
        "逐日收盘首次穿越判定。历史频率 ≠ 概率预测。",
    ]
    # 高点滚出窗口的被动位移提示（近似交易日，不剔假日）
    try:
        roll_off = add_bdays(s2["high_date"], 252)
        notes.append(
            f"高点 {H:.2f} 约 {roll_off} 滚出 252 日窗口——若届时未创新高，"
            "S2 线将随窗内新高点下移（被动位移 ≠ 基本面改善，届时重新生成本页）。")
    except Exception:  # noqa: BLE001
        pass
    return [up, down], notes, root_freq


# ---------------------------------------------------------------- 渲染


_GLYPH = {"up": "▲", "down": "▼", "det": "◆"}
_ACT_LABEL = {"sys": "系统规则", "me": f"个人参数 · {UNVERIFIED}",
              "unset": "个人参数 · 未设置", "note": "",
              "freq": "历史频率", "e13": "E13 干预参考"}


def _act_html(a: dict) -> str:
    k = a["kind"]
    if k == "note":
        return f'<div class="act note">{a["text"]}</div>'
    label = _ACT_LABEL[k]
    tag = f' · {esc(a["tag"])}' if a["tag"] else ""
    return (f'<div class="act {k}"><span class="act-k">{esc(label)}{tag}</span>'
            f'<span class="act-t">{a["text"]}</span></div>')


def _node_html(n: dict) -> str:
    acts = "".join(_act_html(a) for a in n["acts"])
    kids = "".join(_node_html(c) for c in n["children"])
    tag = f'<span class="n-tag">{esc(n["tag"])}</span>' if n["tag"] else ""
    val = f'<span class="n-val">{esc(n["val"])}</span>' if n.get("val") else ""
    return (
        f'<div class="node {n["dir"]}">'
        f'<div class="n-line"><span class="glyph">{_GLYPH[n["dir"]]}</span>'
        f'<span class="n-no">{esc(n["no"])}</span>'
        f'<span class="cond">{n["cond"]}</span>{tag}{val}</div>'
        f'{acts}'
        + (f'<div class="kids">{kids}</div>' if kids else "")
        + "</div>")


def age_badge(d: date | None, now: datetime, warn_days: int, label: str) -> str:
    """数据龄徽章（沿用仪表盘 P0-2 规范：超 warn 转黄，双倍转红）。"""
    if d is None:
        return f'<span class="age crit">{esc(label)} 未知</span>'
    days = (now.astimezone(ET).date() - d).days
    cls = "crit" if days > 2 * warn_days else ("warn" if days > warn_days else "")
    txt = "今日" if days <= 0 else f"{days} 天前"
    return f'<span class="age {cls}">{esc(label)} {txt}</span>'


def _situation_html(live: float, px: dict, s2: dict, pos: dict,
                    det: dict | None, sh_s2: dict | None, now: datetime) -> str:
    dd = s2["drawdown_pct"]
    s2_on = bool(s2["triggered"])
    release = s2["high"] * S2_RATIO
    chg = px.get("chg_pct")
    chg_s = (f'<span class="{"good-text" if chg >= 0 else "crit-text"}">'
             f'{chg:+.2f}%</span>' if chg is not None else "—")

    # 探测器
    if det is None:
        det_v, det_cls, det_ref = "无数据", "crit", "detector_state 缺失"
    elif det["state"] == "CALIBRATING":
        bd = int(det.get("baseline_days") or 0)
        try:
            eta = add_bdays(date.fromisoformat(det["state_date"]),
                            max(0, CALIB_BDAYS - bd))
            eta_s = eta.strftime("%m-%d")
        except Exception:  # noqa: BLE001
            eta_s = "?"
        det_v, det_cls = "标定中", "warn"
        det_ref = f"基线 {bd}/{CALIB_BDAYS} 交易日 · 预计 {eta_s} 期满，期满前不参与"
    elif det["state"] == "RISK_OFF":
        det_v, det_cls = "RISK_OFF", "crit"
        det_ref = f"假想减仓中 · 至 {det.get('risk_off_until') or '?'}"
    else:
        det_v, det_cls = "未见目标风险", "good"
        det_ref = "窄谱：仅覆盖空头知情型下跌"

    # 个人参数芯片
    st = pos.get("_status", {})
    def chip(param: str, text: str) -> str:
        v = pos.get(param)
        if v is None:
            return f'<span class="pchip unset">{esc(text)}：未设置</span>'
        cls = "ok" if st.get(param) == "已确认" or st.get(param) == "已批准" else "pend"
        return f'<span class="pchip {cls}">{esc(text)}：{v:g}{"%" if param != "cost_basis" else ""}</span>'

    cost = pos.get("cost_basis")
    pnl_s = (f'盈亏 {(live / cost - 1) * 100:+.1f}%（成本 {cost:g}）'
             if cost else "成本未填 · 不显示盈亏")

    av = float(pos.get("account_value_usd") or 100000)
    mi = float(pos.get("monthly_inflow_usd") or 0)
    pos_val = av * float(pos.get("position_pct") or 0) / 100
    sh = pos_val / live if live else 0.0
    av_demo = st.get("account_value_usd") == "示例值·请改为真实"
    av_s = f"账户权益 {av:,.0f}" + ("（示例值·请改真实）" if av_demo else "")

    sh_row = ""
    if sh_s2:
        sh_row = (f'<div class="xref">影子引擎读数交叉对照：dd '
                  f'{sh_s2.get("dd_pct", 0) * 100:+.1f}%（asof {esc(sh_s2.get("asof"))}，'
                  f'shift(1) 引擎口径）· 开关 {"OFF 停用" if sh_s2.get("off") else "ON"}'
                  "——与上方实时读数的差为数据龄与口径差，均如实显示</div>")

    stale = ""
    if px.get("error") or px.get("from_cache"):
        stale = '<span class="pill crit"><span class="dot"></span>取价降级/STALE</span>'

    tiles = f"""
<div class="tile"><div class="t-k">现价 TSLA</div>
  <div class="t-v">{live:.2f}</div>
  <div class="t-r">最近日涨跌 {chg_s}（{esc(px.get("chg_date") or "—")}）{stale}</div></div>
<div class="tile"><div class="t-k">距 252 日高回撤</div>
  <div class="t-v {'crit-text' if dd < S2_LINE_PCT else ''}">{dd:+.1f}%</div>
  <div class="t-r">高点 {s2["high"]:.2f}（{esc(s2["high_date"])}）· S2 线 {S2_LINE_PCT:+.0f}%</div></div>
<div class="tile"><div class="t-k">S2 开关（E11 冻结）</div>
  <div class="t-v"><span class="pill {'crit' if s2_on else 'good'}"><span class="dot"></span>{'已触发 · E8-A 停用' if s2_on else '未触发 · E8-A 可入场'}</span></div>
  <div class="t-r">{'解除' if s2_on else '触发'}线 {release:.2f}（= 高 × 0.8）</div></div>
<div class="tile"><div class="t-k">探测器（N3 前向）</div>
  <div class="t-v"><span class="pill {det_cls}"><span class="dot"></span>{esc(det_v)}</span></div>
  <div class="t-r">{esc(det_ref)}</div></div>
<div class="tile"><div class="t-k">我的仓位（{esc(av_s)}）</div>
  <div class="t-v">{pos.get("position_pct", 0):g}% ≈ ${pos_val:,.0f}</div>
  <div class="t-r">≈ {sh:.0f} 股 @{live:.2f} · {esc(pnl_s)} · 月入金 ${mi:,.0f}（E13 口径）</div></div>
<div class="tile"><div class="t-k">个人参数</div>
  <div class="t-v t-chips">{chip("add_budget_pct", "加仓预算")}{chip("emergency_borrow_cap_pct", "应急借款上限")}{chip("trim_line_pct", "减仓线")}{chip("max_pain_pct", "容忍线")}</div>
  <div class="t-r">黄 = 待用户批准（含 E13 背书默认值）· 灰 = 未设置（data/intel/position.json）</div></div>
"""
    ages = (age_badge(px.get("price_asof"), now, 3, "价格数据龄")
            + age_badge(date.fromisoformat(det["state_date"]) if det else None,
                        now, 3, "探测器数据龄")
            + age_badge((datetime.fromisoformat(sh_s2["updated_at"]).astimezone(ET).date()
                         if sh_s2 and sh_s2.get("updated_at") else None), now, 3,
                        "影子引擎数据龄"))
    return (f'<div class="situ card"><div class="tiles">{tiles}</div>'
            f'{sh_row}<div class="ages">{ages}</div></div>')


_LIGHT_TOKENS = """
  color-scheme: light;
  --bg:#f2efe6; --surface:#faf8f1; --surface-2:#e9e3d3;
  --ink:#182028; --ink-2:#454f57; --muted:#77796c;
  --border:rgba(103,84,42,.28); --grid:#ddd5bf; --baseline:#b9ad8e;
  --good:#0c7a52; --good-text:#0a6746; --good-wash:rgba(12,122,82,.10);
  --warn:#b3880f; --warn-text:#84630a; --warn-wash:rgba(179,136,15,.13);
  --crit:#b13e06; --crit-text:#9a3708; --crit-wash:rgba(177,62,6,.08);
  --sys:#8a6520; --sys-wash:rgba(154,116,32,.10);
  --sys-rule:rgba(138,101,32,.38);
  --link:#8a6520;
"""

# 视觉 token 沿用 intel/dashboard.py 的设计稿色板（暗默认/亮覆盖、宋体衬线
# 板块题、等宽数据）；本页自包含，不 import 仪表盘（另一代理施工中）。
_CSS = """
:root {
  color-scheme: dark;
  --bg:#0a0f14; --surface:#121b22; --surface-2:#18242e;
  --ink:#f2efe6; --ink-2:#c9c2ae; --muted:#8a8c82;
  --border:rgba(214,175,110,.22); --grid:#22303a; --baseline:#46525b;
  --good:#7fd4a0; --good-text:#7fd4a0; --good-wash:rgba(127,212,160,.13);
  --warn:#dc9838; --warn-text:#dc9838; --warn-wash:rgba(220,152,56,.13);
  --crit:#da544a; --crit-text:#e0685e; --crit-wash:rgba(224,104,94,.14);
  --sys:#d6af6e; --sys-wash:rgba(214,175,110,.10);
  --sys-rule:rgba(214,175,110,.32);
  --link:#d6af6e;
  --font-serif:"Songti SC","STSong","Noto Serif CJK SC","Source Han Serif SC",serif;
  --font-sans:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
:root[data-theme="light"] { __LIGHT__ }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) { __LIGHT__ }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--ink);
  font: 14px/1.6 var(--font-sans); -webkit-font-smoothing: antialiased; }
main, .topbar-in { max-width: 1080px; margin: 0 auto; padding: 0 24px; }
b { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.good-text { color: var(--good-text); } .crit-text { color: var(--crit-text); }
.muted { color: var(--muted); }

.topbar { border-bottom: 1px solid var(--border); background: var(--surface); }
.topbar-in { display: flex; align-items: center; gap: 14px;
  padding-top: 14px; padding-bottom: 14px; flex-wrap: wrap; }
h1 { font: 600 21px/1.3 var(--font-serif); margin: 0; letter-spacing: .04em; }
.brand .sub { font-family: var(--font-mono); font-size: 11px; letter-spacing: .22em;
  color: var(--muted); text-transform: uppercase; margin-top: 1px; }
.topmeta { margin-left: auto; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.stamp .k { font-size: 11px; color: var(--muted); letter-spacing: .06em; }
.stamp .v { font-family: var(--font-mono); font-size: 12.5px;
  font-variant-numeric: tabular-nums; color: var(--ink-2); }
.themebtn { font: 12px/1 var(--font-mono); letter-spacing: .1em; color: var(--ink-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px;
  padding: 7px 12px; cursor: pointer; }
.themebtn:hover { color: var(--sys); border-color: var(--sys); }
a.backbtn { text-decoration: none; display: inline-flex; align-items: center; margin-right: 8px; }

section { margin-top: 40px; }
h2 { display: flex; align-items: baseline; gap: 10px; margin: 0 0 14px;
  font: 600 17px/1.4 var(--font-serif); letter-spacing: .05em; flex-wrap: wrap; }
h2 .sec-no { font: 400 11px var(--font-mono); letter-spacing: .18em; color: var(--sys); }
h2 .h-sub { font: 400 12px var(--font-sans); color: var(--muted); }
h2::after { content: ""; flex: 1; border-top: 1px solid var(--sys-rule);
  align-self: center; margin-left: 6px; min-width: 40px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }

.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
  font-weight: 600; font-family: var(--font-mono); letter-spacing: .04em;
  padding: 2px 9px; border: 1px solid var(--border); border-radius: 3px;
  color: var(--ink-2); white-space: nowrap; }
.pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex: none; }
.pill.good { color: var(--good-text); } .pill.good .dot { background: var(--good); }
.pill.warn { color: var(--warn-text); } .pill.warn .dot { background: var(--warn); }
.pill.crit { color: var(--crit-text); } .pill.crit .dot { background: var(--crit); }

/* ===== 当前局面卡 ===== */
.situ { padding: 16px 18px; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 860px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .tiles { grid-template-columns: 1fr; } }
.tile { background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 12px 14px; min-width: 0; }
.t-k { font-size: 11px; color: var(--muted); letter-spacing: .08em; }
.t-v { font: 600 22px/1.3 var(--font-mono); font-variant-numeric: tabular-nums;
  margin: 6px 0 3px; }
.t-v.t-chips { display: flex; flex-wrap: wrap; gap: 6px; font: inherit; }
.t-r { font-size: 11.5px; color: var(--muted); font-family: var(--font-mono); }
.pchip { font: 600 11px var(--font-mono); padding: 3px 8px; border-radius: 3px;
  border: 1px solid var(--border); color: var(--ink-2); white-space: nowrap; }
.pchip.pend { color: var(--warn-text); border-color: var(--warn);
  background: var(--warn-wash); }
.pchip.ok { color: var(--good-text); border-color: var(--good);
  background: var(--good-wash); }
.pchip.unset { color: var(--muted); border-style: dashed; }
.xref { margin-top: 12px; font: 12px var(--font-mono); color: var(--muted); }
.ages { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.age { font: 600 11px var(--font-mono); padding: 2px 8px; border-radius: 3px;
  border: 1px solid var(--border); color: var(--muted); }
.age.warn { color: var(--warn-text); border-color: var(--warn); }
.age.crit { color: var(--crit-text); border-color: var(--crit); }

/* ===== 二叉预案树（棋谱缩进风） ===== */
.tree { padding: 18px 20px; }
.root-line { font: 600 14px/1.6 var(--font-serif); color: var(--ink);
  padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface-2); }
.root-line .rp { font-family: var(--font-mono); font-weight: 600; color: var(--sys); }
.node { position: relative; margin-top: 14px; padding: 10px 12px 10px 14px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
.node.up   { border-left: 3px solid var(--good); }
.node.down { border-left: 3px solid var(--crit); }
.node.det  { border-left: 3px solid var(--warn); }
.n-line { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.glyph { font-size: 12px; }
.node.up > .n-line .glyph { color: var(--good); }
.node.down > .n-line .glyph { color: var(--crit); }
.node.det > .n-line .glyph { color: var(--warn); }
.n-no { font: 700 11px var(--font-mono); letter-spacing: .06em; color: var(--muted); }
.cond { font-size: 13.5px; color: var(--ink); }
.cond b { font-weight: 700; font-size: 14.5px; }
.n-tag { font: 600 10.5px var(--font-mono); letter-spacing: .06em; padding: 1px 7px;
  border: 1px solid var(--baseline); border-radius: 3px; color: var(--ink-2);
  white-space: nowrap; }
.n-val { font: 600 10.5px var(--font-mono); font-variant-numeric: tabular-nums;
  padding: 1px 7px; border: 1px solid var(--border); border-radius: 3px;
  color: var(--muted); background: var(--surface-2); white-space: nowrap; }
.kids { margin: 4px 0 0 18px; padding-left: 14px; border-left: 1px dashed var(--baseline); }
.act { display: flex; gap: 8px; align-items: flex-start; margin-top: 8px;
  font-size: 12.5px; line-height: 1.55; border-radius: 4px; padding: 6px 9px; }
.act .act-k { flex: none; font: 700 10.5px/1.9 var(--font-mono); letter-spacing: .04em;
  padding: 0 7px; border-radius: 3px; white-space: nowrap; }
.act.sys { background: var(--sys-wash); }
.act.sys .act-k { color: var(--sys); border: 1px solid var(--sys); }
.act.me { background: var(--warn-wash); }
.act.me .act-k { color: var(--warn-text); border: 1px solid var(--warn); }
/* 未设置参数：保留提示但降为次要样式（小字、无底色、不抢视觉重量） */
.act.unset { border: none; border-left: 2px dashed var(--baseline);
  border-radius: 0; padding: 1px 9px; margin-top: 6px;
  font-size: 11px; line-height: 1.5; color: var(--muted); opacity: .85; }
.act.unset .act-k { color: var(--muted); border: none; padding: 0;
  font-size: 9.5px; line-height: 1.9; letter-spacing: .02em; }
.act.unset .act-t { color: var(--muted); }
.act.note { color: var(--muted); font-size: 12px; border-left: 2px solid var(--baseline); }
/* 历史频率：等宽小字、虚线框——统计注解，不是动作 */
.act.freq { background: transparent; border: 1px dashed var(--border);
  font: 11.5px/1.6 var(--font-mono); color: var(--muted); }
.act.freq .act-k { color: var(--ink-2); border: 1px solid var(--baseline); }
.act.freq .act-t { color: var(--muted); }
.act.freq .act-t b { color: var(--ink-2); font-weight: 600; }
/* E13 干预参考卡：历史案例引用，区别于系统规则与个人参数 */
.act.e13 { background: var(--surface-2); border-left: 2px solid var(--muted); }
.act.e13 .act-k { color: var(--ink-2); border: 1px solid var(--muted); }
.act.e13 .act-t { color: var(--ink-2); font-size: 12px; }
.act.e13 .act-t b { color: var(--ink); }
.act .act-t { min-width: 0; color: var(--ink-2); }
.act.sys .act-t b, .act.me .act-t b { color: var(--ink); }

/* 懒人基准（机会成本锚） */
.bench-line { margin-bottom: 12px; padding: 9px 12px; border: 1px solid var(--border);
  border-left: 3px solid var(--sys); border-radius: 6px; background: var(--sys-wash);
  font-size: 12.5px; line-height: 1.6; color: var(--ink-2); }
.bench-line .bench-k { font: 700 10.5px var(--font-mono); letter-spacing: .06em;
  color: var(--sys); border: 1px solid var(--sys); border-radius: 3px;
  padding: 0 7px; margin-right: 8px; white-space: nowrap; }
.bench-line b { color: var(--ink); }

.notes { margin-top: 14px; font: 12px/1.7 var(--font-mono); color: var(--muted); }
.notes li { margin-top: 4px; }

footer { margin: 48px 0 40px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 12px; color: var(--muted); line-height: 1.8; }
footer .motto { font: 600 13.5px/1.7 var(--font-serif); color: var(--ink-2); }
""".replace("__LIGHT__", _LIGHT_TOKENS)

_JS = """
(function () {
  var root = document.documentElement;
  var q = null;
  try { q = new URLSearchParams(location.search).get("theme"); } catch (e) {}
  if (q === "light" || q === "dark") {
    root.dataset.theme = q;
  } else if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: light)");
    root.dataset.theme = mq.matches ? "light" : "dark";
    if (mq.addEventListener)
      mq.addEventListener("change", function (e) {
        if (!root.dataset.userTheme)
          root.dataset.theme = e.matches ? "light" : "dark";
      });
  }
  var btn = document.getElementById("themebtn");
  if (!btn) return;
  function label() {
    btn.textContent = root.dataset.theme === "dark" ? "切换亮色" : "切换暗色";
  }
  btn.addEventListener("click", function () {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.userTheme = "1";
    label();
  });
  label();
})();
"""


def render() -> str:
    now = datetime.now(timezone.utc)
    pos = load_position()
    det = load_detector_row()
    sh_s2 = load_shadow_s2()
    meta = load_e8a_meta()

    csv_dates, csv_closes = load_daily_closes()
    px: dict | None = None
    if get_price_context is not None:
        try:
            px = get_price_context(csv_dates, csv_closes, now=now)
        except Exception:  # noqa: BLE001
            px = None

    gen_s = now.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
    head = (f'<div class="topbar"><div class="topbar-in">'
            f'<div class="brand"><h1>棋谱预案 · TSLA</h1>'
            f'<div class="sub">PLAYBOOK — PLAN BEFORE EVENT</div></div>'
            f'<div class="topmeta">'
            f'<a class="themebtn backbtn" href="dashboard.html">← 返回仪表盘</a>'
            f'<div class="stamp"><div class="k">生成时刻</div>'
            f'<div class="v">{esc(gen_s)}</div></div>'
            f'<button class="themebtn" id="themebtn" type="button">切换主题</button>'
            f'</div></div></div>')

    if not px or not px.get("s2") or not px.get("closes"):
        body = ('<main><section><h2>数据不可用</h2><div class="card" '
                'style="padding:16px 18px">价格上下文获取失败（yfinance 与本地缓存'
                '均不可用）——预案树需要现价与 252 日高才能换算价位，本次不渲染、'
                '不装新鲜。修复取价后重跑 <code>python -m intel.playbook</code>。'
                '</div></section></main>')
    else:
        s2 = px["s2"]
        live = float(px.get("live_price") or px["closes"][-1])
        # 树价位以序列最后一根为准（与 S2 读数同源）；live 仅展示。
        # 历史频率用纯 CSV 日线（2018-2026，E13 同源样本），不掺临时今日点。
        ref = float(s2["ref_price"])
        tree, notes, root_freq = build_tree(ref, s2, pos, det, meta,
                                            closes=csv_closes)
        situ = _situation_html(ref, px, s2, pos, det, sh_s2, now)
        s2_state_s = ("S2 已触发（E8-A 停用）" if s2["triggered"]
                      else "S2 未触发（E8-A 可入场）")

        # 懒人基准：任何分支动作的机会成本参照（E13 实测合成）
        bench = load_e13_bench(csv_dates)
        bench_html = (
            '<div class="bench-line"><span class="bench-k">懒人基准</span>'
            "同月供 <b>30% DCA + 70% 现金(4%)</b>（不再平衡，E13 对照行合成）："
            f"8 年 XIRR <b>{bench['w2']:+.1f}%</b> / 3 年 <b>{bench['w1']:+.1f}%</b>"
            f"（全仓 DCA-hold {bench['dca_w2']:+.1f}% / {bench['dca_w1']:+.1f}%，"
            "现金 +4.0%）——树里任何分支动作若说不清凭什么赢过这一行，"
            "就不值得做（机会成本锚，出处 E13 §5）。</div>")

        av = float(pos.get("account_value_usd") or 100000)
        pos_val = av * float(pos.get("position_pct") or 0) / 100
        root_line = (f'<div class="root-line">当前局面：<span class="rp">'
                     f'{ref:.2f}</span> · 距 252 日高 <span class="rp">'
                     f'{s2["drawdown_pct"]:+.1f}%</span> · {esc(s2_state_s)}'
                     f' · 仓位 <span class="rp">{pos.get("position_pct", 0):g}%'
                     f' ≈ {_usd(pos_val)}</span>（账户 <span class="rp">{_usd(av)}'
                     "</span>）——以下每条分支都有应手：</div>")
        root_freq_html = _act_html(root_freq) if root_freq else ""
        notes_html = "".join(f"<li>{esc(t)}</li>" for t in notes)
        body = (f"<main><section><h2><span class='sec-no'>01</span>当前局面"
                f"<span class='h-sub'>换算基准与开关状态</span></h2>{situ}</section>"
                f"<section><h2><span class='sec-no'>02</span>二叉预案树"
                f"<span class='h-sub'>▲ 涨支 / ▼ 跌支 / ◆ 探测器支——"
                f"价位为系统真实阈值换算</span></h2>"
                f'<div class="tree card">{bench_html}{root_line}{root_freq_html}'
                + "".join(_node_html(n) for n in tree)
                + f'<ul class="notes">{notes_html}</ul></div></section>'
                "<footer>"
                '<div class="motto">本页是预案不是预测——树不预测走哪条分支，'
                '只保证每条分支都有应手（E10 已证明择时预测跑输持有）。</div>'
                "<div>口径：S2 = 距 252 交易日滚动高回撤 &lt; -20% 停用 E8-A 入场"
                "（E11 冻结，实盘引擎以昨日收盘 shift(1) 评估）；探测器 = N3-H "
                "冻结规则前向值班（虚拟推演，不碰真钱）；E8-A 处于 shadow 白跑期"
                "（≥8 周），全系统未上真钱——且 E18 全历史滚动校验（2020-2026 "
                "十二折）判定同配方年化 −5.45% 为负，该策略线证据等级已降级"
                "（docs/strategy-lab.md E18）。个人参数出自 data/intel/position.json，"
                f"黄色 = {UNVERIFIED}。</div>"
                "<div>历史频率 ≠ 概率预测：样本为 TSLA 2018-2026 单标的日线，"
                "first-passage 滚动起点采样（样本高度重叠、非独立），过去八年"
                "以强趋势上涨为主、频率随行情形态漂移（E13 已证'最优心里价'"
                "不可学习）——频率行只回答'历史上走到这里之后发生过什么'，"
                "不回答'接下来会发生什么'。E13 干预参考卡同理：历史案例引用，"
                "非操作推荐（E13 判决：该策略族 0/180 跑赢 DCA，残值仅风控规则"
                "与出手形态）。金额换算基于 account_value_usd（默认为示例值），"
                "均为四舍五入估算，不构成任何投资建议。</div>"
                "<div>生成：<code>python -m intel.playbook</code>；"
                "计划由 com.tsla.dashboard 周期任务在仪表盘生成后串带执行"
                "（plist 归仪表盘域，主线稍后接线）。</div>"
                "</footer></main>")

    return ("<!DOCTYPE html>\n<html lang='zh-CN'>\n<head>\n"
            "<meta charset='utf-8'>\n"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            "<meta http-equiv='refresh' content='600'>\n"
            "<title>棋谱预案 · TSLA</title>\n"
            f"<style>{_CSS}</style>\n</head>\n<body>\n"
            f"{head}\n{body}\n<script>{_JS}</script>\n</body>\n</html>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成棋谱预案页面（二叉决策树）")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    html = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"写出 {args.out}（{len(html)} 字节）")


if __name__ == "__main__":
    main()
