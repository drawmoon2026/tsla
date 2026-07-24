"""棋谱预案 — 预案先于事件的二叉决策树页面生成器.

思想：树不预测走哪条分支，只保证每条分支都有应手（E10 已证明择时预测
跑输持有）。页面把系统真实阈值换算成具体价位，涨/跌两路各推 3-4 层，
每个分支点标注两类动作：

  「系统规则」  冻结实验产物，附实验编号（E11 / N3-H / E8-A / N6 / E10）
  「个人参数」  data/intel/position.json 里的用户仓位参数——
                除 position_pct 外默认值一律标"未验证·待用户批准"（黄色）；
                未设置的分支如实显示"未设置——此分支无预案，建议补全"

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
# 负数=回撤，如 -20 即 252 日高 × 0.8）。除 position_pct 外默认值未经用户
# 批准——页面一律黄标，改完本文件重跑 `python -m intel.playbook` 即生效。
DEFAULT_POSITION: dict = {
    "_说明": {
        "position_pct": "当前仓位占可投资金的百分比（0-100）",
        "cost_basis": "持仓成本价（美元/股）；null=未填，页面不显示盈亏",
        "add_budget_pct": "S2 解除后可动用的加仓预算，占可投资金百分比；"
                          "只在 S2 解除分支生效，逐笔跟 E8-A 信号分批",
        "trim_line_pct": "老仓位减仓线：距 252 日高的百分比（如 -10 → 高×0.90）；"
                         "null=未设，上行分支将显示'无预案'",
        "max_pain_pct": "最大容忍线：距 252 日高的百分比（如 -50 → 高×0.50）；"
                        "null=未设，下行深跌分支将显示'无预案'",
        "_status": "各参数的验证状态；用户确认某参数后把状态改为'已批准'",
    },
    "position_pct": 30,
    "cost_basis": None,
    "add_budget_pct": 20,
    "trim_line_pct": None,
    "max_pain_pct": None,
    "_status": {
        "position_pct": "已确认",
        "cost_basis": UNVERIFIED,
        "add_budget_pct": UNVERIFIED,
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
    pos.setdefault("_status", {})
    for k, v in DEFAULT_POSITION["_status"].items():
        pos["_status"].setdefault(k, v)
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


# ---------------------------------------------------------------- 树模型


def act_sys(tag: str, text: str) -> dict:
    return {"kind": "sys", "tag": tag, "text": text}


def act_me(param: str, text: str) -> dict:
    return {"kind": "me", "tag": param, "text": text}


def act_unset(param: str, text: str) -> dict:
    return {"kind": "unset", "tag": param, "text": text}


def act_note(text: str) -> dict:
    return {"kind": "note", "tag": "", "text": text}


def node(direction: str, no: str, cond: str, tag: str = "",
         acts: list | None = None, children: list | None = None) -> dict:
    return {"dir": direction, "no": no, "cond": cond, "tag": tag,
            "acts": acts or [], "children": children or []}


def _price_pct(target: float, live: float) -> str:
    return f"{(target / live - 1) * 100:+.1f}%"


def _me_or_unset(pos: dict, param: str, set_text: str | None, unset_hint: str) -> dict:
    """个人参数动作：已设→黄标预案；未设→'无预案，建议补全'。"""
    if pos.get(param) is not None and set_text:
        return act_me(param, set_text)
    return act_unset(param, f"{unset_hint}——此分支无预案，建议补全（{param}）")


def build_tree(live: float, s2: dict, pos: dict, det: dict | None,
               meta: dict) -> tuple[list[dict], list[str]]:
    """构建二叉预案树。返回（根分支列表[涨,跌], 备注行）。

    分支价位全部由系统真实阈值换算：
      S2 解除/触发线 = 252 日滚动高 × 0.8（E11 冻结：dd < -20% 停用）
      下行档位      = 现价 × 0.9 / × 0.8
      个人线        = 252 日高 × (1 + pct/100)（与 S2 同口径）
    """
    H = s2["high"]
    release = H * S2_RATIO            # S2 解除/触发线
    d10, d20 = live * 0.9, live * 0.8
    dd_of = lambda p: (p / H - 1) * 100  # noqa: E731
    s2_on = bool(s2["triggered"])

    ab = pos.get("add_budget_pct")
    trim = pos.get("trim_line_pct")
    pain = pos.get("max_pain_pct")
    trim_price = H * (1 + trim / 100) if trim is not None else None
    pain_price = H * (1 + pain / 100) if pain is not None else None

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

    trim_act = _me_or_unset(pos, "trim_line_pct",
        (f"减仓线 <b>{trim_price:.2f}</b>（252 日高 {trim:+.0f}%）：收盘站上即按"
         "事先计划减老仓位，减多少待批准时一并写入" if trim is not None else None),
        "老仓位减仓线未设置")
    pain_act = _me_or_unset(pos, "max_pain_pct",
        (f"最大容忍线 <b>{pain_price:.2f}</b>（252 日高 {pain:+.0f}%）：跌破即执行"
         "事先批准的减仓/清仓纪律" if pain is not None else None),
        "最大容忍线未设置")

    if s2_on:
        # ---- 当前 S2 已触发：上行主线=解除线，下行主线=-10%/-20% 档位
        up = node("up", "①",
            f"收盘站上 <b>{release:.2f}</b>"
            f"（S2 解除线 = 252 日高 {H:.2f} × 0.8，较现价 {_price_pct(release, live)}）",
            tag="S2 解除",
            acts=[
                act_sys("E11 · E8-A",
                    "E8-A 恢复入场资格（当前 shadow 白跑、未上钱，裁决期 ≥8 周）；"
                    f"信号逐笔出：{gate_s}。S2 以昨日日收盘评估（shift(1)）——"
                    "盘中破线不算，收盘确认次日生效"),
                (act_me("add_budget_pct",
                    f"解除确认后启用加仓预算 {ab:g}%：按预算分批（建议 3-4 笔、"
                    f"每笔 ≤{ab / 3:.0f}%），每笔以 E8-A 信号为准逐笔执行，"
                    "不追价、不一次性打满") if ab is not None else
                 act_unset("add_budget_pct", "加仓预算未设置——此分支无个人预案")),
            ],
            children=[
                node("up", "①-A",
                    f"继续上行收复 252 日高 <b>{H:.2f}</b>"
                    f"（较现价 {_price_pct(H, live)}）→ 创新高",
                    tag="S2 线上移",
                    acts=[
                        act_sys("E11 · E10",
                            "S2 停用线随新高跟踪上移：每创新高 H，停用线 = H×0.8"
                            f"（例：新高 500.00 → 线 400.00；当前线 {release:.2f} 即"
                            f"由高点 {H:.2f} 而来）。系统在高位不做减仓预测"
                            "——E10 已证日线择时跑输持有"),
                        trim_act,
                    ],
                    children=[
                        node("up", "①-A-1", "持续创新高",
                            acts=[act_sys("E11",
                                "停用线持续 = 最新高 ×0.8 跟踪抬升；加仓预算未用完"
                                "的部分继续只跟 E8-A 信号，不因创新高追加")]),
                        node("down", "①-A-2",
                            "自新高 H 回撤 20%（收盘跌破 H×0.8）", tag="S2 再触发",
                            acts=[
                                act_sys("E11",
                                    "E8-A 停用、剩余加仓预算冻结；已建仓位按持有"
                                    "处理——系统无择时卖出规则（E10）"),
                                pain_act,
                            ]),
                    ]),
                node("down", "①-B",
                    f"解除后收盘跌回 <b>{release:.2f}</b> 之下", tag="S2 复触发",
                    acts=[
                        act_sys("E11",
                            "E8-A 再停用、剩余加仓预算冻结；已加部分不因复触发"
                            "卖出（E10：不做来回择时）"),
                        pain_act,
                    ]),
            ])
        down = node("down", "②",
            f"跌破 <b>{d10:.2f}</b>（现价 -10%，距 252 日高 {dd_of(d10):.1f}%）",
            acts=[
                act_sys("E11 · E10",
                    "S2 已在触发态——更深回撤不改变开关状态，系统无新增动作；"
                    "系统不做下跌中的择时卖出（E10：预测跑输持有）"),
                pain_act if pain is not None else
                act_unset("max_pain_pct",
                    "最大容忍线未设置——此分支无预案，建议补全（max_pain_pct）"),
            ],
            children=[
                det_branch,
                node("down", "②-B",
                    f"跌破 <b>{d20:.2f}</b>（现价 -20%，距 252 日高 {dd_of(d20):.1f}%）",
                    acts=[
                        act_sys("N4 · E10",
                            "如实声明：系统在此深度没有加码/抄底规则——坑底签名"
                            "研究已判死（N4/N5），不猜坑底"),
                        pain_act,
                    ],
                    children=[
                        node("up", "②-B-1",
                            f"自低位反弹、收盘站回 <b>{release:.2f}</b>",
                            acts=[act_note("转入分支 ①（S2 解除路径），预案同 ①")]),
                        node("down", "②-B-2",
                            (f"继续跌破最大容忍线 <b>{pain_price:.2f}</b>"
                             f"（252 日高 {pain:+.0f}%）" if pain is not None else
                             "继续下跌（最大容忍线未设置）"),
                            acts=[pain_act]),
                    ]),
            ])
    else:
        # ---- S2 未触发：上行主线=新高跟踪，下行主线=S2 触发线
        up = node("up", "①",
            f"上行创 252 日新高（收盘超过 <b>{H:.2f}</b>，较现价 {_price_pct(H, live)}）",
            tag="S2 线上移",
            acts=[
                act_sys("E11 · E8-A",
                    f"E8-A 维持入场资格（shadow 白跑），信号逐笔出：{gate_s}；"
                    "S2 停用线随新高跟踪上移：每创新高 H，停用线 = H×0.8"),
                trim_act,
            ],
            children=[
                node("up", "①-A", "持续创新高",
                    acts=[act_sys("E11", "停用线持续 = 最新高 ×0.8 跟踪抬升"),
                          trim_act]),
                node("down", "①-B",
                    f"回落、收盘跌破 <b>{release:.2f}</b>（= {H:.2f} × 0.8）",
                    tag="S2 触发",
                    acts=[act_sys("E11",
                        "E8-A 停用入场；已建仓位按持有处理（系统无择时卖出规则，"
                        "E10）"), pain_act]),
            ])
        down = node("down", "②",
            f"下行收盘跌破 <b>{release:.2f}</b>"
            f"（S2 触发线 = 252 日高 {H:.2f} × 0.8，较现价 {_price_pct(release, live)}）",
            tag="S2 触发",
            acts=[
                act_sys("E11",
                    "E8-A 停用入场（以昨日日收盘评估，shift(1)）；系统不做下跌中"
                    "的择时卖出（E10）"),
                pain_act,
            ],
            children=[
                det_branch,
                node("down", "②-B",
                    f"跌破 <b>{d20:.2f}</b>（现价 -20%，距 252 日高 {dd_of(d20):.1f}%）",
                    acts=[
                        act_sys("N4 · E10",
                            "如实声明：系统在此深度没有加码/抄底规则（N4/N5 判死）"),
                        pain_act,
                    ],
                    children=[
                        node("up", "②-B-1",
                            f"反弹、收盘站回 <b>{release:.2f}</b>",
                            acts=[act_note("S2 解除，转回上行路径 ①")]),
                        node("down", "②-B-2",
                            (f"继续跌破最大容忍线 <b>{pain_price:.2f}</b>"
                             f"（252 日高 {pain:+.0f}%）" if pain is not None else
                             "继续下跌（最大容忍线未设置）"),
                            acts=[pain_act]),
                    ]),
            ])

    notes = [
        f"档位换算基准：现价 {live:.2f}（生成时刻快照）、252 日滚动高 {H:.2f}"
        f"（{s2['high_date']}）。现价变动后 -10%/-20% 档位价随之变，以收盘确认为准。",
    ]
    # 高点滚出窗口的被动位移提示（近似交易日，不剔假日）
    try:
        roll_off = add_bdays(s2["high_date"], 252)
        notes.append(
            f"高点 {H:.2f} 约 {roll_off} 滚出 252 日窗口——若届时未创新高，"
            "S2 线将随窗内新高点下移（被动位移 ≠ 基本面改善，届时重新生成本页）。")
    except Exception:  # noqa: BLE001
        pass
    return [up, down], notes


# ---------------------------------------------------------------- 渲染


_GLYPH = {"up": "▲", "down": "▼", "det": "◆"}
_ACT_LABEL = {"sys": "系统规则", "me": f"个人参数 · {UNVERIFIED}",
              "unset": "个人参数 · 未设置", "note": ""}


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
    return (
        f'<div class="node {n["dir"]}">'
        f'<div class="n-line"><span class="glyph">{_GLYPH[n["dir"]]}</span>'
        f'<span class="n-no">{esc(n["no"])}</span>'
        f'<span class="cond">{n["cond"]}</span>{tag}</div>'
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
<div class="tile"><div class="t-k">我的仓位</div>
  <div class="t-v">{pos.get("position_pct", 0):g}%</div>
  <div class="t-r">{esc(pnl_s)}</div></div>
<div class="tile"><div class="t-k">个人参数</div>
  <div class="t-v t-chips">{chip("add_budget_pct", "加仓预算")}{chip("trim_line_pct", "减仓线")}{chip("max_pain_pct", "容忍线")}</div>
  <div class="t-r">黄 = {UNVERIFIED} · 灰 = 未设置（data/intel/position.json）</div></div>
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
  --bg:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ec;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --border:rgba(11,11,11,.10); --grid:#e1e0d9; --baseline:#c3c2b7;
  --good:#0ca30c; --good-text:#006300; --good-wash:rgba(12,163,12,.08);
  --warn:#fab219; --warn-text:#7a5200; --warn-wash:rgba(250,178,25,.14);
  --crit:#d03b3b; --crit-text:#b02a2a; --crit-wash:rgba(208,59,59,.07);
  --sys:#1c5cab; --sys-wash:rgba(28,92,171,.07);
  --link:#1c5cab;
"""

# 视觉 token 沿用 intel/dashboard.py 的设计稿色板（暗默认/亮覆盖、宋体衬线
# 板块题、等宽数据）；本页自包含，不 import 仪表盘（另一代理施工中）。
_CSS = """
:root {
  color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19; --surface-2:#232322;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --border:rgba(255,255,255,.10); --grid:#2c2c2a; --baseline:#383835;
  --good:#0ca30c; --good-text:#0ca30c; --good-wash:rgba(12,163,12,.16);
  --warn:#fab219; --warn-text:#fab219; --warn-wash:rgba(250,178,25,.13);
  --crit:#d03b3b; --crit-text:#e66767; --crit-wash:rgba(208,59,59,.14);
  --sys:#5598e7; --sys-wash:rgba(85,152,231,.10);
  --link:#86b6ef;
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
.themebtn:hover { color: var(--ink); border-color: var(--muted); }

section { margin-top: 40px; }
h2 { display: flex; align-items: baseline; gap: 10px; margin: 0 0 14px;
  font: 600 17px/1.4 var(--font-serif); letter-spacing: .05em; flex-wrap: wrap; }
h2 .sec-no { font: 400 11px var(--font-mono); letter-spacing: .18em; color: var(--muted); }
h2 .h-sub { font: 400 12px var(--font-sans); color: var(--muted); }
h2::after { content: ""; flex: 1; border-top: 1px solid var(--border);
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
.root-line .rp { font-family: var(--font-mono); font-weight: 600; }
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
.kids { margin: 4px 0 0 18px; padding-left: 14px; border-left: 1px dashed var(--baseline); }
.act { display: flex; gap: 8px; align-items: flex-start; margin-top: 8px;
  font-size: 12.5px; line-height: 1.55; border-radius: 4px; padding: 6px 9px; }
.act .act-k { flex: none; font: 700 10.5px/1.9 var(--font-mono); letter-spacing: .04em;
  padding: 0 7px; border-radius: 3px; white-space: nowrap; }
.act.sys { background: var(--sys-wash); }
.act.sys .act-k { color: var(--sys); border: 1px solid var(--sys); }
.act.me { background: var(--warn-wash); }
.act.me .act-k { color: var(--warn-text); border: 1px solid var(--warn); }
.act.unset { border: 1px dashed var(--baseline); color: var(--muted); }
.act.unset .act-k { color: var(--muted); border: 1px solid var(--baseline); }
.act.note { color: var(--muted); font-size: 12px; border-left: 2px solid var(--baseline); }
.act .act-t { min-width: 0; color: var(--ink-2); }
.act.sys .act-t b, .act.me .act-t b { color: var(--ink); }

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
            f'<div class="topmeta"><div class="stamp"><div class="k">生成时刻</div>'
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
        # 树价位以序列最后一根为准（与 S2 读数同源）；live 仅展示
        ref = float(s2["ref_price"])
        tree, notes = build_tree(ref, s2, pos, det, meta)
        situ = _situation_html(ref, px, s2, pos, det, sh_s2, now)
        s2_state_s = ("S2 已触发（E8-A 停用）" if s2["triggered"]
                      else "S2 未触发（E8-A 可入场）")
        root_line = (f'<div class="root-line">当前局面：<span class="rp">'
                     f'{ref:.2f}</span> · 距 252 日高 <span class="rp">'
                     f'{s2["drawdown_pct"]:+.1f}%</span> · {esc(s2_state_s)}'
                     f' · 仓位 <span class="rp">{pos.get("position_pct", 0):g}%</span>'
                     "——以下每条分支都有应手：")
        notes_html = "".join(f"<li>{esc(t)}</li>" for t in notes)
        body = (f"<main><section><h2><span class='sec-no'>01</span>当前局面"
                f"<span class='h-sub'>换算基准与开关状态</span></h2>{situ}</section>"
                f"<section><h2><span class='sec-no'>02</span>二叉预案树"
                f"<span class='h-sub'>▲ 涨支 / ▼ 跌支 / ◆ 探测器支——"
                f"价位为系统真实阈值换算</span></h2>"
                f'<div class="tree card">{root_line}'
                + "".join(_node_html(n) for n in tree)
                + f'<ul class="notes">{notes_html}</ul></div></section>'
                "<footer>"
                '<div class="motto">本页是预案不是预测——树不预测走哪条分支，'
                '只保证每条分支都有应手（E10 已证明择时预测跑输持有）。</div>'
                "<div>口径：S2 = 距 252 交易日滚动高回撤 &lt; -20% 停用 E8-A 入场"
                "（E11 冻结，实盘引擎以昨日收盘 shift(1) 评估）；探测器 = N3-H "
                "冻结规则前向值班（虚拟推演，不碰真钱）；E8-A 处于 shadow 白跑期"
                "（≥8 周），全系统未上真钱。个人参数出自 data/intel/position.json，"
                f"黄色 = {UNVERIFIED}。</div>"
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
