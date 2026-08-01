"""外部看门狗 —— 独立于被监控者的心跳检查（ops_review P1-5）.

设计原则：看门狗必须比被监控者更简单可靠——
  * 纯标准库，零第三方依赖；可用系统 /usr/bin/python3 运行（不吃 .venv 的挂法）
  * 任何单项检查异常（库锁死/文件损坏/JSON 烂掉）都不中断其余检查
  * 只读被监控者的产出（sqlite 只读 URI / 文件 mtime / JSON 时戳），绝不写入它们

产出：
  outputs/watchdog.json        逐系统 ok/stale + 龄期 + 全局状态（dashboard 顶栏消费）
  outputs/watchdog_state.json  每系统最近一次通知时刻（6 小时冷却防轰炸）
  outputs/watchdog.log         自身运行记录（>256KB 滚动到 .1，只留一代）

异常动作：macOS 系统通知（osascript display notification），内容含哪个系统
失联了多久；每系统每 NOTIFY_COOLDOWN_H 小时最多提醒一次。

调度：launchd com.tsla.watchdog（intel/deploy/com.tsla.watchdog.plist），
每 30 分钟一次。「谁来看守看守者」：dashboard 顶栏对 watchdog.json 自身
龄期做检查（>WD 自身 2.5 个周期 → 顶栏「看门狗失联」），双向互保。

用法：
    /usr/bin/python3 -m intel.watchdog          # 跑一轮检查（launchd 同款）
    /usr/bin/python3 -m intel.watchdog --dry    # 只打印，不写文件不发通知
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

DB_PATH = ROOT / "data" / "intel" / "sentinel.sqlite"
DASH_HTML = ROOT / "data" / "intel" / "dashboard.html"
BACKUP_DIR = ROOT / "backups"
REPLAY_META = ROOT / "outputs" / "replay_current" / "meta.json"
SHADOW_STATUS = ROOT / "outputs" / "shadow_status.json"

OUT_JSON = ROOT / "outputs" / "watchdog.json"
STATE_JSON = ROOT / "outputs" / "watchdog_state.json"
LOG_PATH = ROOT / "outputs" / "watchdog.log"
LOG_MAX_BYTES = 256 * 1024  # 超过滚动到 watchdog.log.1（只留一代）

INTERVAL_MIN = 30           # launchd StartInterval，写进 json 供 dashboard 算自身超龄
NOTIFY_COOLDOWN_H = 6       # 每系统通知冷却（防轰炸）

# ---------------------------------------------------------------- 阈值表
# 各子系统"最后活动"的新鲜度上限（分钟）。期望节奏各不同，写死并注释：
SENTINEL_OPEN_MIN = 45      # 哨兵 poll：launchd 每 5 分钟，盘中（ET 工作日
                            # 09:30–16:00）9 个周期没动静就算失联
SENTINEL_CLOSED_MIN = 150   # 盘外放宽到 2.5 小时（机器打盹/网络抖动常态）
DASHBOARD_MIN = 15          # dashboard.html：每 5 分钟重渲，3 个周期没更新算失联
BACKUP_MIN = 26 * 60        # intel.backup：每天 08:30 本地，26 小时容错过点补跑
REPLAY_AGE_MIN = 24 * 60    # replay meta：交易日内 ≤1 天——有已收盘交易日未续演
                            # 且距上次刷新 >24h 才算失联（周末不误报；周一联邦
                            # 假日会晚 1 天误报一次，接受，谁让它装作交易日）
SHADOW_SESSIONS = 3         # shadow_status：会话 ≤3 个交易日——距上次更新已过去
                            # ≥3 个已收盘工作日会话才算失联（长周末不误报）
INTERPRET_MIN = 36 * 60     # LLM 解读：每 ET 日至少落一行 llm_interpret_runs
                            # （ok/absent 都算活动；新闻源全年无休），36h 容余量
SCORER_MIN = 72 * 60        # 前向判分：有新解读/可了结才落 interp_scorer_runs，
                            # 无事静默速退；解读逐日产生 → 正常应逐日一行，
                            # 72h 覆盖长周末静默
WD_SELF_STALE_MIN = 75      # 看门狗自身：dashboard 侧判「看门狗失联」的阈值
                            # （2.5 个周期），写进 json 供 dashboard 消费


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s):
    """容错 ISO 时戳解析；naive 视为 UTC；解析失败返回 None."""
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        try:
            t = datetime.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- 交易时段

def market_open(now: datetime) -> bool:
    t = now.astimezone(ET)
    return t.weekday() < 5 and dtime(9, 30) <= t.time() < dtime(16, 0)


def _close_utc(d) -> datetime:
    return datetime.combine(d, dtime(16, 0), tzinfo=ET).astimezone(timezone.utc)


def last_close_utc(now: datetime) -> datetime:
    """最近一个已过去的工作日收盘时刻（ET 16:00；忽略假日，见阈值表注释）。"""
    d = now.astimezone(ET).date()
    while d.weekday() >= 5 or _close_utc(d) > now:
        d -= timedelta(days=1)
    return _close_utc(d)


def closes_between(ts: datetime, now: datetime) -> int:
    """(ts, now] 内已完成的工作日收盘个数（≈ 期间完整交易日会话数）。"""
    n, d = 0, ts.astimezone(ET).date()
    end = now.astimezone(ET).date()
    while d <= end:
        if d.weekday() < 5 and ts < _close_utc(d) <= now:
            n += 1
        d += timedelta(days=1)
    return n


# ---------------------------------------------------------------- 单项检查
# 每个检查返回 dict(ok, last_utc, age_min, threshold_min, note)；
# 抛什么都由 run_checks 兜住记成 ok=False + error，不影响其余项。

def _age_result(last: datetime, now: datetime, thr_min: float, note: str = ""):
    age = (now - last).total_seconds() / 60.0 if last else None
    return {
        "ok": last is not None and age <= thr_min,
        "last_utc": last.isoformat(timespec="seconds") if last else None,
        "age_min": round(age, 1) if age is not None else None,
        "threshold_min": thr_min,
        "note": note,
    }


def _query_one(sql: str):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def check_sentinel(now: datetime):
    last = parse_ts(_query_one("SELECT MAX(poll_time_utc) FROM poll_log"))
    if market_open(now):
        return _age_result(last, now, SENTINEL_OPEN_MIN, "盘中阈值 45 分钟")
    return _age_result(last, now, SENTINEL_CLOSED_MIN, "盘外阈值 2.5 小时")


def check_dashboard(now: datetime):
    last = datetime.fromtimestamp(DASH_HTML.stat().st_mtime, tz=timezone.utc)
    return _age_result(last, now, DASHBOARD_MIN, "html mtime")


def check_backup(now: datetime):
    # 只认每日备份产物 sentinel-YYYYMMDD.tar.gz，手工快照不算（不能掩盖备份任务死掉）
    files = list(BACKUP_DIR.glob("sentinel-*.tar.gz"))
    if not files:
        return _age_result(None, now, BACKUP_MIN, "无 sentinel-*.tar.gz 备份产物")
    newest = max(files, key=lambda p: p.stat().st_mtime)
    last = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
    return _age_result(last, now, BACKUP_MIN, f"最新 {newest.name}")


def check_replay(now: datetime):
    meta = json.loads(REPLAY_META.read_text(encoding="utf-8"))
    last = parse_ts(meta.get("generated_utc"))
    r = _age_result(last, now, REPLAY_AGE_MIN, "交易日内 ≤1 天")
    # 周末/节后放宽：仅当「存在已收盘交易日晚于上次刷新」时，超 24h 才算失联
    if not r["ok"] and last is not None and last_close_utc(now) <= last:
        r["ok"] = True
        r["note"] = "超 24h 但期间无新收盘交易日（周末/假日），放行"
    return r


def check_shadow(now: datetime):
    st = json.loads(SHADOW_STATUS.read_text(encoding="utf-8"))
    last = parse_ts(st.get("updated_at"))
    age = (now - last).total_seconds() / 60.0 if last else None
    n = closes_between(last, now) if last else 99
    return {
        "ok": last is not None and n < SHADOW_SESSIONS,
        "last_utc": last.isoformat(timespec="seconds") if last else None,
        "age_min": round(age, 1) if age is not None else None,
        "threshold_min": None,
        "note": f"距上次更新已过 {n} 个交易日会话（阈值 <{SHADOW_SESSIONS}）",
    }


def check_interpret(now: datetime):
    last = parse_ts(_query_one("SELECT MAX(created_utc) FROM llm_interpret_runs"))
    return _age_result(last, now, INTERPRET_MIN, "逐日解读（ok/absent 都算活动）")


def check_scorer(now: datetime):
    last = parse_ts(_query_one("SELECT MAX(run_utc) FROM interp_scorer_runs"))
    return _age_result(last, now, SCORER_MIN, "有事才落行，72h 覆盖长周末静默")


CHECKS = [
    ("sentinel", "哨兵采集", check_sentinel),
    ("dashboard", "仪表盘渲染", check_dashboard),
    ("backup", "每日备份", check_backup),
    ("replay", "推演续算", check_replay),
    ("shadow", "影子会话", check_shadow),
    ("interpret", "LLM 解读", check_interpret),
    ("scorer", "前向判分", check_scorer),
]


def run_checks(now: datetime) -> dict:
    systems = {}
    for key, label, fn in CHECKS:
        try:
            r = fn(now)
        except Exception as e:  # noqa: BLE001 —— 单项失败不中断其余检查
            r = {"ok": False, "last_utc": None, "age_min": None,
                 "threshold_min": None, "note": "检查自身异常"}
            r["error"] = f"{type(e).__name__}: {e}"
        r["label"] = label
        r.setdefault("error", None)
        systems[key] = r
    stale = [k for k, r in systems.items() if not r["ok"]]
    return {
        "generated_utc": now.isoformat(timespec="seconds"),
        "overall": "stale" if stale else "ok",
        "stale": stale,
        "interval_min": INTERVAL_MIN,
        "self_stale_min": WD_SELF_STALE_MIN,
        "market_open": market_open(now),
        "systems": systems,
    }


# ---------------------------------------------------------------- 通知与日志

def _fmt_age(age_min) -> str:
    if age_min is None:
        return "无记录"
    if age_min < 90:
        return f"{age_min:.0f} 分钟"
    if age_min < 48 * 60:
        return f"{age_min / 60:.1f} 小时"
    return f"{age_min / 1440:.1f} 天"


def notify(title: str, body: str) -> bool:
    """macOS 系统通知；失败只记日志不抛（通知是尽力而为）。"""
    script = 'display notification "{}" with title "{}"'.format(
        body.replace('"', "'"), title.replace('"', "'"))
    try:
        subprocess.run(["osascript", "-e", script], timeout=15,
                       capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_state() -> dict:
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def maybe_notify(report: dict, now: datetime, dry: bool) -> list:
    """对 stale 系统发系统通知；每系统 6 小时冷却。返回本轮已通知列表。"""
    state = _load_state()
    sent = []
    for key in report["stale"]:
        r = report["systems"][key]
        last_n = parse_ts(state.get(key))
        if last_n and (now - last_n) < timedelta(hours=NOTIFY_COOLDOWN_H):
            continue
        body = f"{r['label']}（{key}）已 {_fmt_age(r['age_min'])} 无活动"
        if r.get("error"):
            body = f"{r['label']}（{key}）检查异常：{r['error'][:120]}"
        elif r.get("note"):
            body += f" · {r['note']}"
        if dry:
            print(f"[dry] 通知：{body}")
            sent.append(key)
        elif notify("TSLA 看门狗告警", body):
            state[key] = now.isoformat(timespec="seconds")
            sent.append(key)
    # 恢复的系统清掉冷却记录，下次再失联可立即提醒
    for key in list(state):
        if key in report["systems"] and report["systems"][key]["ok"]:
            del state[key]
    if not dry:
        try:
            STATE_JSON.write_text(
                json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] state 写入失败：{e}", file=sys.stderr)
    return sent


def append_log(report: dict, sent: list) -> None:
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_MAX_BYTES:
            LOG_PATH.replace(LOG_PATH.with_suffix(".log.1"))  # 简单滚动，留一代
        parts = [f"{k}={'ok' if r['ok'] else 'STALE'}/{_fmt_age(r['age_min'])}"
                 for k, r in report["systems"].items()]
        line = (f"{report['generated_utc']} overall={report['overall']} "
                + " ".join(parts)
                + (f" notified={','.join(sent)}" if sent else "") + "\n")
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:  # noqa: BLE001
        print(f"[watchdog] 日志写入失败：{e}", file=sys.stderr)


def main() -> int:
    dry = "--dry" in sys.argv[1:]
    now = now_utc()
    report = run_checks(now)
    sent = maybe_notify(report, now, dry)
    if dry:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        try:
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            tmp = OUT_JSON.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(OUT_JSON)  # 原子替换，dashboard 读不到半截文件
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] watchdog.json 写入失败：{e}", file=sys.stderr)
        append_log(report, sent)
    summary = " ".join(
        f"{k}={'ok' if r['ok'] else 'STALE'}" for k, r in report["systems"].items())
    print(f"[watchdog] {report['overall']} · {summary}"
          + (f" · 通知 {sent}" if sent else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
