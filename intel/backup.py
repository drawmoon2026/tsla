"""前向数据每日备份 — 不可再生资产保护.

备份对象（全部是"从今天起攒、丢了买不回"的前向数据）：
- data/intel/sentinel.sqlite            哨兵双时间戳事件库（sqlite3 backup API 热备）
- data/intel/position.json              实盘持仓记录
- outputs/shadow_live/journal.sqlite    shadow 交易日记（热备）
- outputs/shadow_e8a/journal.sqlite     shadow E8-A 日记（热备）
- outputs/n8_scoring/                   N8 前向判分产物（整目录）
- outputs/replay_current/meta.json      推演续演状态
- data/intel/musk_tweets.csv            Musk 推文归档（72k 条不可再生，探测器标定源）
- data/TSLA_5m_3y.csv                   冻结数据源单副本（meta.json 引用其 sha256）

体积（P0-C 实测 2026-08-02）：两个 CSV 原文 10.7 MB + 3.8 MB，压缩后 tar.gz
从 3.1 MB 增至 7.2 MB（+4.1 MB/份，30 份滚动共约 220 MB）——远低于 50 MB
分层阈值，维持每日全量方案。

产物：backups/sentinel-YYYYMMDD.tar.gz（项目内 backups/，已 gitignore），
保留最近 BACKUP_KEEP 份滚动删除。每次备份后校验：
- 每个 sqlite 副本 PRAGMA integrity_check 必须 == ok（在打包前查）
- tar 包重新打开可列目录且成员数一致

运行中的 sqlite 库绝不可直接 cp（WAL/页可能撕裂），一律走
`sqlite3.Connection.backup()`——它在页级别拿一致性快照，对写入方无锁阻塞。

用法：
    .venv/bin/python -m intel.backup            # 跑一次备份（launchd 每日 08:30 调用）
    .venv/bin/python -m intel.backup --check backups/sentinel-20260801.tar.gz
                                                # 恢复演练：解包临时目录 + 逐库 integrity_check
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BACKUP_DIR = BASE / "backups"
BACKUP_KEEP = 30  # 滚动保留份数

# (项目内相对路径, 类型)：sqlite=热备份 / file=单文件 / dir=整目录
ITEMS: list[tuple[str, str]] = [
    ("data/intel/sentinel.sqlite", "sqlite"),
    ("data/intel/position.json", "file"),
    ("outputs/shadow_live/journal.sqlite", "sqlite"),
    ("outputs/shadow_e8a/journal.sqlite", "sqlite"),
    ("outputs/n8_scoring", "dir"),
    ("outputs/replay_current/meta.json", "file"),
    # P0-C：不可再生资产（不在 git，/data/ 被 gitignore）
    ("data/intel/musk_tweets.csv", "file"),   # 72,743 条归档，DENSE_QUANTILE 标定源
    ("data/TSLA_5m_3y.csv", "file"),          # 冻结数据源，meta.json 钉其 sha256
]


def _hot_backup_sqlite(src: Path, dst: Path) -> None:
    """sqlite3 backup API 热备份：对运行中的库取页级一致快照."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _integrity_check(db: Path) -> str:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


def _stage(staging: Path) -> tuple[list[str], list[str]]:
    """把所有备份对象按原相对路径复制进 staging，返回 (已备清单, 缺失清单)."""
    staged, missing = [], []
    for rel, kind in ITEMS:
        src = BASE / rel
        dst = staging / rel
        if not src.exists():
            missing.append(rel)
            continue
        if kind == "sqlite":
            _hot_backup_sqlite(src, dst)
            verdict = _integrity_check(dst)
            if verdict != "ok":
                raise RuntimeError(f"integrity_check 失败 {rel}: {verdict}")
        elif kind == "dir":
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        staged.append(rel)
    return staged, missing


def run_backup() -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")  # 本地日期：与 08:30 本地调度同口径
    out = BACKUP_DIR / f"sentinel-{stamp}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="tsla-backup-") as tmp:
        staging = Path(tmp)
        staged, missing = _stage(staging)
        if not staged:
            raise RuntimeError("没有任何备份对象存在——路径全错？")

        n_files = sum(1 for p in staging.rglob("*") if p.is_file())
        tmp_tar = out.with_suffix(".gz.part")
        with tarfile.open(tmp_tar, "w:gz") as tar:
            for rel in staged:
                tar.add(staging / rel, arcname=rel)
        # 校验：tar 可重新打开、文件成员数与 staging 一致
        with tarfile.open(tmp_tar, "r:gz") as tar:
            n_members = sum(1 for m in tar.getmembers() if m.isfile())
        if n_members != n_files:
            tmp_tar.unlink(missing_ok=True)
            raise RuntimeError(f"tar 校验失败：成员 {n_members} != 暂存 {n_files}")
        tmp_tar.replace(out)  # 原子落位，避免半截包顶掉旧包

    size_mb = out.stat().st_size / 1e6
    print(f"[backup] {out.name}  {size_mb:.1f} MB  文件 {n_files} 个")
    for rel in staged:
        print(f"  + {rel}")
    for rel in missing:
        print(f"  ! 缺失跳过 {rel}")

    # 滚动删除：只留最近 BACKUP_KEEP 份（文件名含日期，字典序即时间序）
    olds = sorted(BACKUP_DIR.glob("sentinel-*.tar.gz"))[:-BACKUP_KEEP]
    for p in olds:
        p.unlink()
        print(f"  - 滚动删除 {p.name}")
    return out


def check_archive(archive: Path) -> bool:
    """恢复演练：解包到临时目录，逐 sqlite 副本 integrity_check."""
    ok = True
    with tempfile.TemporaryDirectory(prefix="tsla-restore-") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        root = Path(tmp)
        files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
        print(f"[check] {archive.name} 解包 {len(files)} 个文件：")
        for rel in files:
            p = root / rel
            if p.suffix == ".sqlite":
                verdict = _integrity_check(p)
                flag = "ok" if verdict == "ok" else f"FAIL: {verdict}"
                ok &= verdict == "ok"
                print(f"  {rel}  integrity_check={flag}")
            else:
                print(f"  {rel}  {p.stat().st_size} B")
    print(f"[check] {'全部通过' if ok else '存在损坏'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="前向数据每日备份")
    ap.add_argument("--check", type=Path, metavar="TAR",
                    help="恢复演练：解包指定 tar.gz 并校验，不做新备份")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check_archive(args.check) else 1)
    run_backup()


if __name__ == "__main__":
    main()
