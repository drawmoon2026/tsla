"""E18 全历史滚动校验的引用数字 — 页面文案单一来源.

来源：outputs/e18_rolling/summary.txt 聚合 OOS（2020-07-01 → 2026-07-22，
6.06 年，十二折滚动重训；docs/strategy-lab.md E18 判决 2026-07）。
dashboard.py / playbook.py 所有引用 E18 年化数字的文案统一从本模块取值，
防止散落硬编码在 E18 重跑/修口径后与档案漂移（code_review_0802 P2-7：
"页面主张不超过存档诚实度"）。E18 若重跑，只改这里并核对 summary.txt。
"""

from __future__ import annotations

# 聚合 OOS 年化（A 主口径 gate+S2）
E18_ANN_L1 = -0.0545          # L=1：total −28.80%，年化 −5.45%，MTM MDD −32.27%
E18_ANN_L2 = -0.1096          # L=2（融资 6.5%/yr）：total −50.48%，年化 −10.96%


def _fmt(v: float) -> str:
    """页面展示格式：百分数、负号用 Unicode MINUS（与既有文案一致）。"""
    return f"{v * 100:.2f}%".replace("-", "−")


E18_ANN_L1_TXT = _fmt(E18_ANN_L1)   # "−5.45%"
E18_ANN_L2_TXT = _fmt(E18_ANN_L2)   # "−10.96%"
