# 2025–2026 年 AI 交易算法研究综述

> 检索日期：2026-07-23，检索者：Claude Fable 5（代理 C）
> 面向个人散户级 TSLA 5 分钟 V 形反转统计/回测项目

---

## 一、LLM 驱动的交易代理

### 1.1 主要框架与进展

**TradingAgents**（Tauric Research，[arXiv:2412.20138](https://arxiv.org/abs/2412.20138)，[GitHub](https://github.com/tauricresearch/tradingagents)）是当前最受关注的多代理 LLM 交易框架：模拟交易公司组织结构，设基本面/情绪/技术分析师、多空研究员辩论、风控团队等角色。GitHub API 实测约 9.4 万 star，2026-07 仍在活跃发版（v0.3.1 修复了 Alpha Vantage 前视过滤等正确性问题——官方自己承认早期版本存在前视泄漏，这点值得注意）。论文宣称在累积收益、Sharpe、最大回撤上优于 Buy-&-Hold、MACD 等基线，**但回测窗口短、标的少，属于「论文实验」而非经审计的实盘记录**。另有中文增强版 [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) 支持 A 股。

**FinMem**（arXiv:2311.13743，分层记忆 + 角色设定的 LLM 交易代理）和 **FinGPT**（AI4Finance，[GitHub](https://github.com/AI4Finance-Foundation/FinGPT)，约 2.1 万 star，2026-07 活跃）延续发展；2025 年新增 **ContestTrade**（内部竞争机制多代理，[arXiv:2508.00554](https://arxiv.org/pdf/2508.00554)）、**FinVision**（[arXiv:2411.08899](https://arxiv.org/pdf/2411.08899)）等。**LiveTradeBench**（[arXiv:2511.03628](https://arxiv.org/pdf/2511.03628)）开始做 LLM 实时（非回测）交易评测，这是对回测污染问题的直接回应。

### 1.2 批评性证据（重要）

- **系统综述《Agentic Trading: When LLM Agents Meet Financial Markets》**（[arXiv:2605.19337](https://arxiv.org/pdf/2605.19337)，2026-05）审查 77 项研究：仅 19 项满足基本评估标准；其中**仅 2 项报告了时间一致的训练/测试划分，仅 1 项记录了交易成本模型，15/19 可复现性为最低级 R0，无一项研究控制生存者偏差**。结论：该领域架构创新远快于评估规范。
- **参数化前视偏差**：LLM 预训练语料本身编码了历史行情与新闻，即使管道代码只喂当期数据，模型也可能「记得」2023 年 TSLA 发生了什么——这种泄漏无法靠检查代码排除（[arXiv:2605.24564](https://arxiv.org/html/2605.24564)；[arXiv:2607.04958](https://arxiv.org/pdf/2607.04958)；[Papers with Backtest 课程](https://paperswithbacktest.com/course/look-ahead-bias-llm-trading)）。唯一干净的解法是用「训练截止于回测起点之前」的模型，成本极高。
- 结论：**LLM 代理回测中报告的超额收益，目前应默认视为「未经验证的宣传性内容」**，除非是 LiveTradeBench 这类前向实时评测。

## 二、深度强化学习（DRL）交易

- 2025 年仍有大量 PPO/SAC 论文：PPO 期权交易宣称「85% 盈利交易预测准确率」（[SAGE 2025](https://journals.sagepub.com/doi/10.1177/15741702251398696)）、多模态 PPO 组合优化年化 16.24%、Sharpe 0.86（[Opast 2025](https://www.opastpublishers.com/open-access-articles/portfolio-optimization-through-a-multimodal-deep-reinforcement-learning-framework-8984.html)）、Cluster Embedding-PPO 在五大指数上跑赢（[MDPI Symmetry 2026](https://www.mdpi.com/2073-8994/18/1/112)）。**注意：不同论文里 PPO/SAC/DDPG 谁更强结论互相矛盾**（如 [ACM 2022 对比研究](https://dl.acm.org/doi/10.1145/3529836.3529857)中 PPO 反而跑输基准），这本身就是「结果对超参数与回测窗口高度敏感」的证据。
- 公认难点：日内即时奖励设计易导致过拟合、高方差；市场非平稳使离线训练的策略在 regime 切换后失效（[知乎：强化学习在量化交易的应用](https://zhuanlan.zhihu.com/p/30621322993)；[DeepScalper, arXiv:2201.09058](https://arxiv.org/pdf/2201.09058)）。
- 工程生态：**FinRL**（[GitHub](https://github.com/AI4Finance-Foundation/FinRL)，约 1.6 万 star，2026-07 活跃）及 FinRL Contest 2025（[官网](https://open-finance-lab.github.io/FinRL_Contest_2025/)）是最成熟的入口，2025 赛题已转向 RL+LLM 混合（FinRL-DeepSeek）与因子工程+集成（FinRL-AlphaSeek）——后者其实说明纯 RL 端到端效果有限，实践中在退回「RL 只做仓位/执行层」。
- 定性判断：**DRL 在学术基准上「有实证」，但几乎没有可信的散户级日内实盘证据**；85% 准确率一类数字无成本模型、无样本外协议，按上文综述标准应存疑。

## 三、时间序列基础模型（TSFM）

- **对金融预测的负面实证最扎实**：Rahimikia et al. (2025) 在日频超额收益预测上发现 **Chronos、TimesFM 系零样本模型稳定跑输 CatBoost/LightGBM 集成**（转引自 [Jonathan Kinlay 综述](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)）；Goel et al. 在 21 个全球股指的已实现波动率预测上发现零样本 TimesFM 2.0 **打不过 20 年前的 HAR 计量模型**，须微调才勉强可比（[arXiv:2607.05291](https://arxiv.org/html/2607.05291v1)）。
- **基准污染争议**：多项研究指出主流 TSFM 的测试集与预训练语料重叠，精度虚高 47%–184%，是该领域最大争议（[知乎：时序大模型到底值不值得做](https://zhuanlan.zhihu.com/p/2042269956985835832)）。Lag-Llama 在 2025–2026 年的九模型横评中也表现平平（[arXiv:2604.16428](https://arxiv.org/pdf/2604.16428) 等）。
- **金融专用 TSFM**：**Kronos**（清华，[arXiv:2508.02739](https://arxiv.org/abs/2508.02739)，NeurIPS 2025，[GitHub](https://github.com/shiyu-coder/Kronos) 约 3.3 万 star）在 45 个交易所 120 亿根 K 线上预训练，宣称价格预测 RankIC 比最好 TSFM 高 93%。这是「通用 TSFM 不行、领域专用预训练可能行」的代表，但 RankIC 提升 ≠ 扣费后可交易收益，独立复现还很少，且 2026-04 后仓库更新放缓。开放权重：[HuggingFace](https://huggingface.co/NeoQuasar/Kronos-base)。
- 定性判断：「零样本 TSFM 预测股价」目前**证据偏向否定**；有实证支撑的结论反而是：**梯度提升树仍是收益预测的更强基线**。

## 四、传统机器学习（GBDT / 在线学习）在日内策略中的实践

- 从业者实践：对高流动性股票做 5–30 分钟持仓的均值回归，用 GBDT 学习「动量尖峰后的短期反转」，特征为多时间尺度动量（5/15/60 分钟）与量比（[Medium: Boosting Your Trading Strategy](https://medium.com/@conniezhou678/machine-learning-for-algorithm-trading-part-12-boosting-your-trading-strategy-why-gradient-2d185b57e8f0)；教科书级参考：[Stefan Jansen, ML4T 第12章](https://stefan-jansen.github.io/machine-learning-for-trading/12_gradient_boosting_machines/)）。这与本项目的 V 形反转场景几乎同构。
- 冷静的实证：MNQ 期货日内数据上 LSTM vs GBDT 的 walk-forward 研究（[arXiv:2605.17724](https://arxiv.org/pdf/2605.17724)）：准确率仅 54.8%、精确率 59.8%，且**各训练窗口的头部特征不断漂移，说明模型并未学到稳定结构，而是在逐期拟合当期最灵的特征**。这是对日内 ML 最诚实的定量描述：edge 存在但薄、且不稳定。
- 在线/增量学习：针对 regime 切换的增量更新（流式重训、HMM 状态检测 + 分状态模型）是 2025 年的务实方向（[arXiv:2303.07925](https://arxiv.org/pdf/2303.07925)；[QuantInsti: HMM+RF regime trading](https://blog.quantinsti.com/regime-adaptive-trading-python/)；[ScienceDirect: 增量 RL+自监督](https://www.sciencedirect.com/science/article/abs/pii/S0957417425019165)）。
- 定性判断：这一方向**实证密度最高、宣传水分最少**，业界（含 FinRL 竞赛头部方案）实际都在用 GBDT + 因子工程。

## 五、开源项目盘点（星数为 2026-07 GitHub API / 检索实测）

| 项目 | 星数 | 活跃度 | 定位与成熟度 |
|---|---|---|---|
| [TradingAgents](https://github.com/tauricresearch/tradingagents) | ~94.2k | 2026-07 活跃 | 多代理 LLM，热度极高但偏研究 demo，自身修过前视 bug |
| [Qlib](https://github.com/microsoft/qlib)（微软） | ~46.5k | 2026-04 | 最成熟的 AI 量化平台（监督/RL/自动化研发），偏日频因子研究 |
| [Kronos](https://github.com/shiyu-coder/Kronos) | ~32.8k | 2026-04 | 金融 K 线基础模型，研究向，可下载权重 |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | ~20.9k | 2026-07 | 金融 LLM（情绪/新闻），非交易执行框架 |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | ~15.8k | 2026-07 | DRL 交易教学/竞赛生态，工程质量一般但社区大 |
| Qbot / Abu / [ai_quant_trade](https://github.com/charliedream1/ai_quant_trade) / VeighNa | 6k–17k | 活跃 | 偏 A 股/教学聚合（[盘点来源](https://zhuanlan.zhihu.com/p/1951584062096536991)、[头条 2026 盘点](https://www.toutiao.com/article/7624559216245604910/)） |

注意：**star 数与实盘可用性基本无关**——TradingAgents 星数是 Qlib 两倍，但工程可靠性远低于后者。

---

## 六、对本项目（TSLA 5 分钟 V 形反转、纯规则、散户规模）的适用性评估

### 值得引入（性价比从高到低）

1. **GBDT 过滤器（LightGBM/XGBoost）——最值得做**。不改变现有规则框架：仍由阈值触发 V 形信号，但用 GBDT 对每次触发输出「反弹成功概率」，只交易高分信号。特征就用已有的统计量（急跌幅度/速度、量比、时段、隔夜 gap、VIX 等）。成本：单机 CPU、零 API 费用、几天工作量；这正是文献中实证最扎实的用法（ML4T 第12章、MNQ 研究）。**必须配 purged walk-forward 验证**，否则只是把网格扫描的过拟合换成模型过拟合。
2. **Regime 感知（HMM 或简单波动率分档）**。把「高波动/低波动、趋势/震荡」作为开关或参数切换条件，对抗网格参数在 regime 切换后失效的问题。门槛低（hmmlearn 即可），有成熟教程（QuantInsti）。
3. **在线/滚动重估参数**。用滚动窗口定期重估阈值与止盈止损，替代一次性全样本网格扫描——这是把「在线学习」思想以最低成本落地，也顺便暴露当前回测里参数全样本调优的隐性前视。
4. **借用 Qlib 的验证基建**（而非其全套平台）：它的时序交叉验证、成本建模思路值得抄，即使不迁移代码。

### 谨慎试验（可玩，不要指望 alpha）

- **Kronos 微调做 5 分钟短线预测**：它是唯一针对 K 线的开放权重基础模型，消费级 GPU 可跑 small/base 版。但 RankIC 提升未被独立验证为扣费后收益，建议只作为 GBDT 的对照基线。
- **LLM 做新闻/情绪特征**（FinGPT 式）：TSLA 是新闻驱动型标的，V 形急跌常由消息触发，用 LLM 给「下跌原因是否为一次性情绪冲击」打标作为一个特征，逻辑上说得通；但注意 API 成本、以及历史回测中的参数化前视偏差（模型「记得」马斯克哪条推特）——**历史段打标结果不可信，只能前向使用**。

### 不值得跟（对本项目而言是炒作/负性价比）

- **多代理 LLM 交易框架（TradingAgents 及其仿品）**：评估综述显示该领域 19 项合格研究中仅 2 项有干净时序划分、几乎无成本模型（arXiv:2605.19337）；其决策频率（日级、叙事驱动）与 5 分钟统计套利完全不匹配，每次决策数美分到数美元的 API 成本会吃掉散户级日内 edge。
- **端到端 DRL（PPO/SAC 直接学买卖）**：结论对超参敏感、论文间互相矛盾、无可信散户实盘证据；本项目的规则策略本质是一个可解释的低维策略，用 DRL 重学它只会增加过拟合面。
- **零样本通用 TSFM（TimesFM/Chronos/Lag-Llama）直接预测价格**：多项独立实证显示跑不过 GBDT 甚至 HAR，且基准污染争议未决。
- **任何宣称高胜率/高年化的论文数字**（如「85% 准确率」）：按本综述第一节的可复现性统计，默认不可信，除非提供成本模型 + 时间一致划分 + 前向验证。

### 一句话结论

2025–2026 年真正的共识性发现是反直觉的：**在收益预测上，梯度提升树 + 严格的 walk-forward 验证仍然打败绝大多数「大模型」方案**；LLM 的可靠价值在信息处理（新闻/情绪特征）而非决策本身。对本项目，最优路径是「规则触发 + ML 信号过滤 + regime 开关 + 滚动重估」，而不是推倒重来上代理或 RL。
