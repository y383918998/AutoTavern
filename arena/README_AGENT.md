# AutoTavern 本地优化交接说明（给后续 AI agent）

## 任务目标

在**本地 TavernGym**中持续改进酒馆战棋策略，并仅在严格的留出种子评估通过后晋升策略。长期目标是降低平均名次；`1.28` 是远期北极星，不是当前验收阈值。当前本地固定对手环境的稳定策略约为 `4.1–4.3` 平均名次，不能与不同视频或论文环境的数字直接比较。

## 强约束

- 只使用 `player_view` 和 `action_mask`；不得读取完整 `state_json`、未来随机数或未公开的对手/商店信息来决策。
- 反事实采集可以从快照分支，但保存的数据只能包含可见特征、合法动作和最终名次；每个候选分支必须使用同一组**重新播种**的续局种子。
- 不提交 API key、`.env`、模型权重、`.npz` 训练数据、日志或游戏存档。`arena/artifacts/` 必须保持本地忽略。
- 不自动启动线上 arena；线上操作须由用户明确指示，并先运行 `--check`。
- 不因单次小样本变好而晋升。所有候选都必须通过独立种子、配对评估。

## 当前稳定策略

入口：`smart_agent_v8:act`。

`smart_agent_v8` 是 `v7` 加“低血量不升级酒馆”约束。最强已批准的局部消融结果：

| 对比 | 局数 | 基线平均名次 | 候选平均名次 | Top-4 | 第八 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v7 → v8（升级约束） | 512 | 4.506 | 4.326 | 257 → 272 | 59 → 49 | 已晋升 |
| v7 → v8（稳定留出集） | 1024 | 4.327 | 4.232 | 558 → 575 | 116 → 101 | 支持 v8，但未达旧脚本 0.15 门槛 |
| v8 → v9 choice 规则 | 512 | 4.131 | 4.156 | 295 → 295 | 42 → 49 | 拒绝 |

运行稳定复验：

```powershell
cd F:\HS_auto\AutoTavern\arena
.\run_stable_validation.ps1 -Games 1024 -Jobs 4
```

## 选择学习：已完成与结论

相关文件：

- `collect_choice_preferences.py`：只采集 Discover、饰品、暗月礼物、choose-one、友方目标等可见选择；使用相同重播种续局。
- `audit_choice_preferences.py`、`merge_choice_preferences.py`：验证协议、模拟器 hash 和特征形状；不得混合协议。
- `train_choice_ranker.py`：紧凑线性 listwise 基线。
- `evaluate_choice_tree_ranker.py`：仅研究用的非线性树基线，**不是 policy loader**。

v2 特征协议为 `choice-reseeded-counterfactual-v2`，61 维：保留 v1 的局面/属性特征，追加稳定卡牌 ID hash、可见效果文本、类型与交互特征。v1（30 维）可审计，但不可与 v2 合并。

最新两批 v2 Discover 数据及跨批结果：

| 批次 | 状态 | 候选 | 续局评估 | 最佳候选均值 |
|---|---:|---:|---:|---:|
| `930250289` | 646 | 1,909 | 11,454 | 3.555 |
| `193702371` | 671 | 2,001 | 12,006 | 3.307 |

- 线性 ranker 双向外部验证未超过随机基线 10 个百分点，拒绝。
- ExtraTrees 双向外部验证相对随机的名次收益为 `-0.001` 与 `+0.034`，低于研究门槛 `+0.10`，拒绝。
- 所有学习模型均未接入 `smart_agent_v8`。

采集一批新版 Discover：

```powershell
$seed = Get-Random -Minimum 1 -Maximum 1800000000
.\run_choice_learning.ps1 -Games 256 -SamplesPerGame 4 -Continuations 6 -Jobs 6 -Families discover -Seed $seed
```

饰品同理：将 `-Families discover` 改为 `-Families trinket`。优先使用两批不同 seed 做外部验证，示例：

```powershell
python train_choice_ranker.py --train artifacts\choice_learning_A\preferences.npz --valid artifacts\choice_learning_B\preferences.npz --out artifacts\local_only_ranker.npz --family discover --seed 123
python evaluate_choice_tree_ranker.py --train artifacts\choice_learning_A\preferences.npz --valid artifacts\choice_learning_B\preferences.npz --family discover --seed 123
```

## 推荐的下一项工作

不要继续只增加相同的 v2 Discover 数据。当前证据表明瓶颈是语义而不是单纯样本量。按此顺序推进：

1. 从公开卡牌目录构建**可见文本机制标签**：召唤、战吼、亡语、复生、经济、酒馆、战斗、持续/下一回合、流派、饰品持续效果。
2. 为 Discover 与饰品分别建模；不得共享未经验证的分数。
3. 建立混合对手的本地评估池，再以固定留出种子做 paired evaluation。
4. 仅当候选同时满足：平均名次改善至少 `0.15`、Top-4 不下降、第八不增加，才更换稳定策略；否则保留 v8 并记录拒绝原因。

## 验证命令

```powershell
python choice_examples.py
python positioning_examples.py
python -m pytest ..\tests -q
```

已知测试基线：`52 passed`。
