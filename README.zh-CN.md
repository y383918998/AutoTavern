# AutoTavern 中文说明

AutoTavern 是面向《炉石传说：酒馆战棋》的 AI 研究代码与数据发布。它不连接暴雪服务，也不包含游戏客户端文件；核心目标是让研究者能够复现规则、运行完整八人对局、训练决策策略，并用固定题集评测模型的判断能力。

> 原始英文说明：[README.md](README.md)。本文件是中文上手指南，不替代项目的原始文档。

## 项目包含什么

| 组件 | 用途 |
| --- | --- |
| **TavernGym** | 由 Rust 实现、通过 Python 调用的确定性八人酒馆战棋模拟器。可控制单个玩家，或批量运行多局游戏来训练强化学习/自博弈策略。 |
| **TavernBench** | 960 道决策与规则推理题（其中 368 道为论文评测子集），含题目、标签、部分可恢复的游戏状态、答案和评分代码。 |

当前实现对应客户端 **36.6.3.253932**、服务器规则 **36.6.3**。仓库发布的是模拟器与基准；训练好的 TavernAgent 权重、训练代码和题目生成器没有包含在内。

## 环境要求

- Python **3.10+**
- Rust 工具链；仓库通过 `rust-toolchain.toml` 固定到 **1.98.1**，在仓库目录执行 Rust 命令时 rustup 会自动安装/启用它。
- 若只想给已发布的 TavernBench 答案打分，只需 Python 标准库，无需编译模拟器、模型访问权限或 API Key。

## 安装

在仓库根目录执行：

```powershell
python -m pip install .
```

用于本地开发与测试：

```powershell
python -m pip install -e ".[test]"
pytest
```

若要运行 TavernBench 中调用 OpenAI 兼容接口的模型脚本，安装可选依赖：

```powershell
python -m pip install ".[bench]"
```

## 先跑通模拟器

新建任意 Python 文件并运行下面示例。它让八个内置脚本机器人完成一局对战：

```python
import taverngym

game = taverngym.Game(seed=1)
while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            game.step(seat, game.bot_action(seat))

print(list(game.placements()))
```

如果要让自己的算法控制 0 号玩家，则改用 `BattlegroundsEnv`。观察值中含 `action_mask`，只能从掩码标记为合法的 245 个离散动作中选择：

```python
import taverngym

env = taverngym.BattlegroundsEnv(seed=0)
obs, info = env.reset()
action = int(obs["action_mask"].nonzero()[0][0])  # 示例：选第一个合法动作
obs, reward, terminated, truncated, info = env.step(action)
```

训练多个并行对局或自博弈时使用 `BatchEnv`；详见 [TavernGym 文档](docs/TAVERNGYM.md)。

## 先跑通基准评分

下列命令会对仓库随附的已发布答案进行评分，不调用任何外部模型：

```powershell
cd tavernbench
python -m tavernbench.report --set data/eval --runs results/eval `
  --systems tavernagent,gpt-6-astra,claude-opus-5.5,jev
```

要评测自己的模型，先使用 `tavernbench.run_openai`（或其他 runner）生成每题一行的 `answers.jsonl`，再执行：

```powershell
python -m tavernbench.report --set data/eval --runs answers --systems YOUR_MODEL
```

完整格式、其他 runner 和工具调用评测方式见 [TavernBench 文档](docs/TAVERNBENCH.md)。

## 你接下来可以做什么

根据目标选择一条路径：

1. **熟悉 API / 做规则实验**：先运行上面的完整对局示例；用 `Game.view(0)` 查看玩家可见状态，用 `state_json()` 和 `Game.from_json()` 保存、恢复分支状态。
2. **训练策略**：从 `BattlegroundsEnv` 开始做“一名学习者对七个内置机器人”，稳定后用 `BatchEnv` 增加并行局数，或以 `controlled=range(8)` 做自博弈。观察为 3,227 个浮点特征和 124 个卡牌索引；动作空间为 245 个动作。
3. **评测语言模型**：先对 `tavernbench/results/eval` 的已发布答案跑评分，熟悉输出；再生成自己的 `answers.jsonl`，针对 368 题 `data/eval` 集合复现实验。模型运行器需要 `.[bench]` 可选依赖与对应 API 配置。
4. **验证或改动引擎**：运行 `pytest`，并运行 `cargo test --release`。规则配置集中在 `crates/taverngym/src/config.rs`；卡牌覆盖范围见 [CONTENT.md](docs/CONTENT.md)。

## 重要目录

```text
crates/taverngym/       Rust 模拟器引擎
python/taverngym/       Python 包与绑定接口
tests/                  Python 测试
tavernbench/data/       题目、标签和状态
tavernbench/results/    发布的模型答案
tavernbench/tavernbench/评分与模型运行器
docs/                   模拟器、基准、架构和内容覆盖文档
```

## 实用文档索引

- [TavernGym：接口、观察、动作、批处理与配置](docs/TAVERNGYM.md)
- [TavernBench：数据格式、评分与模型运行](docs/TAVERNBENCH.md)
- [架构说明](docs/ARCHITECTURE.md)
- [已实现的卡牌与内容](docs/CONTENT.md)
- [Timewarp 规则](docs/TIMEWARP.md)

## 许可

代码、文档以及 TavernBench 的任务、标签、状态和结果采用 [MIT License](LICENSE)。第三方游戏内容保留其原有权利，详见 [NOTICE](NOTICE)。
