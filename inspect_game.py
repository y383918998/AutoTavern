import json
import taverngym

game = taverngym.Game(seed=1)

# 0 号玩家当前能看到的游戏状态
view = json.loads(game.view(0))
print(json.dumps(view, ensure_ascii=False, indent=2))

# 所有动作的总数量
print("动作数量：", taverngym.N_ACTIONS)

# 打印前 20 个动作名称
for action in range(20):
    print(action, taverngym.action_name(action))