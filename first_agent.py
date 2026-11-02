import taverngym

env = taverngym.BattlegroundsEnv(seed=0)
obs, info = env.reset()

for turn in range(1000):
    legal_actions = obs["action_mask"].nonzero()[0]

    # 优先从所有合法动作中寻找“购买”动作
    chosen_action = None

    for action in legal_actions:
        name = taverngym.action_name(int(action))

        if "buy" in name.lower():
            chosen_action = int(action)
            break

    # 如果目前没有可购买的随从，就执行第一个合法动作
    if chosen_action is None:
        chosen_action = int(legal_actions[0])

    print(
        f"第 {turn} 次决策，选择：",
        taverngym.action_name(chosen_action),
    )

    obs, reward, terminated, truncated, info = env.step(chosen_action)

    if terminated or truncated:
        print("对局结束，奖励：", reward)
        break