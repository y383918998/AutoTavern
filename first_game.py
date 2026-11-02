import taverngym

game = taverngym.Game(seed=1)

while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            action = game.bot_action(seat)
            game.step(seat, action)

print("最终名次：", list(game.placements()))