from envs import build_beer_game


def test_env(order_quantities):
    states = bg.reset()
    total_r = 0
    for i in range(0, 20):
        states, reward, terminal = bg.step(order_quantities[i])
        total_r += reward
    return total_r


bg = build_beer_game(player='manufacturer')
assert test_env([0, 0] + [8] * 30) == -224.
assert test_env([4] * 30) == -644.
assert test_env([12] * 30) == -658.

bg = build_beer_game(player='distributor')
assert test_env([0, 0] + [8] * 30) == -170.
assert test_env([4] * 30) == -784.
assert test_env([12] * 30) == -644.

bg = build_beer_game(player='wholesaler')
assert test_env([0, 0] + [8] * 30) == -158.0
assert test_env([4] * 30) == -864.0
assert test_env([12] * 30) == -888.0

bg = build_beer_game(player='retailer')
assert test_env([0, 0] * 30) == -1656.0
assert test_env([4] * 30) == -798.0
assert test_env([12] * 30) == -1260.0
