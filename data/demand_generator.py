from email.mime import base
import numpy as np
import itertools

DEFAULT_DISCOUNTS_GRID = np.arange(0, 0.65, step=0.05)
DEFAULT_WEEKS_NUMBER = 30


def create_random_demand(
    discount_grid: np.array = DEFAULT_DISCOUNTS_GRID,
    max_demand: float = 1.0,
    elasticity=1.2,
    seed: int = 12345,
) -> np.array:
    # introduce elasticity concept = now it is entirely random
    temp_demand = base_demand = (
        np.random.rand(DEFAULT_WEEKS_NUMBER) * max_demand
    )
    demand_result = base_demand
    for i in range(1, len(DEFAULT_DISCOUNTS_GRID)):
        delta_price = DEFAULT_DISCOUNTS_GRID[i] - DEFAULT_DISCOUNTS_GRID[i - 1]
        temp_demand = temp_demand / (1 - elasticity * delta_price)
        demand_result = np.vstack([demand_result, temp_demand])
    return demand_result
