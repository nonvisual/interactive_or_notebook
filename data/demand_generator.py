import numpy as np
import itertools

DEFAULT_DISCOUNTS_GRID = np.arange(0, 0.65, step=0.05)
DEFAULT_WEEKS_NUMBER = 30


def create_random_demand(
    discount_grid=DEFAULT_DISCOUNTS_GRID, max_demand=1.0, seed=12345
):
    return (
        np.random.rand(len(DEFAULT_DISCOUNTS_GRID), DEFAULT_WEEKS_NUMBER)
        * max_demand
    )
