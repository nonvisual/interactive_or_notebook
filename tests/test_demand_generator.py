from data.demand_generator import (
    DEFAULT_DISCOUNTS_GRID,
    DEFAULT_WEEKS_NUMBER,
    create_random_demand,
)
import numpy as np


def test_create_random_demand():

    demand = create_random_demand(
        discount_grid=DEFAULT_DISCOUNTS_GRID, max_demand=5.0, seed=12345
    )
    assert demand.shape == (
        len(DEFAULT_DISCOUNTS_GRID),
        DEFAULT_WEEKS_NUMBER,
    ), "Demand should have correct shape"
    assert np.all(demand > 0.0), "Demand should be positive"
