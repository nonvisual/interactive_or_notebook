import pulp
import logging
import pandas as pd
import numpy as np
from data.demand_generator import DEFAULT_DISCOUNTS_GRID


def parse_discounts(decision_vars, model):
    if model.status != pulp.const.LpStatusOptimal:
        raise Exception(f"Cannot parse solution of not optimally solved model")
    indices = set([(i, w) for (i, d, w) in decision_vars.keys() if d == 0])

    logging.info(f"Status is optimal, parsing solution")
    df = pd.Series(
        [
            DEFAULT_DISCOUNTS_GRID[d]
            for (i, d, w) in decision_vars.keys()
            if decision_vars[i, d, w].varValue > 0.5
        ],
        index=indices,
        name="solution",
        dtype=np.float,
    )
    objective = pulp.value(model.objective)
    return df, objective
