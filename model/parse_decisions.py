import pulp
import logging
import pandas as pd
import numpy as np
from data.demand_generator import DEFAULT_DISCOUNTS_GRID


def parse_discounts(decision_vars, model, article_data):
    if model.status != pulp.const.LpStatusOptimal:
        raise Exception(f"Cannot parse solution of not optimally solved model")

    articles_num = len(article_data)
    weeks_num = article_data[0].demand.shape[1]
    levels_num = article_data[0].demand.shape[0]
    indices = [(i, w) for i in range(articles_num) for w in range(weeks_num)]

    logging.info(f"Status is optimal, parsing solution")

    df = pd.DataFrame(
        [
            (
                i,
                w,
                [
                    d
                    for d in range(levels_num)
                    if decision_vars[i, d, w].varValue > 0.5
                ][0],
            )
            for i in range(articles_num)
            for w in range(weeks_num)
        ],
        columns=["article", "week", "discount"],
    )
    df = df.sort_values(["article", "week"])
    objective = pulp.value(model.objective)
    return df, objective
