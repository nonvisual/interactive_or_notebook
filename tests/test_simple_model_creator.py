import imp
from hypothesis import given
from data.article_data import ArticleData
from model.simple_model_creator import create_model
from model.parse_decisions import parse
from data.demand_generator import create_random_demand
import pulp
import pandas as pd
import numpy as np
from tests.utils.instances_generator import create_random_article


def test_simple_model_is_optimal():
    sneakers = ArticleData(
        name="sneakers",
        demand=create_random_demand(max_demand=25.0, elasticity=2.5),
        black_price=65.99,
        stock=100,
        salvage_value=20.5,
    )

    t_shirts = ArticleData(
        name="t-shirt",
        demand=create_random_demand(max_demand=25.0, elasticity=2.5),
        black_price=25.99,
        stock=100,
        salvage_value=10.5,
    )
    model, vars = create_model([sneakers, t_shirts])
    status = model.solve()

    assert pulp.constants.LpStatus[status] == "Optimal"


@given(create_random_article())
def test_is_optimal_and_one_disount_per_decision_is_chosen(article_data):
    model, vars = create_model([article_data])
    status = model.solve()

    df = pd.Series(
        [vars[i, d, w].varValue for (i, d, w) in vars.keys()],
        index=vars.keys(),
        name="solution",
        dtype=np.int64,
    )
    aggregated = df.reset_index().groupby(["level_0", "level_2"]).sum()
    assert pulp.constants.LpStatus[status] == "Optimal"
    assert [(aggregated["solution"] == 1).all()]
