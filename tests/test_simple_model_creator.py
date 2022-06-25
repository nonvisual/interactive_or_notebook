import imp
from hypothesis import Phase, given, reproduce_failure, settings
from data.article_data import ArticleData
from model.simple_model_creator import create_model
from model.parse_decisions import parse_discounts
from data.demand_generator import create_random_demand
from evaluation.kpi_calculator import calculate
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
    model, discount_vars, stock_vars = create_model([sneakers, t_shirts])
    status = model.solve()

    assert pulp.constants.LpStatus[status] == "Optimal"


@settings(print_blob=True)
@given(create_random_article())
def test_is_optimal_and_one_discount_per_decision_is_chosen(article_data):
    model, discount_vars, stock_vars = create_model([article_data])
    gap = 0.005
    status = model.solve(pulp.PULP_CBC_CMD(fracGap=gap))

    df = pd.Series(
        [
            discount_vars[i, d, w].varValue
            for (i, d, w) in discount_vars.keys()
        ],
        index=discount_vars.keys(),
        name="solution",
        dtype=np.int64,
    )
    aggregated = df.reset_index().groupby(["level_0", "level_2"]).sum()

    solution, _ = parse_discounts(discount_vars, model, [article_data])
    _, _, weekly_profit, left_value = calculate(
        solution["discount"].values, article_data
    )
    assert pulp.constants.LpStatus[status] == "Optimal"
    assert [(aggregated["solution"] == 1).all()]
    assert all(
        i >= -0.0 for i in weekly_profit
    ), "Profit should be non-negative"


@settings(
    deadline=700,
    print_blob=True,
    max_examples=50,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
)
@given(create_random_article())
def test_profit_is_bigger_then_simple_baseline(article_data):
    model, discount_vars, stock_vars = create_model([article_data])
    gap = 0.05
    status = model.solve(pulp.PULP_CBC_CMD(fracGap=gap))

    df = pd.Series(
        [
            int(discount_vars[i, d, w].varValue)
            for (i, d, w) in discount_vars.keys()
        ],
        index=discount_vars.keys(),
        name="solution",
        dtype=np.int64,
    )

    stock_df = pd.Series(
        [stock_vars[i, w].varValue for (i, w) in stock_vars.keys()],
        index=stock_vars.keys(),
        name="stock",
        dtype=float,
    )
    aggregated = df.reset_index().groupby(["level_0", "level_2"]).sum()

    solution, objective = parse_discounts(discount_vars, model, [article_data])
    discounts = solution["discount"].values
    sales, weekly_stock, weekly_profit, left_value = calculate(
        discounts, article_data
    )
    (
        zero_discount_sales,
        _,
        zero_discount_weekly_profit,
        zero_discount_left_value,
    ) = calculate([0] * len(discounts), article_data)

    assert pulp.constants.LpStatus[status] == "Optimal"
    assert sum(weekly_profit) + left_value >= (
        sum(zero_discount_weekly_profit) + zero_discount_left_value
    ) * (
        1 - gap
    ), "Solution objective is below simple 0-discount baseline (accounting for gap)"


def test_black_price_scenario_is_optimal():
    sneakers = ArticleData(
        name="sneakers",
        demand=create_random_demand(max_demand=10.0, elasticity=6.5),
        black_price=65.99,
        stock=145,
        salvage_value=65.99,
    )
    model, discount_vars, stock_vars = create_model([sneakers])
    stock = ([stock_vars[i, w].varValue for (i, w) in stock_vars.keys()],)

    status = model.solve(pulp.PULP_CBC_CMD(fracGap=0.001))
    solution, objective = parse_discounts(discount_vars, model, [sneakers])
    discounts = solution["discount"].values
    sales, weekly_stock, weekly_profit, left_value = calculate(
        discounts, sneakers
    )
    (
        zero_discount_sales,
        _,
        zero_discount_weekly_profit,
        zero_discount_left_value,
    ) = calculate([0] * len(discounts), sneakers)

    assert pulp.constants.LpStatus[status] == "Optimal"
    assert sum(weekly_profit) + left_value == (
        sum(zero_discount_weekly_profit) + zero_discount_left_value
    )
