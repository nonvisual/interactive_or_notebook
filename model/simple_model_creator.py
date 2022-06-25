from typing import List
import pulp
import pandas as pd
import numpy as np
from data.article_data import ArticleData
from enum import Enum
from data.demand_generator import DEFAULT_DISCOUNTS_GRID, DEFAULT_WEEKS_NUMBER


class Objective(Enum):
    Profit = 1
    Overstock = 2


VAT = 0.19


def create_model(
    articles: List[ArticleData],
    objective: Objective = Objective.Profit,
    weeks_number=DEFAULT_WEEKS_NUMBER,
):
    # Method which is generating a simple model for inventory optimization

    model = pulp.LpProblem(
        "Profit_maximizing_problem", pulp.constants.LpMaximize
    )
    indices = [
        (i, d, w)
        for i in range(len(articles))
        for d in range(articles[i].demand.shape[0])
        for w in range(weeks_number)
    ]

    # variables
    decision_vars = pulp.LpVariable.dicts(
        "Discount", indices, 0, 1, pulp.LpInteger
    )
    sales_vars = pulp.LpVariable.dicts(
        "Sale", indices, 0, None, pulp.LpContinuous
    )
    stock_vars = pulp.LpVariable.dicts(
        "Stock",
        [(i, w) for i in range(len(articles)) for w in range(weeks_number)],
        0,
        None,
        pulp.LpContinuous,
    )

    m_vars = pulp.LpVariable.dicts(
        "BigMVars",
        [(i, w) for i in range(len(articles)) for w in range(weeks_number)],
        0,
        1,
        pulp.LpInteger,
    )

    # objectives
    overstock = -pulp.lpSum(
        [stock_vars[i, weeks_number - 1] for i, article in enumerate(articles)]
    )

    profit = pulp.lpSum(
        [
            sales_vars[i, d, w]
            * (1 - DEFAULT_DISCOUNTS_GRID[d])
            * articles[i].black_price
            for (i, d, w) in indices
        ]
    ) / (1 + VAT) + pulp.lpSum(
        [
            stock_vars[i, weeks_number - 1] * article.salvage_value
            for i, article in enumerate(articles)
        ]
    )

    # constraints
    # sales are below demand
    for i, article in enumerate(articles):
        for w in range(weeks_number):
            for d, discount in enumerate(DEFAULT_DISCOUNTS_GRID):
                model += (
                    sales_vars[i, d, w]
                    <= decision_vars[i, d, w] * article.demand[d, w]
                )

    # stock flow constraint
    for i, article in enumerate(articles):
        for w in range(1, weeks_number):
            model += stock_vars[i, w] == stock_vars[i, w - 1] - pulp.lpSum(
                [
                    sales_vars[i, d, w]
                    for d in range(len(DEFAULT_DISCOUNTS_GRID))
                ]
            )
    for i, article in enumerate(articles):
        model += stock_vars[i, 0] == article.stock - pulp.lpSum(
            [sales_vars[i, d, 0] for d in range(len(DEFAULT_DISCOUNTS_GRID))]
        )

    # only one discount is chosen
    for i, article in enumerate(articles):
        for w in range(weeks_number):
            model += (
                pulp.lpSum(
                    [
                        decision_vars[i, d, w]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                == 1
            )
    slack = 0.001

    # no stock hedging
    for i, article in enumerate(articles):
        for w in range(0, weeks_number):
            model += (
                stock_vars[i, w - 1]
                if w > 0
                else (
                    article.stock
                    - pulp.lpSum(
                        [
                            sales_vars[i, d, w]
                            for d in range(len(DEFAULT_DISCOUNTS_GRID))
                        ]
                    )
                )
                <= 1.1 * article.stock * m_vars[i, w] + slack
            )
            model += (
                pulp.lpSum(
                    [
                        decision_vars[i, d, w] * article.demand[d, w]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                - pulp.lpSum(
                    [
                        sales_vars[i, d, w]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                <= 1.1
                * max(
                    [
                        article.demand[d, w]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                * (1 - m_vars[i, w])
                + slack
            )

    # Set objective
    model += profit if objective == Objective.Profit else overstock

    return model, decision_vars, stock_vars
