from typing import List
import pulp
import pandas as pd
import numpy as np
from data.article_data import ArticleData
from enum import Enum
from data.demand_generator import DEFAULT_DISCOUNTS_GRID, DEFAULT_WEEKS_NUMBER
from model.simple_model_creator import Objective, create_model


def create_model_with_steps(
    articles: List[ArticleData],
    objective: Objective = Objective.Profit,
    weeks_number=DEFAULT_WEEKS_NUMBER,
):
    model, decision_vars, stock_vars = create_model(
        articles, objective, weeks_number
    )
    for i, article in enumerate(articles):
        for w in range(1, weeks_number):
            model += (
                pulp.lpSum(
                    [
                        decision_vars[i, d, w] * DEFAULT_DISCOUNTS_GRID[d]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                - pulp.lpSum(
                    [
                        decision_vars[i, d, w - 1] * DEFAULT_DISCOUNTS_GRID[d]
                        for d in range(len(DEFAULT_DISCOUNTS_GRID))
                    ]
                )
                >= -0.01
            )
    return model, decision_vars, stock_vars
