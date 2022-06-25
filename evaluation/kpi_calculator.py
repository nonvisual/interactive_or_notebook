import pandas as pd
import numpy as np

from data.article_data import ArticleData
from data.demand_generator import DEFAULT_DISCOUNTS_GRID
from model.simple_model_creator import VAT


def calculate(discounts: np.array, article_data: ArticleData):
    assert set(np.unique(discounts)) <= set(
        range(article_data.demand.shape[0])
    ), "unknown discount level, is not on grid"

    choices = [(discounts[i], i) for i in range(len(discounts))]
    selected_demand = [article_data.demand[i] for i in choices]
    weekly_sales = []
    weekly_stock = []
    weekly_profit = []
    current_stock = article_data.stock
    for i in range(len(selected_demand)):
        weekly_sales.append(min(current_stock, selected_demand[i]))
        current_stock = max(0, current_stock - weekly_sales[-1])
        weekly_stock.append(current_stock)
        weekly_profit.append(
            article_data.black_price
            * weekly_sales[-1]
            * (1 - DEFAULT_DISCOUNTS_GRID[choices[i][0]])
            / (1 + VAT)
        )
    left_value = weekly_stock[-1] * article_data.salvage_value
    return weekly_sales, weekly_stock, weekly_profit, left_value
