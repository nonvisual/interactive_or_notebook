import pandas as pd
import numpy as np

from data.article_data import ArticleData


def calculate(discounts: np.array, article_data: ArticleData):
    assert set(np.unique(discounts)) <= set(
        range(article_data.demand.shape[0])
    ), "unknown discount level, is not on grid"

    choices = [(discounts[i], i) for i in range(len(discounts))]
    selected_demand = [article_data.demand[i] for i in choices]
    weekly_sales = []
    weekly_stock = []
    current_stock = article_data.stock
    for i in range(len(selected_demand)):
        weekly_sales.append(min(current_stock, selected_demand[i]))
        current_stock = max(0, current_stock - weekly_sales[-1])
        weekly_stock.append(current_stock)

    return weekly_sales, weekly_stock
