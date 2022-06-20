import hypothesis.strategies as st

from data.demand_generator import DEFAULT_DISCOUNTS_GRID, DEFAULT_WEEKS_NUMBER
from hypothesis.extra.numpy import arrays
from data.article_data import ArticleData
import numpy as np


@st.composite
def create_random_article(
    draw,
    grid_size=len(DEFAULT_DISCOUNTS_GRID),
    weeks_number=DEFAULT_WEEKS_NUMBER,
):
    name = draw(st.characters())
    demand = draw(
        arrays(np.float, (grid_size, weeks_number), elements=st.floats(0, 10))
    )
    black_price = draw(st.floats(0, 100.0))
    stock = draw(st.integers(0, 200))
    salvage_value = draw(st.floats(0, black_price))

    return ArticleData(
        black_price=black_price,
        name=name,
        demand=demand,
        stock=stock,
        salvage_value=salvage_value,
    )
