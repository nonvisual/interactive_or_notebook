from dataclasses import dataclass
import pandas as pd


@dataclass
class ArticleData:
    name: str
    demand: pd.DataFrame
    black_price: float
    stock: int
    salvage_value: float
