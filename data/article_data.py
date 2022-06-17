from dataclasses import dataclass
import numpy as np


@dataclass
class ArticleData:
    name: str
    demand: np.array
    black_price: float
    stock: int
    salvage_value: float
