from typing import Callable, Optional, TypedDict

import datasets
import pandas as pd


class DatasetMetadata(TypedDict):
    path: str
    name: Optional[str]
    split: Optional[str]
    converter: Callable[[datasets.Dataset], pd.DataFrame]
