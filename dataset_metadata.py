from typing import Dict, Optional, TypedDict

import datasets


class DatasetMetadata(TypedDict):
    """Metadata for a dataset.

    Attributes:
        path: The path to the dataset.
        name: The name of the dataset.
        split: The split of the dataset.
        column_renaming: A mapping from the names of the columns in the
            dataset to the names of the columns in the output dataset.
        column_casting: A mapping from the names of the columns in the
            dataset to the types of the columns in the output dataset.
            The key should be the name in the original dataset.
    """

    path: str
    name: Optional[str]
    split: Optional[str]
    column_renaming: Dict[str, str]
    column_casting: Dict[str, datasets.features.features.FeatureType]
