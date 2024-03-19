from typing import Dict, Optional, TypedDict


class DatasetMetadata(TypedDict):
    """Metadata for a dataset.

    Attributes:
        path: The path to the dataset.
        name: The name of the dataset.
        split: The split of the dataset.
        column_mapping: A mapping from the names of the columns in the
            dataset to the names of the columns in the output dataset.
    """

    path: str
    name: Optional[str]
    split: Optional[str]
    column_mapping: Dict[str, str]
