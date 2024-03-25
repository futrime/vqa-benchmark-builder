import datasets
import pandas as pd


def convert_simple_dataset(dataset: datasets.Dataset) -> pd.DataFrame:
    dataframe = dataset.select_columns(["question", "answer"]).to_pandas()
    assert isinstance(dataframe, pd.DataFrame)

    return dataframe


def convert_multiple_answers_dataset(dataset: datasets.Dataset) -> pd.DataFrame:
    dataframe = dataset.select_columns(["question", "answers"]).to_pandas()
    assert isinstance(dataframe, pd.DataFrame)

    return dataframe
