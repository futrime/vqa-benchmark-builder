from typing import Callable

import datasets
import pandas as pd


def make_convert_multiple_answers_dataset(
    question_column: str = "question", answer_column: str = "answers"
) -> Callable[[datasets.Dataset], pd.DataFrame]:
    def convert_multiple_answers_dataset(dataset: datasets.Dataset) -> pd.DataFrame:
        dataframe = dataset.select_columns([question_column, answer_column]).to_pandas()
        assert isinstance(dataframe, pd.DataFrame)

        # Convert list of answers to string
        dataframe[answer_column] = dataframe[answer_column].apply(
            lambda x: "; ".join(x)
        )

        # Rename columns for consistency
        dataframe.rename(
            columns={question_column: "question", answer_column: "answer"},
            inplace=True,
        )

        return dataframe

    return convert_multiple_answers_dataset


def make_convert_simple_dataset(
    question_column: str = "question", answer_column: str = "answer"
) -> Callable[[datasets.Dataset], pd.DataFrame]:
    def convert_simple_dataset(dataset: datasets.Dataset) -> pd.DataFrame:
        dataframe = dataset.select_columns([question_column, answer_column]).to_pandas()
        assert isinstance(dataframe, pd.DataFrame)

        # Rename columns for consistency
        dataframe.rename(
            columns={question_column: "question", answer_column: "answer"},
            inplace=True,
        )

        return dataframe

    return convert_simple_dataset
