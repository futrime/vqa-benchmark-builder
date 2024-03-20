from typing import List

import datasets

from dataset_metadata import DatasetMetadata
from matchers import *
from pattern import Pattern

DATASET_LIST: List[DatasetMetadata] = [
    # {
    #     "path": "lmms-lab/ChartQA",
    #     "name": None,
    #     "split": "test",
    #     "column_renaming": {
    #         "question": "question",
    #         "answer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/DocVQA",
    #     "name": "DocVQA",
    #     "split": "validation",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    # {
    #     "path": "lmms-lab/DocVQA",
    #     "name": "InfographicVQA",
    #     "split": "validation",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    # {
    #     "path": "lmms-lab/GQA",
    #     "name": "train_all_instructions",
    #     "split": "train",
    #     "column_renaming": {
    #         "question": "question",
    #         "fullAnswer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/llava-bench-coco",
    #     "name": None,
    #     "split": "train",
    #     "column_renaming": {
    #         "question": "question",
    #         "answer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/llava-bench-in-the-wild",
    #     "name": None,
    #     "split": "train",
    #     "column_renaming": {
    #         "question": "question",
    #         "gpt_answer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/MP-DocVQA",
    #     "name": None,
    #     "split": "val",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    # {
    #     "path": "lmms-lab/OK-VQA",
    #     "name": None,
    #     "split": "val2014",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    # {
    #     "path": "flaviagiammarino/path-vqa",
    #     "name": None,
    #     "split": "train",
    #     "column_renaming": {
    #         "question": "question",
    #         "answer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/textvqa",
    #     "name": None,
    #     "split": "validation",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    # {
    #     "path": "ruanchaves/visual7w-gpt",
    #     "name": None,
    #     "split": "train",
    #     "column_renaming": {
    #         "question": "question",
    #         "answer": "answer",
    #     },
    #     "column_casting": {},
    # },
    # {
    #     "path": "lmms-lab/VizWiz-VQA",
    #     "name": None,
    #     "split": "val",
    #     "column_renaming": {
    #         "question": "question",
    #         "answers": "answer",
    #     },
    #     "column_casting": {
    #         "answers": datasets.Value("string"),
    #     },
    # },
    {
        "path": "flaviagiammarino/vqa-rad",
        "name": None,
        "split": "train",
        "column_renaming": {
            "question": "question",
            "answer": "answer",
        },
        "column_casting": {},
    },
    # {
    #     "path": "lmms-lab/VQAv2",
    #     "name": None,
    #     "split": "validation",
    #     "column_renaming": {
    #         "question": "question",
    #         "multiple_choice_answer": "answer",
    #     },
    #     "column_casting": {},
    # },
]

PATTERN_LIST: List[Pattern] = [
    {
        "name": "color",
        "matcher": match_color,
    },
    {
        "name": "counting",
        "matcher": match_counting,
    },
    {
        "name": "emotion",
        "matcher": match_emotion,
    },
    {
        "name": "object",
        "matcher": match_object_factory(),
    },
    {
        "name": "shape",
        "matcher": match_shape,
    },
    {
        "name": "spatial relationship",
        "matcher": match_spatial_relationship,
    },
    {
        "name": "texture",
        "matcher": match_texture,
    },
]
