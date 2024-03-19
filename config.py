from typing import List

from dataset_metadata import DatasetMetadata
from matchers import match_color, match_counting
from pattern import Pattern

DATASET_LIST: List[DatasetMetadata] = [
    {
        "path": "lmms-lab/ChartQA",
        "name": None,
        "split": "test",
        "column_mapping": {
            "question": "question",
            "answer": "answer",
        },
    },
    {
        "path": "lmms-lab/DocVQA",
        "name": "DocVQA",
        "split": "validation",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "lmms-lab/DocVQA",
        "name": "InfographicVQA",
        "split": "validation",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "lmms-lab/GQA",
        "name": "train_all_instructions",
        "split": "train",
        "column_mapping": {
            "question": "question",
            "fullAnswer": "answer",
        },
    },
    {
        "path": "lmms-lab/llava-bench-coco",
        "name": None,
        "split": "train",
        "column_mapping": {
            "question": "question",
            "answer": "answer",
        },
    },
    {
        "path": "lmms-lab/llava-bench-in-the-wild",
        "name": None,
        "split": "train",
        "column_mapping": {
            "question": "question",
            "gpt_answer": "answer",
        },
    },
    {
        "path": "lmms-lab/MP-DocVQA",
        "name": None,
        "split": "val",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "lmms-lab/OK-VQA",
        "name": None,
        "split": "val2014",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "flaviagiammarino/path-vqa",
        "name": None,
        "split": "train",
        "column_mapping": {
            "question": "question",
            "answer": "answer",
        },
    },
    {
        "path": "lmms-lab/textvqa",
        "name": None,
        "split": "validation",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "ruanchaves/visual7w-gpt",
        "name": None,
        "split": "train",
        "column_mapping": {
            "question": "question",
            "answer": "answer",
        },
    },
    {
        "path": "lmms-lab/VizWiz-VQA",
        "name": None,
        "split": "val",
        "column_mapping": {
            "question": "question",
            "answers": "answer",
        },
    },
    {
        "path": "flaviagiammarino/vqa-rad",
        "name": None,
        "split": "train",
        "column_mapping": {
            "question": "question",
            "answer": "answer",
        },
    },
    {
        "path": "lmms-lab/VQAv2",
        "name": None,
        "split": "validation",
        "column_mapping": {
            "question": "question",
            "multiple_choice_answer": "answer",
        },
    },
]

PATTERN_LIST: List[Pattern] = [
    {
        "name": "color",
        "matcher": match_color,
    },
    {
        "name": "count",
        "matcher": match_counting,
    },
]
