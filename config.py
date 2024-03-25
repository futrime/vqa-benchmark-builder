from typing import List

import datasets

from matchers import *
from pattern import Pattern

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
