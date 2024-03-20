import re
from typing import Callable

import requests


def match_color(question: str, answer: str) -> bool:
    question_regex_list = [
        r"color",
        r"colour",
    ]
    answer_regex_list = [
        r"green",
        r"blue",
        r"red",
        r"yellow",
        r"orange",
        r"purple",
        r"pink",
        r"brown",
        r"black",
        r"white",
        r"gray",
        r"grey",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_counting(question: str, answer: str) -> bool:
    question_regex_list = [
        r"how many",
        r"number",
    ]
    answer_regex_list = [
        r"\d+",
        r"zero",
        r"one",
        r"two",
        r"three",
        r"four",
        r"five",
        r"six",
        r"seven",
        r"eight",
        r"nine",
        r"ten",
        r"[a-z]+teen",
        r"twenty",
        r"thirty",
        r"forty",
        r"fifty",
        r"sixty",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_object_factory() -> Callable[[str, str], bool]:
    url = "https://github.com/anishathalye/imagenet-simple-labels/raw/master/imagenet-simple-labels.json"

    response = requests.get(url)

    json_data = response.json()

    assert isinstance(json_data, list)

    def match_object(question: str, answer: str) -> bool:
        question_regex_list = [
            r"object",
            r"thing",
        ]

        answer_regex_list = json_data

        return any(
            re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
        ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)

    return match_object


def match_shape(question: str, answer: str) -> bool:
    question_regex_list = [
        r"shape",
    ]
    answer_regex_list = [
        r"circle",
        r"triangle",
        r"square",
        r"rectangle",
        r"pentagon",
        r"hexagon",
        r"heptagon",
        r"octagon",
        r"nonagon",
        r"decagon",
        r"ellipse",
        r"oval",
        r"star",
        r"cross",
        r"heart",
        r"crescent",
        r"semicircle",
        r"annulus",
        r"ring",
        r"trapezoid",
        r"parallelogram",
        r"rhombus",
        r"kite",
        r"arrow",
        r"spiral",
        r"cube",
        r"pyramid",
        r"cone",
        r"cylinder",
        r"sphere",
        r"torus",
        r"ellipsoid",
        r"prism",
        r"frustum",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_texture(question: str, answer: str) -> bool:
    question_regex_list = [
        r"texture",
    ]
    answer_regex_list = [
        r"smooth",
        r"rough",
        r"bumpy",
        r"spiky",
        r"prickly",
        r"spiny",
        r"slippery",
        r"sticky",
        r"soft",
        r"hard",
        r"brittle",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_spatial_relationship(question: str, answer: str) -> bool:
    question_regex_list = [
        r"left",
        r"right",
        r"above",
        r"below",
        r"top",
        r"bottom",
        r"front",
        r"back",
        r"inside",
        r"outside",
        r"near",
        r"far",
        r"between",
        r"next to",
        r"beside",
        r"under",
        r"over",
        r"on",
        r"off",
        r"against",
        r"towards",
        r"away",
        r"up",
        r"down",
        r"north",
        r"south",
        r"east",
        r"west",
        r"northeast",
        r"northwest",
        r"southeast",
        r"southwest",
        r"uphill",
        r"downhill",
        r"upstream",
        r"downstream",
        r"upwind",
        r"downwind",
    ]
    answer_regex_list = [
        r"left",
        r"right",
        r"above",
        r"below",
        r"top",
        r"bottom",
        r"front",
        r"back",
        r"inside",
        r"outside",
        r"near",
        r"far",
        r"between",
        r"next to",
        r"beside",
        r"under",
        r"over",
        r"on",
        r"off",
        r"against",
        r"towards",
        r"away",
        r"up",
        r"down",
        r"north",
        r"south",
        r"east",
        r"west",
        r"northeast",
        r"northwest",
        r"southeast",
        r"southwest",
        r"uphill",
        r"downhill",
        r"upstream",
        r"downstream",
        r"upwind",
        r"downwind",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)
