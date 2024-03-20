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
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    )


def match_emotion(question: str, answer: str) -> bool:
    question_regex_list = [
        r"face",
        r"facial",
        r"expression",
        r"emotion",
    ]
    answer_regex_list = [
        r"smiling",
        r"frowning",
        r"grimacing",
        r"scowling",
        r"grinning",
        r"smirking",
        r"laughing",
        r"crying",
        r"tearful",
        r"weeping",
        r"sobbing",
        r"pouting",
        r"pursed",
        r"puckered",
        r"smiling",
        r"grinning",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_age(question: str, answer: str) -> bool:
    question_regex_list = [
        r"age",
        r"how old",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    )


def match_human_dress(question: str, answer: str) -> bool:
    question_regex_list = [
        r"dress",
        r"clothes",
        r"outfit",
        r"attire",
        r"apparel",
    ]
    answer_regex_list = [
        r"shirt",
        r"t-shirt",
        r"blouse",
        r"top",
        r"sweater",
        r"jacket",
        r"coat",
        r"vest",
        r"hoodie",
        r"sweatshirt",
        r"cardigan",
        r"blazer",
        r"tunic",
        r"kimono",
        r"robe",
        r"dress",
        r"gown",
        r"skirt",
        r"pants",
        r"jeans",
        r"shorts",
        r"leggings",
        r"jumpsuit",
        r"overalls",
        r"romper",
        r"swimsuit",
        r"bikini",
        r"underwear",
        r"bra",
        r"panties",
        r"boxers",
        r"briefs",
        r"thong",
        r"pajamas",
        r"nightgown",
        r"nightshirt",
        r"nightdress",
        r"nightie",
        r"nightwear",
        r"sleepwear",
        r"loungewear",
        r"activewear",
        r"athleisure",
        r"uniform",
        r"costume",
        r"garment",
        r"clothing",
        r"apparel",
        r"outfit",
        r"attire",
        r"ensemble",
        r"getup",
        r"regalia",
        r"raiment",
        r"vesture",
        r"raiment",
        r"robe",
        r"vestment",
        r"vestment",
        r"trousseau",
        r"wardrobe",
        r"threads",
        r"rags",
        r"togs",
        r"togas",
        r"togae",
        r"togated",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_gender(question: str, answer: str) -> bool:
    question_regex_list = [
        r"man",
        r"woman",
        r"female",
        r"girl",
        r"lady",
        r"boy",
        r"male",
        r"gentleman",
    ]
    answer_regex_list = [
        r"man",
        r"woman",
        r"female",
        r"girl",
        r"lady",
        r"boy",
        r"male",
        r"gentleman",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_hair_style(question: str, answer: str) -> bool:
    question_regex_list = [
        r"hair",
        r"hairstyle",
        r"hairstyling",
        r"haircut",
        r"hairdo",
    ]
    answer_regex_list = [
        r"long",
        r"short",
        r"curly",
        r"straight",
        r"wavy",
        r"spiky",
        r"frizzy",
        r"bald",
        r"shaved",
        r"buzzed",
        r"mohawk",
        r"afro",
        r"braided",
        r"ponytail",
        r"bun",
        r"bob",
        r"bangs",
        r"parted",
        r"mullet",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_height(question: str, answer: str) -> bool:
    question_regex_list = [
        r"height",
        r"tall",
        r"short",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    )


def match_human_pose(question: str, answer: str) -> bool:
    question_regex_list = [
        r"pose",
        r"position",
    ]
    answer_regex_list = [
        r"standing",
        r"sitting",
        r"lying",
        r"walking",
        r"running",
        r"jumping",
        r"bending",
        r"twisting",
        r"stretching",
        r"reaching",
        r"climbing",
        r"swimming",
        r"diving",
        r"skating",
        r"skiing",
        r"riding",
        r"driving",
        r"flying",
        r"falling",
        r"landing",
        r"sliding",
        r"rolling",
        r"spinning",
        r"flipping",
        r"cartwheeling",
        r"handstanding",
        r"headstanding",
        r"pushing",
        r"pulling",
        r"lifting",
        r"carrying",
        r"throwing",
        r"catching",
        r"shooting",
        r"punching",
        r"kicking",
        r"blocking",
        r"tackling",
        r"defending",
        r"attacking",
        r"fighting",
        r"wrestling",
        r"boxing",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_race(question: str, answer: str) -> bool:
    question_regex_list = [
        r"race",
        r"ethnicity",
    ]
    answer_regex_list = [
        r"white",
        r"black",
        r"asian",
        r"hispanic",
        r"latino",
        r"middle eastern",
        r"indian",
        r"native american",
        r"pacific islander",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


def match_human_weight(question: str, answer: str) -> bool:
    question_regex_list = [
        r"weight",
        r"heavy",
        r"light",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    )


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


def match_orientation(question: str, answer: str) -> bool:
    question_regex_list = [
        r"orientation",
    ]
    answer_regex_list = [
        r"up",
        r"down",
        r"left",
        r"right",
        r"north",
        r"south",
        r"east",
        r"west",
    ]

    return any(
        re.search(regex, question, re.IGNORECASE) for regex in question_regex_list
    ) or any(re.search(regex, answer, re.IGNORECASE) for regex in answer_regex_list)


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
