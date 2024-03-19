from typing import Dict, List

from pattern import Pattern

PATTERN_LIST: List[Pattern] = [
    {
        "name": "color",
        "common": [
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
        ],
        "question": [r"color", r"colour"],
        "answer": [],
    },
    {
        "name": "shape",
        "common": [],
        "question": [r"shape"],
        "answer": [
            r"round",
            r"square",
            r"rectangle",
            r"triangle",
            r"oval",
            r"circular",
            r"elliptical",
            r"hexagonal",
            r"pentagonal",
            r"cylindrical",
            r"spherical",
            r"cuboid",
            r"pyramidal",
            r"conical",
            r"crescent",
            r"heart-shaped",
        ],
    },
    {
        "name": "texture",
        "common": [],
        "question": [r"texture"],
        "answer": [
            r"smooth",
            r"rough",
            r"bumpy",
            r"grainy",
            r"wrinkled",
            r"spiky",
            r"prickly",
            r"soft",
            r"hard",
            r"slippery",
            r"sticky",
            r"fluffy",
            r"fuzzy",
            r"feathery",
            r"wooly",
            r"silky",
            r"velvety",
            r"leathery",
            r"rubbery",
            r"plastic",
            r"metallic",
            r"glassy",
            r"porous",
            r"fibrous",
            r"granular",
            r"powdery",
            r"crumbly",
            r"brittle",
            r"elastic",
            r"resilient",
            r"tough",
            r"fragile",
            r"delicate",
            r"sturdy",
            r"solid",
            r"liquid",
            r"gaseous",
        ],
    },
    {
        "name": "object",
        "common": [],
        "question": [r"object", r"thing", r"item"],
        "answer": [
            r"person",
            r"car",
            r"tree",
            r"dog",
            r"cat",
            r"bicycle",
            r"bus",
            r"traffic light",
            r"stop sign",
            r"bench",
            r"building",
            r"sky",
            r"road",
            r"sidewalk",
            r"truck",
            r"motorcycle",
            r"boat",
            r"bird",
            r"horse",
            r"cow",
            r"sheep",
            r"chair",
            r"table",
            r"airplane",
            r"train",
            r"elephant",
            r"bear",
            r"zebra",
            r"giraffe",
            r"backpack",
            r"umbrella",
            r"suitcase",
            r"handbag",
            r"tie",
            r"frisbee",
            r"skis",
            r"snowboard",
            r"kite",
            r"baseball bat",
            r"baseball glove",
            r"skateboard",
            r"surfboard",
            r"tennis racket",
            r"bottle",
            r"wine glass",
            r"cup",
            r"fork",
            r"knife",
            r"spoon",
            r"bowl",
            r"banana",
            r"apple",
            r"sandwich",
            r"orange",
            r"broccoli",
            r"carrot",
            r"hot dog",
            r"pizza",
            r"donut",
            r"cake",
            r"chair",
            r"couch",
            r"potted plant",
            r"bed",
            r"dining table",
            r"toilet",
            r"tv",
            r"laptop",
            r"mouse",
            r"remote",
            r"keyboard",
            r"cell phone",
            r"microwave",
            r"oven",
            r"toaster",
            r"sink",
            r"refrigerator",
            r"book",
            r"clock",
            r"vase",
            r"scissors",
            r"teddy bear",
            r"hair drier",
            r"toothbrush",
        ],
    },
    {
        "name": "spatial relationship",
        "common": [],
        "question": [r"where", r"position", r"location"],
        "answer": [
            r"left",
            r"right",
            r"top",
            r"bottom",
            r"inside",
            r"outside",
            r"above",
            r"below",
            r"over",
            r"under",
            r"behind",
            r"in front of",
            r"next to",
            r"near",
            r"far",
            r"between",
            r"among",
            r"around",
            r"through",
            r"across",
            r"over",
            r"under",
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
        ],
    },
    {
        "name": "count",
        "common": [],
        "question": [r"how many", r"number"],
        "answer": [
            r"\d+",
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
            r"eleven",
            r"twelve",
            r"thirteen",
            r"fourteen",
            r"fifteen",
            r"sixteen",
            r"seventeen",
            r"eighteen",
            r"nineteen",
            r"twenty",
            r"thirty",
            r"forty",
            r"fifty",
            r"sixty",
            r"seventy",
            r"eighty",
            r"ninety",
            r"hundred",
            r"thousand",
            r"million",
            r"billion",
            r"trillion",
            r"quadrillion",
            r"quintillion",
            r"sextillion",
            r"septillion",
            r"octillion",
            r"nonillion",
            r"decillion",
            r"undecillion",
            r"duodecillion",
            r"tredecillion",
            r"quattuordecillion",
            r"quindecillion",
            r"sexdecillion",
            r"septendecillion",
            r"octodecillion",
            r"novemdecillion",
            r"vigintillion",
            r"centillion",
            r"googol",
            r"googolplex",
            r"infinitillion",
        ],
    },
]
