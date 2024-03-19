import re


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
