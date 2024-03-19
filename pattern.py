from typing import List, TypedDict


class Pattern(TypedDict):
    """A pattern is a list of regular expressions that match the same
    concept. For example, the pattern "color" might include the regular
    expressions "color" and "colour".

    Attributes:
        name: The name of the pattern.
        question: A list of regular expressions that match the concept in
            the question.
        answer: A list of regular expressions that match the concept in
            the answer.
    """

    name: str
    common: List[str]
    question: List[str]
    answer: List[str]
