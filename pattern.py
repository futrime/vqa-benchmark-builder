from typing import Callable, TypedDict


class Pattern(TypedDict):
    """A pattern for matching a pair of question and answer to a tag.

    Attributes:
        name: The name of the pattern.
        matcher: A function that takes a question and an answer and returns
            True if the question and answer match the pattern, and False
            otherwise.
    """

    name: str
    matcher: Callable[[str, str], bool]
