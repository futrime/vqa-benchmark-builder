from typing import List, TypedDict

from ontology import onto

IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"


class ImageDescriptor(TypedDict):
    id: int
    objects: List["ObjectDescriptor"]


class ObjectDescriptor(TypedDict):
    name: str
    location: List[float]


class QaEntry(TypedDict):
    id: int
    image_id: int
    question: str
    steps: List[str]
    answer: str


def main():
    pass


if __name__ == "__main__":
    main()
