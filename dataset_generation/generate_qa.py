import json
import logging
import random
from typing import List, TypedDict

import owlready2
import tqdm
from ontology import Thing, onto

IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"
OUTPUT_QA_FILE = "data/custom/qa.json"

DUMP_OUTPUT_QA_FILE_INTERVAL = 1000
LOGGING_LEVEL = logging.INFO
RANDOM_SEED = 42
QA_COUNT_PER_IMAGE = 5


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
    logging.basicConfig(level=LOGGING_LEVEL)

    random.seed(RANDOM_SEED)

    with open(IMAGE_DESCRIPTOR_FILE, "r") as f:
        image_descriptors: List[ImageDescriptor] = json.load(f)

    qa_entries: List[QaEntry] = []
    for i, image_descriptor in tqdm.tqdm(
        enumerate(image_descriptors), total=len(image_descriptors)
    ):
        id_list = [i * QA_COUNT_PER_IMAGE + j for j in range(QA_COUNT_PER_IMAGE)]
        qa_entries_of_current_image = generate_qa_list(id_list, image_descriptor)
        qa_entries.extend(qa_entries_of_current_image)

        if i % DUMP_OUTPUT_QA_FILE_INTERVAL == 0:
            with open(OUTPUT_QA_FILE, "w") as f:
                json.dump(qa_entries, f, indent=4)


def generate_qa_list(
    id_list: List[int], image_descriptor: ImageDescriptor
) -> List[QaEntry]:
    rendered_object_names: List[str] = [x["name"] for x in image_descriptor["objects"]]
    rendered_instances: List[owlready2.Thing] = [
        x for x in onto.individuals() if x.name in rendered_object_names
    ]
    assert len(rendered_instances) == len(rendered_object_names)

    qa_list: List[QaEntry] = []
    for id in id_list:
        question_set = set([x["question"] for x in qa_list])
        is_generated = False
        while not is_generated:
            qa_entry = try_generate_qa(id, rendered_instances)
            if qa_entry["question"] not in question_set:
                is_generated = True

        qa_list.append(qa_entry)

    return qa_list


def try_generate_qa(id: int, rendered_instances: List[owlready2.Thing]) -> QaEntry:
    # Generate answer first.
    answer_should_be_yes = random.random() < 0.5

    if answer_should_be_yes:
        selected_thing = random.choice(rendered_instances)
    else:
        selected_thing = random.choice(list(Thing.instances()))

    question = f"Is there a {selected_thing.name} in the image?"
    steps = [
        f"There are {len(rendered_instances)} objects in the image.",
        f"The objects are: {', '.join([x.name for x in rendered_instances])}.",
    ]

    if answer_should_be_yes:
        steps.append(f"The {selected_thing.name} is in the image.")
    else:
        steps.append(f"The {selected_thing.name} is not in the image.")

    qa_entry: QaEntry = {
        "id": id,
        "image_id": id,
        "question": question,
        "steps": steps,
        "answer": "yes" if answer_should_be_yes else "no",
    }

    return qa_entry


if __name__ == "__main__":
    main()
