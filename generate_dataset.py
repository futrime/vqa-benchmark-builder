import json
import logging
import random
from typing import Any, Generator, List, NoReturn, Optional, TypedDict

import owlready2
import tqdm

from ontology import Thing, onto

IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"
OUTPUT_TRAIN_METADATA_FILE = "data/custom/train.json"
OUTPUT_VAL_METADATA_FILE = "data/custom/val.json"
OUTPUT_TEST_METADATA_FILE = "data/custom/test.json"
OUTPUT_SFT_FILE = "data/custom/sft.json"

DUMP_OUTPUT_QA_FILE_INTERVAL = 1000
LOGGING_LEVEL = logging.INFO
RANDOM_SEED = 42
RATIO_TRAIN = 0.8
RATIO_VAL = 0.1
RATIO_TEST = 0.1
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


class SftEntry(TypedDict):
    id: str
    image: str
    conversations: List["ConversationDescriptor"]


class ConversationDescriptor(
    TypedDict(
        "ConversationDescriptor",
        {
            "from": str,
            "value": str,
        },
    )
):
    pass


class QaEntryGenerator:
    def __init__(self, possible_instances: List[owlready2.Thing]):
        self._id_generator = self._make_id_generator()
        self._possible_instances = possible_instances

    def generate(self, image_descriptor: ImageDescriptor) -> QaEntry:
        rendered_instances: List[owlready2.Thing] = [
            x
            for x in self._possible_instances
            if x.name in [x["name"] for x in image_descriptor["objects"]]
        ]

        id = self._generate_id()
        while True:
            qa_entry = self._try_generate(
                id, image_descriptor["id"], rendered_instances
            )
            if qa_entry is not None:
                return qa_entry

    def generate_multiple(
        self, image_descriptors: List[ImageDescriptor], count: int
    ) -> List[QaEntry]:
        qa_entries = []
        for image_descriptor in tqdm.tqdm(image_descriptors):
            for _ in range(count):
                qa_entry = self.generate(image_descriptor)
                qa_entries.append(qa_entry)
        return qa_entries

    def _try_generate(
        self, id: int, image_id: int, rendered_instances: List[owlready2.Thing]
    ) -> Optional[QaEntry]:
        try:
            # Generate answer first.
            answer_should_be_yes = random.random() < 0.5

            if answer_should_be_yes:
                selected_thing = random.choice(rendered_instances)
            else:
                instances = self._possible_instances
                selected_thing = random.choice(
                    [x for x in instances if x not in rendered_instances]
                )

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
                "image_id": image_id,
                "question": question,
                "steps": steps,
                "answer": "yes" if answer_should_be_yes else "no",
            }

            return qa_entry

        except:
            return None

    def _generate_id(self) -> int:
        return next(self._id_generator)

    def _make_id_generator(self) -> Generator[int, Any, NoReturn]:
        id = 0
        while True:
            yield id
            id += 1


def main():
    logging.basicConfig(level=LOGGING_LEVEL)

    random.seed(RANDOM_SEED)

    with open(IMAGE_DESCRIPTOR_FILE, "r") as f:
        image_descriptors: List[ImageDescriptor] = json.load(f)

    logging.info("Generating QA entries...")

    qa_entry_generator = QaEntryGenerator(list(Thing.instances()))

    # Split the image descriptors into train, val, and test sets.
    random.shuffle(image_descriptors)
    train_count = int(len(image_descriptors) * RATIO_TRAIN)
    val_count = int(len(image_descriptors) * RATIO_VAL)

    train_image_descriptors = image_descriptors[:train_count]
    val_image_descriptors = image_descriptors[train_count : train_count + val_count]
    test_image_descriptors = image_descriptors[train_count + val_count :]

    logging.info("Generating QA entries for train set...")
    train_qa_entries = qa_entry_generator.generate_multiple(
        train_image_descriptors, QA_COUNT_PER_IMAGE
    )
    logging.info("Generating QA entries for val set...")
    val_qa_entries = qa_entry_generator.generate_multiple(
        val_image_descriptors, QA_COUNT_PER_IMAGE
    )
    logging.info("Generating QA entries for test set...")
    test_qa_entries = qa_entry_generator.generate_multiple(
        test_image_descriptors, QA_COUNT_PER_IMAGE
    )

    logging.info("Dumping QA entries...")
    with open(OUTPUT_TRAIN_METADATA_FILE, "w") as f:
        json.dump(train_qa_entries, f, indent=4)
    with open(OUTPUT_VAL_METADATA_FILE, "w") as f:
        json.dump(val_qa_entries, f, indent=4)
    with open(OUTPUT_TEST_METADATA_FILE, "w") as f:
        json.dump(test_qa_entries, f, indent=4)

    logging.info("Generating SFT entries...")

    sft_entries: List[SftEntry] = []
    for qa_entry in tqdm.tqdm(train_qa_entries):
        id = str(qa_entry["id"])
        image = f"{qa_entry['image_id']}.png"

        gpt_output = ""
        for step_num, step in enumerate(qa_entry["steps"]):
            gpt_output += f"{step_num + 1}. {step}\n"

        gpt_output += f"<answer>{qa_entry['answer']}</answer>"

        conversations: List[ConversationDescriptor] = [
            {
                "from": "human",
                "value": qa_entry["question"],
            },
            {
                "from": "gpt",
                "value": gpt_output,
            },
        ]

        sft_entries.append(
            {
                "id": id,
                "image": image,
                "conversations": conversations,
            }
        )

    with open(OUTPUT_SFT_FILE, "w") as f:
        json.dump(sft_entries, f, indent=4)


if __name__ == "__main__":
    main()
