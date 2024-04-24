import random
import re
from typing import List, Tuple

import pandas as pd

from dataset_generation.ontology import Food

ONTOLOGY_FILE_PATH = "data/custom/ontology.json"
QA_FILE_PATH = "data/custom/iqa.json"

RANDOM_SEED = 42
MAX_ITEMS_IN_IMAGE = 10


def camel_case_to_phrase(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def make_iqsa(
    class_list_list: List[List[str]],
    entity_list: List[str],
    relation_list: List[str],
) -> Tuple[List[str], str, List[str], str]:
    assert len(relation_list) >= 1
    assert len(relation_list) == len(entity_list) - 1
    assert len(class_list_list) == len(entity_list)

    sampled_class_list = [random.choice(class_list) for class_list in class_list_list]
    sampled_class_list = [camel_case_to_phrase(x) for x in sampled_class_list]
    entity_list = [x.replace("_", " ") for x in entity_list]
    relation_list = [x.replace("_", " ") for x in relation_list]

    question = f"Which {sampled_class_list[0]} in the picture {relation_list[0]} a kind of {sampled_class_list[1]}, "
    for i in range(1, len(relation_list) - 2):
        question += f"that {relation_list[i]} a kind of {sampled_class_list[i + 1]}, "
    question += f"that {relation_list[-1]} {entity_list[-1]}?"

    img_desc = random.sample(
        list(Food.instances()), random.randint(2, MAX_ITEMS_IN_IMAGE)
    )
    img_desc = [str(x).replace(".", "") for x in img_desc]
    img_desc = [x.replace("_", " ") for x in img_desc]
    img_desc.append(entity_list[0])
    random.shuffle(img_desc)

    steps = [
        f"In picture there are {', '.join(img_desc[:-1])} and {img_desc[-1]}.",
        f"We can see that {entity_list[0]} is in the picture.",
    ]
    for i in range(len(relation_list)):
        steps.append(f"{entity_list[i]} {relation_list[i]} {entity_list[i + 1]}.")
    steps.append(f"Therefore, the answer is {entity_list[0]}.")

    answer = entity_list[0]

    return img_desc, question, steps, answer


def main() -> None:
    random.seed(42)

    ontology_df = pd.read_json(ONTOLOGY_FILE_PATH)

    qa_df = pd.DataFrame(columns=["id", "img_desc", "question", "steps", "answer"])

    column_names = ontology_df.columns.tolist()
    class_column_names = [x for x in column_names if "class" in x]
    entity_column_names = [x for x in column_names if "entity" in x]
    relation_column_names = [x for x in column_names if "relation" in x]

    for index, row in ontology_df.iterrows():
        class_list_list: List[List[str]] = [getattr(row, x) for x in class_column_names]
        entity_list: List[str] = [getattr(row, x) for x in entity_column_names]
        relation_list: List[str] = [getattr(row, x) for x in relation_column_names]

        img_desc, question, steps, answer = make_iqsa(
            class_list_list, entity_list, relation_list
        )

        qa_df.loc[-1] = [index, img_desc, question, steps, answer]
        qa_df.index = qa_df.index + 1
        qa_df = qa_df.sort_index()

    qa_df.to_json(QA_FILE_PATH, orient="records", indent=4)


if __name__ == "__main__":
    main()
