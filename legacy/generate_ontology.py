import itertools
import os
import re
from collections import ChainMap
from typing import List

import numpy as np
import owlready2
import pandas as pd

from dataset_generation.ontology import onto

MAX_DEPTH = 5
ONTOLOGY_FILE_PATH = "data/custom/ontology.json"


# recursive method to get class hierarchy
def get_types_hierarchy(x):
    if issubclass(x, owlready2.Thing):
        # to filter out Thing's, uncomment below
        # if x is not Thing:
        return [x, *get_types_hierarchy(x.__bases__[0])]
    return []


# gets a tuple representing a relationship (recursive with get_object_pairs)
def get_relation_pairs(o, depth=1):
    return [
        ("relation", p, get_object_pairs(getattr(o, p._name), depth=depth + 1))
        for p in o.get_properties()
        if owlready2.ObjectPropertyClass
    ]


# gets a tuple representing a pair of objects (recursive with get_relation_pairs), [0, 1, 2, 3] = [entity, types, x, relation_pairs]
def get_object_pairs(objects, limit_domain=[], depth=1):
    if depth > MAX_DEPTH:
        return []
    return [
        ("entity", get_types_hierarchy(type(x)), x, get_relation_pairs(x, depth))
        for x in objects
        if len(limit_domain) == 0 or any([isinstance(x, y) for y in limit_domain])
    ]


# works with dictify to flatten out the hierarchy to a one-dict per entity/relationship combination including recurse
def dict_collapse(e, depth=1):
    has_relationships = len(e[3]) > 0
    if has_relationships:
        return [
            {f"e{depth}": e[0:3], f"r{depth}": r[0:2], "n": dictify(r[2], depth + 1)}
            for r in e[3]
        ]
    else:
        return [{f"e{depth}": e[0:3], f"r{depth}": None, "n": []}]


# removes the "n" column which is used to indicate another level of recursion
def drop_n(d):
    return {k: v for k, v in d.items() if k != "n"}


# helper to flatten nested lists, put two sublists into a combined list.
def flatten(l):
    return list(itertools.chain.from_iterable(l))


# Removes a single level of recursion then calls itself again
def flatten_dicts(dict_list):
    # handle case of no next level and case of having a next level then recurse
    recursed_set = [{**drop_n(e), **n} for e in dict_list for n in e["n"]]
    if any(["n" in e for e in recursed_set]):
        recursed_set = flatten_dicts(recursed_set)
    return recursed_set + [drop_n(e) for e in dict_list if len(e["n"]) == 0]


# controller method to orchestrate the other 2
def dictify(object_pairs, depth=1):
    if depth > MAX_DEPTH:
        return []
    collapsed_dicts = flatten([dict_collapse(e, depth) for e in object_pairs])
    return flatten_dicts(collapsed_dicts)


# --- JY Build the table
# --- Third method set, no more recursion, these methods are just helpful to coerce to a table-like format, could easily write your own methods instead :) ---
# --- Output: dictionary with an entity or list of classes per column making it easier to filter and work with ---

# convert the whole ontology to a table format
from collections import ChainMap


def tablify(object_tuple_dict):
    def generate_entity_dict(k, v):
        if v is None:
            return {}
        elif v[0] == "entity":
            return {f"{k}_classes": v[1], f"{k}_entity": v[2]}
        else:
            return {f"{k}_relation": v[1]}

    return [
        dict(ChainMap(*[generate_entity_dict(k, v) for k, v in x.items()]))
        for x in object_tuple_dict
    ]


def stringify_table(dict_table_format):
    return [
        {
            k: [x.name for x in v] if isinstance(v, list) else v._name
            for k, v in x.items()
        }
        for x in dict_table_format
    ]


def unique_concat(*args):
    multi_space = re.compile("\\s+")
    multi_space_v = np.vectorize(lambda x: multi_space.sub(" ", x))
    return np.char.array(multi_space_v(np.unique(np.concatenate(args))))


def main() -> None:
    object_pairs = get_object_pairs(onto.individuals(), [onto.Vegetable])
    object_tuple_dict = dictify(object_pairs)
    dict_table_format = tablify(object_tuple_dict)
    string_table = stringify_table(dict_table_format)

    # table_cols_threelevels = [
    #     "e1_classes",
    #     "e1_entity",
    #     "r1_relation",
    #     "e2_classes",
    #     "e2_entity",
    #     "r2_relation",
    #     "e3_classes",
    #     "e3_entity",
    # ]

    # string_df = (
    #     pd.DataFrame(string_table)[table_cols_threelevels]
    #     .fillna("")
    #     .query("r1_relation!=''")
    # )

    table_cols_threelevels: List[str] = []
    for i in range(1, MAX_DEPTH):
        table_cols_threelevels.extend(
            [
                f"e{i}_classes",
                f"e{i}_entity",
                f"r{i}_relation",
            ]
        )
    table_cols_threelevels.extend(
        [
            f"e{MAX_DEPTH}_classes",
            f"e{MAX_DEPTH}_entity",
        ]
    )

    string_df = pd.DataFrame(string_table)
    string_df = string_df[table_cols_threelevels]
    string_df = string_df.dropna()

    for i in range(1, MAX_DEPTH + 1):
        for j in range(1, MAX_DEPTH + 1):
            if i == j:
                continue

            string_df = string_df.drop(
                string_df[
                    (string_df[f"e{i}_entity"] == string_df[f"e{j}_entity"])
                ].index
            )

    os.makedirs(os.path.dirname(ONTOLOGY_FILE_PATH), exist_ok=True)
    string_df.to_json(ONTOLOGY_FILE_PATH, orient="records")


if __name__ == "__main__":
    main()
