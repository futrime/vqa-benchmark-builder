import json
import os
from typing import List, TypedDict

import bpy
import mathutils
import numpy as np

IQA_FILE_PATH = "data/custom/iqa.json"
IMAGE_OUTPUT_DIR = "data/custom/images/"


class Iqa(TypedDict):
    id: int
    img_desc: List[str]
    question: str
    steps: List[str]


def main():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    with open(IQA_FILE_PATH) as f:
        iqa_list: List[Iqa] = json.load(f)

    for iqa in iqa_list:
        iqa: Iqa
        try:
            unlink_object(iqa["img_desc"])
            place_object(iqa["img_desc"])
            render_image(os.path.join(IMAGE_OUTPUT_DIR, f"{iqa['id']}.png"))

        except Exception as e:
            print(f"error: {e}")

        delete_object(iqa["img_desc"])


def copy_new_object(name, location):
    move = mathutils.Vector(location)
    obj = bpy.data.objects[name]
    cp_obj = obj.copy()
    cp_obj.location = move
    collection = bpy.data.collections["Collection"]
    collection.objects.link(cp_obj)


def get_empty_locs():
    locs = []
    for obj in bpy.data.scenes["Scene"].collection.all_objects:
        # for obj in bpy.data.scollection.all_objects:
        if "Empty." in obj.name:
            locs.append(obj.location)
    return locs


def place_object(names):
    locs = get_empty_locs()
    num_objects = len(names)
    # selected_locs = random.choices(locs, k=num_objects)
    # #selected_names = random.sample(names, num_objects)
    #
    # for i in range(num_objects):
    #     copy_new_object(names[i], selected_locs[i])
    #     print(names[i], selected_locs[i])

    selected_indexes = np.random.choice(len(locs), num_objects, replace=False)

    for i, i_loc in enumerate(selected_indexes):
        copy_new_object(names[i], locs[i_loc])


def render_image(path: str):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=1)


def unlink_object(names):
    collection = bpy.data.collections["Collection"]
    # sub_collection = bpy.data.scenes['Scene'].collection.all_objects
    unlink_objects = []
    # for obj in collection.objects:
    for obj in collection.objects:
        if obj.name_full in names:
            unlink_objects.append(obj.name_full)

    # for obj in bpy.data.objects:
    for obj in bpy.data.objects:
        if obj.name_full in unlink_objects:
            collection.objects.unlink(obj)


def delete_object(names):
    # delete after save
    for name in names:
        collection = bpy.data.collections["Collection"]
        for obj in collection.objects:
            if name + "." in obj.name_full and obj.name_full != name:
                collection.objects.unlink(obj)


if __name__ == "__main__":
    main()
