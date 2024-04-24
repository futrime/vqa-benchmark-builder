import json
import os
import random
from typing import List, TypedDict

import bpy
import bpy.types

OUTPUT_IMAGE_DIR = "data/custom/images_new/"
OUTPUT_IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"

POSSIBLE_OBJECT_COLLECTION_NAME = "Collection"
PLACEHOLDER_COLLECTION_NAME = "Empty Objects"

IMAGE_COUNT = 100
RANDOM_SEED = 42


class ImageDescriptor(TypedDict):
    object_list: List[str]


def main():
    random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    setup_gpu_rendering()

    # Make all possible object invisible.
    for obj in get_collection_objects(POSSIBLE_OBJECT_COLLECTION_NAME):
        obj.hide_render = True

    image_descriptors: List[ImageDescriptor] = []

    for index in range(IMAGE_COUNT):
        object_list = generate_image(index)
        image_descriptors.append(
            {
                "object_list": object_list,
            }
        )

    with open(OUTPUT_IMAGE_DESCRIPTOR_FILE, "w") as f:
        json.dump(image_descriptors, f)


def generate_image(index: int) -> List[str]:
    """Generates an image with a random number of objects placed at random locations.

    Args:
        index: The index of the image.

    Returns:
        A list of object names that were placed in the image.
    """

    placeholder_locations = get_placeholder_locations()
    possible_objects = get_collection_objects(POSSIBLE_OBJECT_COLLECTION_NAME)

    random.shuffle(placeholder_locations)
    random.shuffle(possible_objects)

    assert len(possible_objects) >= 1
    object_count = random.randint(
        1, min(len(possible_objects), len(placeholder_locations))
    )

    new_objects: List[bpy.types.Object] = []

    for i in range(object_count):
        obj = possible_objects[i]
        location = placeholder_locations[i]

        new_object = place_object_at(obj, location)
        new_objects.append(new_object)

    bpy.context.scene.render.filepath = os.path.join(OUTPUT_IMAGE_DIR, f"{index}.png")
    bpy.ops.render.render(write_still=True)

    result = [str(x.name.removesuffix(".new")) for x in new_objects]

    # Clean up.
    for obj in new_objects:
        bpy.data.objects.remove(obj)

    return result


def get_collection_objects(collection_name: str) -> List[bpy.types.Object]:
    return [x for x in bpy.data.collections[collection_name].all_objects]


def get_placeholder_locations() -> List[bpy.types.bpy_prop_array]:
    return [x.location for x in get_collection_objects(PLACEHOLDER_COLLECTION_NAME)]


def place_object_at(
    obj: bpy.types.Object, location: bpy.types.bpy_prop_array
) -> bpy.types.Object:
    new_object = obj.copy()
    assert isinstance(new_object, bpy.types.Object)

    new_object.hide_render = False
    new_object.name = f"{obj.name}.new"
    new_object.location = location

    collection: bpy.types.Collection = bpy.data.collections["Collection"]
    collection.objects.link(new_object)

    return new_object


def setup_gpu_rendering():
    for scene in bpy.data.scenes:
        scene: bpy.types.Scene
        scene.cycles.device = "GPU"


if __name__ == "__main__":
    main()
