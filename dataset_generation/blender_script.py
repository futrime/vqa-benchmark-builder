import json
import math
import os
import random
from typing import List, TypedDict

import bpy
import bpy.types

OUTPUT_IMAGE_DIR = "data/custom/images/"
OUTPUT_IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"

POSSIBLE_OBJECT_COLLECTION_NAME = "Collection"
PLACEHOLDER_COLLECTION_NAME = "Empty Objects"

IMAGE_COUNT = 100
RANDOM_SEED = 42


class ObjectDescriptor(TypedDict):
    name: str
    location: List[float]


class ImageDescriptor(TypedDict):
    objects: List[ObjectDescriptor]


def main():
    random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # Make sure that IMAGE_COUNT images can be generated.
    placeholder_locations = get_placeholder_locations()
    possible_objects = get_collection_objects(POSSIBLE_OBJECT_COLLECTION_NAME)
    max_permutation_count = math.perm(len(possible_objects), len(placeholder_locations))
    if IMAGE_COUNT > max_permutation_count:
        raise ValueError(
            f"IMAGE_COUNT {IMAGE_COUNT} exceeds the maximum number of permutations: {max_permutation_count}"
        )

    setup_gpu_rendering()

    # Make all possible object invisible.
    for obj in get_collection_objects(POSSIBLE_OBJECT_COLLECTION_NAME):
        obj.hide_render = True

    image_descriptors: List[ImageDescriptor] = []

    for index in range(IMAGE_COUNT):
        image_descriptor = generate_image(index, image_descriptors)
        image_descriptors.append(image_descriptor)

    with open(OUTPUT_IMAGE_DESCRIPTOR_FILE, "w") as f:
        json.dump(image_descriptors, f)


def generate_image(
    index: int, image_descriptors: List[ImageDescriptor]
) -> ImageDescriptor:
    """Generates an image with a random number of objects placed at random locations.

    Args:
        index: The index of the image.

    Returns:
        A list of object names that were placed in the image.
    """

    placeholder_locations = get_placeholder_locations()
    possible_objects = get_collection_objects(POSSIBLE_OBJECT_COLLECTION_NAME)

    is_generated = False
    while not is_generated:
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

        # Create image descriptor.
        image_descriptor: ImageDescriptor = {"objects": []}
        for new_object in new_objects:
            location = [
                float(new_object.location[0]),
                float(new_object.location[1]),
                float(new_object.location[2]),
            ]
            object_descriptor: ObjectDescriptor = {
                "name": new_object.name.removesuffix(".new"),
                "location": location,
            }
            image_descriptor["objects"].append(object_descriptor)

        if image_descriptor not in image_descriptors:
            is_generated = True

    # Render.
    bpy.context.scene.render.filepath = os.path.join(OUTPUT_IMAGE_DIR, f"{index}.png")
    bpy.ops.render.render(write_still=True)

    # Clean up.
    for obj in new_objects:
        bpy.data.objects.remove(obj)

    return image_descriptor


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
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()


if __name__ == "__main__":
    main()
