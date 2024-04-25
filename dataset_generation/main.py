from ontology import onto

IMAGE_DESCRIPTOR_FILE = "data/custom/image_descriptors.json"


def main():
    object_list = [str(x).removeprefix(".") for x in onto.individuals()]
    print(object_list)


if __name__ == "__main__":
    main()
