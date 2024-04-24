from ontology import Food, onto


def main():
    object_list = [str(x).removeprefix(".") for x in onto.individuals()]
    print(object_list)


if __name__ == "__main__":
    main()
