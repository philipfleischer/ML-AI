def sorter(navn, reverse=False):
    navn.sort(key=lambda x: len(x), reverse=reverse)

    for n in navn:
        print(f"{n} ({len(n)})")


def main():
    navn = ["Philip", "Ola", "Peder"]
    sorter(navn)
    print()
    sorter(navn, True)


if __name__ == "__main__":
    main()
