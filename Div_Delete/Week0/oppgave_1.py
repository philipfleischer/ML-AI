def avg_grade(ob):
    ny_ordbok = {}

    for name, grades in ob.items():
        tot = 0
        for grade in grades:
            tot += grade

        ny_ordbok[name] = tot / len(grades)

    return ny_ordbok


def print_avg_grade(ob):
    for name, grade in ob.items():
        print(f"Name: {name} and average grade: {grade}")


def main():
    ordbok = {
        "Philip": [5, 4, 3],
        "Jonas": [5, 4, 3],
        "Emil": [5, 4, 3],
        "Kasper": [5, 4, 4],
    }
    avg_ordbok = avg_grade(ordbok)
    print_avg_grade(avg_ordbok)


if __name__ == "__main__":
    main()
