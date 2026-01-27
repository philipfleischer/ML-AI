import numpy as np


def mat_cons():
    # 1) 3x3 matrise fylt med 0
    arr1 = np.zeros((3, 3))
    print("1) 3x3 matrise fylt med 0 -> arr1: \n", arr1, "\n")

    # 2) 2x4 matrisse fylt med 1
    arr2 = np.ones((2, 4))
    print("2) 2x4 matrisse fylt med 1 -> arr2: \n", arr2, "\n")

    # 3) Tall fra 1 til 25 med 5 steg
    arr3 = np.arange(1, 26, 5)
    print("3) Tall fra 1 til 25 med 5 steg -> arr3: \n", arr3, "\n")

    # 4) 10 tilfeldige desimaltall mellom 0 og 1
    arr4 = np.random.rand(10)
    print("4) 10 tilfeldige desimaltall mellom 0 og 1 -> arr4: \n", arr4, "\n")

    # 5) Tilfeldige heltall mellom 1 og 10 i en 3x3 matrise
    arr5 = np.random.randint(1, 11, (3, 3))
    print(
        "5) Tilfeldige heltall mellom 1 og 10 i en 3x3 matrise -> arr5: \n", arr5, "\n"
    )

    # 6) Fast startverdi (seed) for reproduserbarhet
    np.random.seed(123)
    arr6 = np.random.rand(10)
    print("6) Fast startverdi (seed) for reproduserbarhet -> arr6: \n", arr6, "\n")

    np.random.seed(123)
    arr7 = np.random.rand(10)
    print(
        "6) Fast startverdi (seed) for reproduserbarhet andre gang -> arr6: \n",
        arr7,
        "\n",
    )


def main():
    mat_cons()


if __name__ == "__main__":
    main()
