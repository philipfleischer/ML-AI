import numpy as np


def arrays():
    # 1) 1D-array B med 1..16
    B = np.arange(1, 17)
    print("1D B: \n", B, "\n")

    # 2) reshape til (4,4)
    M = B.reshape(4, 4)
    print("M (4x4): \n", M, "\n")

    # 3) Transponer -> Bryter rader og kolonner i en matrise (rad->kolonne, kolonne->rad)
    MT = M.T
    print("M.T:\n", MT, "\n")

    # 4) Stable vertikalt -> Legger matriser oppå hverandre ved å slå sammen radene (øker antall rader)
    S = np.vstack((M, MT))
    print("Stablet (M over M.T):\n", S, "\n")

    # 5) Sum av alle elementer
    total_sum = S.sum()
    print("Sum av alle elementer:", total_sum, "\n")

    # 6) Maks i hver rad
    row_max = S.max(axis=1)
    print("Maks i hver rad:", row_max, "\n")

    # 7) Indeks til min i hver kolonne
    col_argmin = S.argmin(axis=0)
    print("Indeks til min i hver kolonne:", col_argmin, "\n")


def main():
    arrays()


if __name__ == "__main__":
    main()
