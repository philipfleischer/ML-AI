import numpy as np


def broadcast():
    # 1) Opprett array A med verdiene 1-15, på formen (5,3)
    A = np.arange(1, 16).reshape(5, 3)

    # 2) Opprett radvektor v, på formen [10,0,-10]
    v = np.array([10, 0, -10])

    # 3) Opprett kolonnevektor r, på formen r = [[1], [2], [3], [4], [5]]
    r = np.arange(1, 6).reshape(5, 1)

    # 4) Legg v til hver rad i A (broadcasting)
    B = A + v

    # 5) Multipliser hver rad i resultatet med c (broadcasting)
    C = B * r

    # 6) Beregn gjennomsnitt a kolonner og summen av rader
    col_mean = C.mean(axis=0)
    row_sum = C.sum(axis=1)

    return A, v, r, B, C, col_mean, row_sum


def main():
    A, v, r, B, C, col_mean, row_sum = broadcast()

    print("A:")
    print(A, "\n")

    print("Radvektor v:")
    print(v, "\n")

    print("Kolonnevektor r:")
    print(r, "\n")

    print("A + v:")
    print(B, "\n")

    print("(A + v) * r:")
    print(C, "\n")

    print("Kolonne-gjennomsnitt (akse=0):")
    print(col_mean, "\n")

    print("Radsummer (akse=1):")
    print(row_sum, "\n")


if __name__ == "__main__":
    main()
