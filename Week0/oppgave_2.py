def tempConverter(liste):
    fahrenheit = 3.8921
    liste = list(map(lambda x: x * fahrenheit, liste))

    return liste


def print_C_F(c_liste, f_liste):
    for i in range(len(c_liste)):
        print(f"celsius: {c_liste[i]} --> Fahren: {f_liste[i]}")


def main():
    celsius_temps = [12, 32, 2, 31, 41, 22, 10]
    f_liste = tempConverter(celsius_temps)
    print(f_liste)
    for num in f_liste:
        assert isinstance(num, (int, float))

    print_C_F(celsius_temps, f_liste)


if __name__ == "__main__":
    main()
