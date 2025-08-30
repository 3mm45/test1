# import pandas as pd
# import csv
# import scipy.stats as stats
# import numpy as np
#
# # Učitaj podatke
# matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
# mat = matrica.copy()
#
# # Funkcija za izračunavanje phi vrednosti
# def calculate_phi(chi2, n):
#     return np.sqrt(chi2 / n)
#
# # Podaci iz istraživanja
# print('Otvoreno pitanje 1')
# veliki_okvir1 = [53, 212]  # Broj odgovora i neodgovora
# mali_okvir1 = [39, 196]    # Broj odgovora i neodgovora
# # Hi-kvadrat test
# chi2_1, p_value_1, dof_1, expected_1 = stats.chi2_contingency([veliki_okvir1, mali_okvir1])
#
# # Ukupan broj uzoraka
# n1 = sum(veliki_okvir1) + sum(mali_okvir1)
#
# # Phi vrednost
# phi_1 = calculate_phi(chi2_1, n1)
#
# # Ispis rezultata
# print(f"Hi-kvadrat vrednost je {chi2_1} kod Otvorena pitanja 1: Problemi ili negativna iskustva ")
# print(f"P-vrednost 1: {p_value_1}")
# print(f"Phi vrednost 1: {phi_1}")
# if p_value_1 < 0.05:
#     print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja 1: Problemi ili negativna iskustva.")
# else:
#     print("Nema dovoljno dokaza da se odbaci hipoteza, rezultat > 0.05, kod Otvorena pitanja 1: Problemi ili negativna iskustva.")
#
# print('Otvoreno pitanje 2')
# veliki_okvir2 = [64, 201]  # Broj odgovora i neodgovora
# mali_okvir2 = [43, 192]
# # Hi-kvadrat test
# chi2_2, p_value_2, dof_2, expected_2 = stats.chi2_contingency([veliki_okvir2, mali_okvir2])
#
# # Ukupan broj uzoraka
# n2 = sum(veliki_okvir2) + sum(mali_okvir2)
#
# # Phi vrednost
# phi_2 = calculate_phi(chi2_2, n2)
#
# # Ispis rezultata
# print(f"Hi-kvadrat vrednost je {chi2_2}, kod Otvorena pitanja 2: Dodavanje informacija ")
# print(f"P-vrednost 2: {p_value_2}")
# print(f"Phi vrednost 2: {phi_2}")
# if p_value_2 < 0.05:
#     print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja 2: Dodavanje informacija ")
# else:
#     print("Nema dovoljno dokaza da se odbaci hipoteza, rezultat > 0.05, kod Otvorena pitanja 2: Dodavanje informacija.")
#
# print('Otvoreno pitanje 3')
# veliki_okvir3 = [52, 213]  # Broj odgovora i neodgovora
# mali_okvir3 = [30, 205]
# # Hi-kvadrat test
# chi2_3, p_value_3, dof_3, expected_3 = stats.chi2_contingency([veliki_okvir3, mali_okvir3])
#
# # Ukupan broj uzoraka
# n3 = sum(veliki_okvir3) + sum(mali_okvir3)
#
# # Phi vrednost
# phi_3 = calculate_phi(chi2_3, n3)
#
# # Ispis rezultata
# print(f"Hi-kvadrat vrednost je {chi2_3}, kod Otvorena pitanja 3: Sugestije za poboljšanje ")
# print(f"P-vrednost 3: {p_value_3}")
# print(f"Phi vrednost 3: {phi_3}")
# if p_value_3 < 0.05:
#     print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja 3: Sugestije za poboljšanje.")
# else:
#     print("Nema dovoljno dokaza da se odbaci hipoteza, rezultat > 0.05, kod Otvorena pitanja 3: Sugestije za poboljšanje.")
#
# print('Otvoreno pitanje Ukupno')
# veliki_okvir = [90, 169]  # Broj odgovora i neodgovora
# mali_okvir = [56, 175]
# # Hi-kvadrat test
# chi2_total, p_value_total, dof_total, expected_total = stats.chi2_contingency([veliki_okvir, mali_okvir])
#
# # Ukupan broj uzoraka
# n_total = sum(veliki_okvir) + sum(mali_okvir)
#
# # Phi vrednost
# phi_total = calculate_phi(chi2_total, n_total)
#
# # Ispis rezultata
# print(f"Hi-kvadrat vrednost je {chi2_total}, kod Otvorena pitanja na ukupnom uzorku: Sugestije za poboljšanje")
# print(f"P-vrednost na


import pandas as pd
import scipy.stats as stats
import numpy as np

# Učitaj podatke
matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
mat = matrica.copy()

# Funkcija za hi-kvadrat test i izračunavanje phi vrednosti
def chi2_test_and_phi(data, question):
    chi2, p_value, dof, expected = stats.chi2_contingency(data)
    n = sum(data[0]) + sum(data[1])
    phi = np.sqrt(chi2 / n)

    # Ispis rezultata
    print(f"\nHi-kvadrat vrednost za '{question}': {chi2}")
    print(f"P-vrednost za '{question}': {p_value}")
    print(f"Phi vrednost za '{question}': {phi}")
    if p_value < 0.05:
        print(f"Hipoteza je statistički značajna za '{question}' (p < 0.05).")
    else:
        print(f"Nema dovoljno dokaza da se odbaci hipoteza za '{question}' (p > 0.05).")

# Podaci za sva pitanja
questions_data = [
    ([53, 212], [39, 196], "Otvoreno pitanje 1: Problemi ili negativna iskustva"),
    ([64, 201], [43, 192], "Otvoreno pitanje 2: Dodavanje informacija"),
    ([52, 213], [30, 205], "Otvoreno pitanje 3: Sugestije za poboljšanje"),
    ([90, 169], [56, 175], "Otvoreno pitanje Ukupno: Sugestije")
]

# Izvršavanje testa za svako pitanje
for veliki_okvir, mali_okvir, question in questions_data:
    chi2_test_and_phi([veliki_okvir, mali_okvir], question)
