# H2: Veći okviri za unos odgovora na pitanja otvorenog tipa biće povezani  sa  manjim brojem neodgovora.

import pandas as pd
import csv
matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
mat = matrica.copy()

# Primer Python koda za hi-kvadrat test
import scipy.stats as stats
import numpy as np

# Podaci iz istraživanja
print('Otvoreno pitanje 1')
# Otvorena 1: Problemi ili negativna iskustva
veliki_okvir1 = [53, 212]  # Broj odgovora i neodgovora
mali_okvir1 = [39, 196]    # Broj odgovora i neodgovora
# Hi-kvadrat test
chi2, p_value, dof, expected = stats.chi2_contingency([veliki_okvir1, mali_okvir1])

# Ispis rezultata
print(f"Hi-kvadrat vrednost je {chi2} kod Otvorena pitanja 1: Problemi ili negativna iskustva ")
print(f"P-vrednost 1: {p_value}")
if p_value < 0.05:
    print("Hipoteza je statistički značajna, rezultat je < 0.05, , kod Otvorena pitanja 1: Problemi ili negativna iskustva.")
else:
    print("Nema dovoljno dokaza da se odbaci  hipoteza, rezultat > 0.05, kod Otvorena pitanja 1: Problemi ili negativna iskustva. ")

print('Otvoreno pitanje 2')
# Otvorena pitanja 2: Dodavanje informacija
veliki_okvir2 = [64, 201]  # Broj odgovora i neodgovora
mali_okvir2 = [43, 192]
# Hi-kvadrat test
chi2, p_value, dof, expected = stats.chi2_contingency([veliki_okvir2, mali_okvir2])

# Ispis rezultata
print(f"Hi-kvadrat vrednost je {chi2}, kod Otvorena pitanja 2: Dodavanje informacija ")
print(f"P-vrednost 2: {p_value}")
if p_value < 0.05:
    print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja 2: Dodavanje informacija ")
else:
    print("Nema dovoljno dokaza da se odbaci  hipoteza, rezultat > 0.05, Otvorena pitanja 2: Dodavanje informacija ")

print('Otvoreno pitanje 3')
veliki_okvir3 = [52, 213]  # Broj odgovora i neodgovora
mali_okvir3 = [30, 205]

# Hi-kvadrat test
chi2, p_value, dof, expected = stats.chi2_contingency([veliki_okvir3, mali_okvir3])

# Ispis rezultata
print(f"Hi-kvadrat vrednost je {chi2}, kod Otvorena pitanja 3: Sugestije za poboljšanje ")
print(f"P-vrednost 3: {p_value}")
if p_value < 0.05:
    print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja 3: Sugestije za poboljšanje ")
else:
    print("Nema dovoljno dokaza da se odbaci  hipoteza, rezultat > 0.05, kod Otvorena pitanja 3: Sugestije za poboljšanje ")

print('Otvoreno pitanje Ukupno')
# Otvoreno pitanje UKUPNO
# Ima odgovora 56m 90v Nema odgovora 169m 175v
veliki_okvir = [90, 169]  # Broj odgovora i neodgovora
mali_okvir = [56, 175]

# Hi-kvadrat test
chi2, p_value, dof, expected = stats.chi2_contingency([veliki_okvir, mali_okvir])

# Ispis rezultata
print(f"Hi-kvadrat vrednost je {chi2}, kod Otvorena pitanja na ukupnom uzorku: Sugestije za poboljšanje ")
print(f"P-vrednost na ukupnom uzorku: {p_value}")
if p_value < 0.05:
    print("Hipoteza je statistički značajna, rezultat je < 0.05, kod Otvorena pitanja na ukupnom uzorku: Sugestije za poboljšanje ")
else:
    print("Nema dovoljno dokaza da se odbaci  hipoteza, rezultat > 0.05, kod Otvorena pitanja na ukupnom uzorku: Sugestije za poboljšanje ")



# # Hipoteza 1, Ne postoji statistički značajna razlika u vremenu popunjavanja formata upitnika sa svim pitanjima na jednoj stranici i formata u kome je svako pitanje na zasebnoj stranici.
# import pandas as pd
# matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
# mat = matrica.copy()
#
# print("****** H1 ******")
# # Primer Python koda za hi-kvadrat test
# import scipy.stats as stats
# import numpy as np
#
# # Podaci iz istraživanja
# # ima_odgvora
# jedna_strana = [368.09, 194.57]  # duzina vremena popunjavanja kad ima odgovora na otvorena pitanja i kad nema odgovora
# slajdovi = [877.84, 166.69]    # duzina vremena popunjavanja kad ima odgovora na otvorena pitanja i kad nema odgovora
# # nema odgovora
# # ima_odg = [368.09 , 877.84]  # Broj odgovora i neodgovora za "sve zajedno"
# # nema_odg = [194.57 , 166.69]
#
# # Hi-kvadrat test
# chi2, p_value, dof, expected = stats.chi2_contingency([jedna_strana, slajdovi])
# # chi2, p_value, dof, expected = stats.chi2_contingency([ima_odg, nema_odg])
#
# # Ispis rezultata
# print(f"Hi-kvadrat vrednost: {chi2}")
# print(f"P-vrednost: {p_value}")
# if p_value < 0.05:
#     print("Hipoteza je statistički značajna.")
# else:
#     print("Nema dovoljno dokaza da se odbaci nulta hipoteza.")

