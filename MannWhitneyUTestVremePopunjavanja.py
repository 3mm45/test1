import pandas as pd

from scipy.stats import mannwhitneyu

# Učitaj podatke
matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
matrica = matrica[matrica['interviewtime'] < 3600]
matrica = matrica[matrica['BrOdgovora'] >= 5]


def mannWhitneyUTest(ime, mat):
    print(ime)
    kolona1 = mat[mat['FormatUpitnika'] == 1]['interviewtime']
    kolona2 = mat[mat['FormatUpitnika'] == 2]['interviewtime']
    U1, p_value = mannwhitneyu(kolona1, kolona2)
    print(f"Mann-Whitney U test: U1 = {U1}, p = {p_value}")
    print("**************************")


# Ukupno medijana
mannWhitneyUTest("Ukupno", matrica)

# Ima1 odgovora
imaOdg = matrica[matrica['O123Ima1Nema2'] == 1]
mannWhitneyUTest("Ima odgovora", imaOdg)

# Nema2 odgovora
nemaOdg = matrica[matrica['O123Ima1Nema2'] == 2]
mannWhitneyUTest("Nema odgovora", nemaOdg)
