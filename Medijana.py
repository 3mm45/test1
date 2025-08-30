import pandas as pd
import pandas as pd
import numpy as np
from scipy import stats

from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency


# Učitaj podatke
matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
matrica = matrica[matrica['interviewtime'] < 3600]
matrica = matrica[matrica['interviewtime'] > 30]


# Ukupno medijana
ukupnoJednaStrana = matrica[matrica['FormatUpitnika'] == 1]
medUkupnoJednaStrana = ukupnoJednaStrana['interviewtime'].median()
print(f"Ukupno jedna strana medijana: {medUkupnoJednaStrana}")

ukupnoSlajdovi = matrica[matrica['FormatUpitnika'] == 2]
medUkupnoSlajdovi = ukupnoSlajdovi['interviewtime'].median()
print(f"Ukupno slajdovi medijana: {medUkupnoSlajdovi}")

# Ima1 odg medijana
imaOdg = matrica[matrica['O123Ima1Nema2'] == 1]

imaJednaStrana = imaOdg[imaOdg['FormatUpitnika'] == 1]
medImaJednaStrana = imaJednaStrana['interviewtime'].median()
print(f"Ima1 jedna strana medijana: {medImaJednaStrana}")

imaSlajdovi = imaOdg[imaOdg['FormatUpitnika'] == 2]
medImaSlajdovi = imaSlajdovi['interviewtime'].median()
print(f"Ima1 slajdovi medijana: {medImaSlajdovi}")

# Nema2 odg medijana
nemaOdg = matrica[matrica['O123Ima1Nema2'] == 2]

nemaJednaStrana = nemaOdg[nemaOdg['FormatUpitnika'] == 1]
medNemaJednaStrana = nemaJednaStrana['interviewtime'].median()
print(f"Nema2 jedna strana medijana: {medNemaJednaStrana}")

nemaSlajdovi = nemaOdg[nemaOdg['FormatUpitnika'] == 2]
medNemaSlajdovi= nemaSlajdovi['interviewtime'].median()
print(f"Nema2 slajdovi medijana: {medNemaSlajdovi}")
