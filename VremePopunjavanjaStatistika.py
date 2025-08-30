import pandas as pd
import numpy as np
from scipy import stats

from scipy.stats import mannwhitneyu
# Učitaj podatke
matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))
matrica = matrica[matrica['interviewtime'] < 3600]
matrica = matrica[matrica['BrOdgovora'] >= 5]

def celaStatistika(ime, kolona):
    print(ime)
    n = len(kolona)
    mean = np.mean(kolona)
    median = np.median(kolona)
    std_dev = np.std(kolona, ddof=1)  # ddof=1 for sample standard deviation
    std_err = stats.sem(kolona)
    print(f"N: {n}, Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}, Standard Error of Mean: {std_err}")
    # print(f"N: {n},")
    # print(f"Mean: {mean},")
    # print(f"Median: {median},")
    # print(f"Standard Deviation: {std_dev},")
    # print(f"Standard Error of Mean: {std_err}")
    print("")

# Ukupno
celaStatistika("Ukupno - Jedna Strana", matrica[matrica['FormatUpitnika'] == 1]['interviewtime'])
celaStatistika("Ukupno - Slajdovi", matrica[matrica['FormatUpitnika'] == 2]['interviewtime'])

# Ima1 odgovora
imaOdg = matrica[matrica['O123Ima1Nema2'] == 1]
celaStatistika("Ima odgovora - Jedna Strana", imaOdg[imaOdg['FormatUpitnika'] == 1]['interviewtime'])
celaStatistika("Ima odgovora - Slajdovi", imaOdg[imaOdg['FormatUpitnika'] == 2]['interviewtime'])

# Nema2 odgovora
nemaOdg = matrica[matrica['O123Ima1Nema2'] == 2]
celaStatistika("Nema odgovora - Jedna Strana", nemaOdg[nemaOdg['FormatUpitnika'] == 1]['interviewtime'])
celaStatistika("Nema odgovora - Slajdovi", nemaOdg[nemaOdg['FormatUpitnika'] == 2]['interviewtime'])
