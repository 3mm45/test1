import pandas as pd
import pandas as pd
import numpy as np
from scipy import stats

from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency


# Učitaj podatke
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")
# matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/rekodirano_emma2.csv"))

# Filtriranje outliera
matrica = matrica[matrica['traj'] < 3600]
matrica = matrica[matrica['traj'] > 30]

# Definišite form1 i form2 za dužinu odgovora
form1 = matrica[(matrica['formatUp'] == 1)].copy()
form1_mean = form1['traj'].mean()
form1_medijana = form1['traj'].median()

form2 = matrica[(matrica['formatUp'] == 2)].copy()
form2_mean = form2['traj'].mean()
form2_medijana = form1['traj'].median()

print(f"Jedna strana br odgovora: {form1['traj'].size}")
print(f"Jedna strana mean: {form1_mean}")
print(f"Jedna strana medijana: {form1_medijana}")
print()
print(f"Slajdovi br odgovora: {form2['traj'].size}")
print(f"Slajdovi mean: {form2_mean}")
print(f"Jedna strana medijana: {form2_medijana}")
print()

# Mann-Whitney U test za poređenje dužine odgovora između formata
U1, p_value = mannwhitneyu(form1['traj'], form2['traj'])
print(f"Mann-Whitney U test: U1 = {U1}, p = {p_value}")
print()
print("proba")



# Podaci za format "Jedna strana"
form1 = matrica[(matrica['formatUp'] == 1)].copy()

# Podaci za format "Slajdovi"
form2 = matrica[(matrica['formatUp'] == 2)].copy()

# Podaci kada postoji odgovor na otvorena pitanja
form1_ima = form1[form1['O123Ima1Nema2'] == 1]
form2_ima = form2[form2['O123Ima1Nema2'] == 1]

# Podaci kada nema odgovora na otvorena pitanja
form1_nema = form1[form1['O123Ima1Nema2'] == 2]
form2_nema = form2[form2['O123Ima1Nema2'] == 2]

def descriptive_stats(data):
    n = len(data)
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    std_err = stats.sem(data)
    return n, mean, median, std_dev, std_err

# Ukupno - Jedna strana
n1, mean1, median1, std1, se1 = descriptive_stats(form1['traj'])

# Ukupno - Slajdovi
n2, mean2, median2, std2, se2 = descriptive_stats(form2['traj'])

# Otvorena = Ima odgovora - Jedna strana
n3, mean3, median3, std3, se3 = descriptive_stats(form1_ima['traj'])

# Otvorena = Ima odgovora - Slajdovi
n4, mean4, median4, std4, se4 = descriptive_stats(form2_ima['traj'])

# Otvorena = Nema odgovora - Jedna strana
n5, mean5, median5, std5, se5 = descriptive_stats(form1_nema['traj'])

# Otvorena = Nema odgovora - Slajdovi
n6, mean6, median6, std6, se6 = descriptive_stats(form2_nema['traj'])

# Kreirajte DataFrame za rezultate
results = pd.DataFrame({
    'Format pitanja': ['Jedna strana', 'Slajdovi', 'Jedna strana', 'Slajdovi', 'Jedna strana', 'Slajdovi'],
    'Kategorija': ['Ukupno', 'Ukupno', 'Otvorena = Ima odgovora', 'Otvorena = Ima odgovora', 'Otvorena = Nema odgovora', 'Otvorena = Nema odgovora'],
    'N': [n1, n2, n3, n4, n5, n6],
    'Mean': [mean1, mean2, mean3, mean4, mean5, mean6],
    'Median': [median1, median2, median3, median4, median5, median6],
    'Std. Deviation': [std1, std2, std3, std4, std5, std6],
    'Std. Error Mean': [se1, se2, se3, se4, se5, se6]
})

print(results)


