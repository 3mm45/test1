import pandas as pd

from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency


# Učitaj podatke
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

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
