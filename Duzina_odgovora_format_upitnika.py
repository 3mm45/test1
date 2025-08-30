# Kod od profesora
import numpy as np
# import numpy as np
# from scipy.stats import chi2_contingency
# # Sredjivanje martrice
#
# matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))
# # autlajieri
# matrica = matrica[matrica['traj'] < 3600]
#
# # Razlike u duzini popunjavanja
#
# Manji(1) == hidden < 51, Veci(2) == hidden > 50
# # Dugi ili kratki tekst
# matrica['dugi'] = matrica.apply(lambda x: 1 if x['hidden'] < 51 else 2, axis=1)
# # Ukupna duzina unetog teksta
# matrica['duzodg'] = 0
# matrica['duzodgm'] = 0
# matrica['duzodgv'] = 0
# matrica['txt01'] = matrica['txt01'].astype(str)
# matrica['txt02'] = matrica['txt02'].astype(str)
# matrica['txt03'] = matrica['txt03'].astype(str)
# matrica['txt01mali'] = matrica['txt01mali'].astype(str)
# matrica['txt02mali'] = matrica['txt02mali'].astype(str)
# matrica['txt03mali'] = matrica['txt03mali'].astype(str)
# matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt01']), axis=1)
# matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt02']), axis=1)
# matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt03']), axis=1)
# matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt01mali']), axis=1)
# matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt02mali']), axis=1)
# matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt03mali']), axis=1)
# matrica['duzodgv'] = matrica.apply(lambda x: 0 if x['duzodgv'] < 10 else x['duzodgv'], axis=1)
# matrica['duzodgm'] = matrica.apply(lambda x: 0 if x['duzodgm'] < 10 else x['duzodgm'], axis=1)
# matrica['duzodg'] = matrica['duzodgv'] + matrica['duzodgm']
# # Da li je odgovorio na barem jedan txtbox
# matrica['imaot'] = matrica.apply(lambda x: 1 if x['duzodg'] > 0 else 0, axis=1)
# # Da li je odgovorio na sve iz grida
# matrica['imasve'] = matrica.iloc[:,7:20].sum(axis=1,min_count=13)
#
# # Samo sa svim odogovrima
# matrica = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()
# samosvi = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()
# matrica = matrica[(~matrica['imasve'].isna())].copy()
# samosvi = matrica.copy()
#
# samosvi = samosvi[samosvi['duzodg'] > 0]
#
# fig = interaction_plot(samosvi['formatUp'], samosvi['dugi'], samosvi['duzodg'], colors=['red','blue'], markers=['D','^'], ms=10)
# plt.show()
#
# model = ols('duzodg ~ C(formatUp) + C(dugi) + C(formatUp):C(dugi)', data=samosvi).fit()
# sm.stats.anova_lm(model, typ=3)
# model.summary()
#
#
# matrica[(matrica['formatUp'] == 1) & (matrica['duzodg'] > 0)]['duzodg'].mean()
# matrica[(matrica['formatUp'] == 2) & (matrica['duzodg'] > 0)]['duzodg'].mean()
#
# form1 = samosvi[(samosvi['formatUp'] == 1) & (samosvi['duzodg'] > 0)].copy()
# form2 = samosvi[(samosvi['formatUp'] == 2) & (samosvi['duzodg'] > 0)].copy()
#
# form1 = samosvi[(samosvi['dugi'] == 2) & (samosvi['formatUp'] == 2)].copy()
# form2 = samosvi[(samosvi['dugi'] == 2) & (samosvi['formatUp'] == 1)].copy()
#
# form3 = samosvi[(samosvi['duzodg'] > 0)].copy()
#
# form1['duzodg'].mean()
# form2['duzodg'].mean()
#
# U1, p = mannwhitneyu(form1['duzodg'], form2['duzodg'])
# print(U1, p)
#
# from scipy import stats
# stats.ttest_ind(form1['duzodg'], form2['duzodg'], equal_var=False)
#
# U1, p = mannwhitneyu(form3['duzodgm'], form3['duzodgv'])
# print(U1, p)
#
# form1 = matrica[(matrica['imaot'] == 0) & (matrica['formatUp'] == 1)].copy()
# form2 = matrica[(matrica['imaot'] == 0) & (matrica['formatUp'] == 2)].copy()
#
# U1, p = mannwhitneyu(form1['interviewtime'], form2['interviewtime'])
# print(U1, p)
#
# proba = pd.crosstab(index=samosvi['imaot'], columns=samosvi['zaAnalytics1[SQ001]'])
# stat, p, dof, expected = chi2_contingency(proba)
# print("p value is " + str(p))
# print("#8-Duzina odgovora")

#
#
# fig = interaction_plot(matrica['formatUp'], matrica['imaot'], matrica['interviewtime'], colors=['red','blue'], markers=['D','^'], ms=10)
# plt.show()
#
# nesub = matrica[matrica['submitdate'].isna()].copy()
#
# print("9 -521red")
# # CFA
#
# from semopy import Model
# from semopy import Optimizer
# from semopy import gather_statistics
# from semopy.inspector import inspect
# from semopy import calc_stats
#
# model_spec = '''
#     F1 =~ g02 + g03 + g04 + g11
#     F2 =~ g01 + g05 + g12
#     F3 =~ g06 + g07 + g08 + g09
# '''
#
# model = Model(model_spec)
# # model.load_dataset(faktor)
# opt = Optimizer(model)
# ofv = opt.optimize()
# inspect(opt)
#
# stat = gather_statistics(opt)
# stat
# print(stat.rmsea)
#
#
# U1, p = mannwhitneyu(form1['interviewtime'], form2['interviewtime'])
# print(U1, p)



# CHATGPTpoprvka
import pandas as pd
import numpy

import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency


# Učitaj podatke
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Filtriranje outliera
# matrica = matrica[matrica['traj'] < 3600]
# Dugi ili kratki tekst
matrica['dugi'] = matrica['hidden'].apply(lambda x: 1 if x < 51 else 2)

# Inicijalizacija kolona za dužinu teksta
matrica['duzodg'] = 0
matrica['duzodgm'] = 0
matrica['duzodgv'] = 0

# Pretvaranje kolona u stringove
txt_columns = ['txt01', 'txt02', 'txt03', 'txt01mali', 'txt02mali', 'txt03mali']
for col in txt_columns:
    matrica[col] = matrica[col].astype(str)

# Računanje dužine odgovora za velike i male txt kolone

matrica['duzodgv'] = np.maximum(matrica['txt01'].str.len(), matrica['txt02'].str.len(), matrica['txt03'].str.len())
# duzOdgVelikiMean = matrica['duzodgv'].mean()
# duzOdgVelikiMedian = matrica['duzodgv'].median()
# print(f"duzOdgVeliki: mean: {duzOdgVelikiMean}, median: {duzOdgVelikiMedian}")
matrica['duzodgm'] = np.maximum(matrica['txt01mali'].str.len(), matrica['txt02mali'].str.len(), matrica['txt03mali'].str.len())
# duzOdgMaliMean = matrica['duzodgm'].mean()
# duzOdgMaliMedian = matrica['duzodgm'].median()
# print(f"duzOdgMali: mean: {duzOdgMaliMean}, median: {duzOdgMaliMedian}")


# Postavljanje dužine odgovora na 0 ako je manja od 10
matrica['duzodgv'] = matrica['duzodgv'].apply(lambda x: 0 if x < 10 else x)
matrica['duzodgm'] = matrica['duzodgm'].apply(lambda x: 0 if x < 10 else x)

# Ukupna dužina odgovora
matrica['duzodg'] = matrica['duzodgv'] + matrica['duzodgm']

# Da li je odgovorio na barem jedan txtbox
matrica['imaot'] = matrica['duzodg'].apply(lambda x: 1 if x > 0 else 0)

# Da li je odgovorio na sve iz grida (provera validnosti)
matrica['imasve'] = matrica.iloc[:, 7:20].sum(axis=1, min_count=13)

# # Filtriranje samo sa validnim odgovorima
# matrica = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()
# samosvi = matrica[(~matrica['duzodg'].isna()) & (matrica['duzodg'] > 0)].copy()

# Filtrirajte matrica i sačuvajte rezultat
filtered_matrica = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()

# Napravite kopiju za samosvi
samosvi = filtered_matrica.copy()

# Dodatno filtrirajte samosvi prema duzodg
# EMMA
samosvi = samosvi[samosvi['duzodg'] > 30]

# vizuelizacija interakcije
from statsmodels.graphics.factorplots import interaction_plot

# print(samosvi['formatUp'].map(lambda x: str(x)))
print(samosvi['dugi'])

# print(samosvi['duzodg'])

formatUpitnika = {
    1: "Jedna strana",
    2: "Slajdovi"
}
formatStr = samosvi['formatUp'].map(lambda x: formatUpitnika.get(x, "Nepoznato")).array

velicinaTekstBoxa = {
    1: "Manji okvir",
    2: "Veći okvir"
}
samosvi['Veličina okvira za unos odogovora'] = samosvi['dugi'].map(lambda x: velicinaTekstBoxa.get(x, "Nepoznato"))
# duzinaTekstBoxa[] = duzinaTekstBoxa['dugi']

fig = interaction_plot(formatStr, samosvi['Veličina okvira za unos odogovora'], samosvi['duzodg'], colors=['red','blue'], markers=['D','^'], ms=10, ylabel="Prosečna vrednost dužine odgovora", xlabel="Format upitnika")
plt.show()
# ANOVA analiza
model = ols('duzodg ~ C(formatUp) + C(dugi) + C(formatUp):C(dugi)', data=samosvi).fit()
anova_results = sm.stats.anova_lm(model, typ=3)
print("ANOVA")
print(anova_results)
print(model.summary())
print("****")

# Srednje vrednosti dužine odgovora po formatima
form1_mean = matrica[(matrica['formatUp'] == 1) & (matrica['duzodg'] > 0)]['duzodg'].mean()
form2_mean = matrica[(matrica['formatUp'] == 2) & (matrica['duzodg'] > 0)]['duzodg'].mean()
print(f"Prosečna dužina odgovora za format 1: {form1_mean}")
print(f"Prosečna dužina odgovora za format 2: {form2_mean}")

# Definišite form1 i form2 za dužinu odgovora
form1 = samosvi[(samosvi['formatUp'] == 1) & (samosvi['duzodg'] > 0)].copy()
form2 = samosvi[(samosvi['formatUp'] == 2) & (samosvi['duzodg'] > 0)].copy()


# Mann-Whitney U test za poređenje dužine odgovora između formata
U1, p_value = mannwhitneyu(form1['duzodg'], form2['duzodg'])
print(f"Mann-Whitney U test: U1 = {U1}, p = {p_value}")

# T-test za poređenje dužine odgovora između formata
t_stat, p_val = ttest_ind(form1['duzodg'], form2['duzodg'], equal_var=False)
print(f"T-test: t = {t_stat}, p = {p_val}")
# Chi-square test
proba = pd.crosstab(index=samosvi['imaot'], columns=samosvi['zaAnalytics1[SQ001]'])
stat, p, dof, expected = chi2_contingency(proba)
print(f"Chi-square test: p value = {p}")
# SEM analiza
from semopy import Model, Optimizer, gather_statistics

model_spec = '''
    F1 =~ g02 + g03 + g04 + g11
    F2 =~ g01 + g05 + g12
    F3 =~ g06 + g07 + g08 + g09
'''

model = Model(model_spec)
# Pretpostavljam da imate dataset 'faktor' koji se koristi za SEM model, ako nemate potrebno je da ga definišete
# model.load_dataset(faktor)

opt = Optimizer(model)
ofv = opt.optimize()
stats = gather_statistics(opt)
print(stats)
print(f"RMSEA: {stats['rmsea']}")









