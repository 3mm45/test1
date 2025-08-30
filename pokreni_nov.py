
# Sredjivanje matrice
# Učitavanje podataka iz CSV datoteke
# matrica_2 = pd.read_csv('/Users/emmasarok/Desktop/Emma/FAX/MASTER RAD/Gotovi fajlovi/Matrice/MatricaFinodProfeRecodiranaCSV.csv')
# # Pregled prvih redova podataka print(data.head())- prvi red
# print(matrica_2.head(0, ))

# values = df.get(["ipaddr", "uloga"])
# print(values)

# Preimenovanje specifičnih kolona
# df = df.rename(columns={'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie'})
# rename_grid12 = matrica_2.rename(columns={"G1izgled": "U1 Izgled", "G2pocetna": "U2 Početnastr", "G3gde": "U3 Gde naći", "G4nauciti" : "U4 Lakoća učenja", "G5informacije" : "U5 Informacije", "G6Terminologia" : "U6 Terminologia", "G7 predmeti" : "U7 Predmeti", "G8aktuelnosti": "U8 Aktuelnosti", "G9azuriranje" : "U9 Ažurnost", "G10experimenti" : "U10 Experimenti", "G11buduciUce" : "U11 Mogućnost učenja", "G12brzina" : "U12 Brzina"})
# print("\nPosle preimenovanja kolona:")
# print(rename_grid12.head(0, ))
# print(df.head())

# # Provera nedostajućih vrednosti u svakom stupcu - ne razumem sta sam dobila
# print(matrica_2.isnull().sum())

# Izračunavanje osnovnih statistika
# print(matrica_2.describe())
# ptnj = "/Users/emmasarok/Desktop/Emma/FAX/MASTER RAD/Gotovi fajlovi/Matrice/"


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
from scipy.stats import mannwhitneyu

import geoip2.database
from scipy.stats import chi2_contingency
import numpy as np
#from scipy.stats import gkruskal
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import squareform
#
# ptnj = 'D:/Fakultet/Master radovi/Ema Sarok/'
#
#
# # Ucitavanje matrica
#
# matrica1 = pd.DataFrame(pd.read_csv(ptnj + "results-survey155965.csv")) # Slajdovi
# matrica2 = pd.DataFrame(pd.read_csv(ptnj + "results-survey512454.csv")) # Sve jedno
#
# matrica1['formatUp'] = 2
# matrica2['formatUp'] = 1
#
# #### Matrica 1
#
# # Deljenje po hidden
#
# matrica1a = matrica1[matrica1['hidden']<51].copy() # Redom
# matrica1b = matrica1[matrica1['hidden']>50].copy() # Izmesano
# matrica1h = matrica1[matrica1['hidden'].isna()].copy()
#
# # Brisanje praznih kolona koje nisu potrebne
#
# matrica1a = matrica1a.drop(matrica1a.iloc[:, 26:39],axis = 1)
# matrica1b = matrica1b.drop(matrica1b.iloc[:, 13:26],axis = 1)
# matrica1h = matrica1h.drop(matrica1h.iloc[:, 13:26],axis = 1)
#
# # preimenovanje kolona
#
# matrica1b = matrica1b[['id', 'submitdate', 'lastpage', 'startlanguage',  'startdate', 'datestamp', 'ipaddr', 'DemografskePol[SQ001]', 'Demografske[SQ001]', 'zaAnalytics1[SQ001]', 'vreme[SQ001]', 'zaAnalzticsMozda[SQ001]', 'hidden', 'b[SQ001].1', 'b2[SQ001].1', 'b3[SQ001].1', 'b4[SQ001].1', 'b5[SQ001].1', 'b6[SQ001].1', 'b7[SQ001].1', 'b8[SQ001].1', 'b9[SQ001].1', 'b10[SQ001].1', 'b11[SQ001].1', 'b12[SQ001].1', 'b13[SQ001].1', 'textboxPrvi', 'textboxDrugi', 'textboxTreci', 'textboxPrvimali', 'textboxDrugimali', 'textboxTrecimali', 'interviewtime', 'formatUp']]
# matrica1h = matrica1h[['id', 'submitdate', 'lastpage', 'startlanguage',  'startdate', 'datestamp', 'ipaddr', 'DemografskePol[SQ001]', 'Demografske[SQ001]', 'zaAnalytics1[SQ001]', 'vreme[SQ001]', 'zaAnalzticsMozda[SQ001]', 'hidden', 'b[SQ001].1', 'b2[SQ001].1', 'b3[SQ001].1', 'b4[SQ001].1', 'b5[SQ001].1', 'b6[SQ001].1', 'b7[SQ001].1', 'b8[SQ001].1', 'b9[SQ001].1', 'b10[SQ001].1', 'b11[SQ001].1', 'b12[SQ001].1', 'b13[SQ001].1', 'textboxPrvi', 'textboxDrugi', 'textboxTreci', 'textboxPrvimali', 'textboxDrugimali', 'textboxTrecimali', 'interviewtime', 'formatUp']]
#
# old_column_names = ['b[SQ001].1', 'b2[SQ001].1', 'b3[SQ001].1', 'b4[SQ001].1',
#        'b5[SQ001].1', 'b6[SQ001].1', 'b7[SQ001].1', 'b8[SQ001].1',
#        'b9[SQ001].1', 'b10[SQ001].1', 'b11[SQ001].1', 'b12[SQ001].1',
#        'b13[SQ001].1']
# new_column_names = ['b[SQ001]','b2[SQ001]','b3[SQ001]','b4[SQ001]',
#       'b5[SQ001]','b6[SQ001]','b7[SQ001]','b8[SQ001]',
#       'b9[SQ001]','b10[SQ001]','b11[SQ001]','b12[SQ001]',
#       'b13[SQ001]']
#
# column_rename_mapping = {old: new for old, new in zip(old_column_names, new_column_names)}
# matrica1b.rename(columns=column_rename_mapping, inplace=True)
# matrica1h.rename(columns=column_rename_mapping, inplace=True)
#
# del old_column_names
# del new_column_names
# del column_rename_mapping
#
# ### Matrica 2 - grid
#
# # Deljenje po hidden
#
# matrica2a = matrica2[matrica2['hidden']<51].copy()
# matrica2b = matrica2[matrica2['hidden']>50].copy()
#
# # Brisanje praznih kolona koje nisu potrebne
#
# matrica2a = matrica2a.drop(matrica2a.iloc[:, 26:39],axis = 1)
# matrica2b = matrica2b.drop(matrica2b.iloc[:, 13:26],axis = 1)
#
# # preimenovanje kolona
#
# old_column_names = ['MatrixVSsingle[SQ001]', 'MatrixVSsingle[SQ002]',
#        'MatrixVSsingle[SQ003]', 'MatrixVSsingle[SQ004]',
#        'MatrixVSsingle[SQ005]', 'MatrixVSsingle[SQ006]',
#        'MatrixVSsingle[SQ007]', 'MatrixVSsingle[SQ008]',
#        'MatrixVSsingle[SQ009]', 'MatrixVSsingle[SQ010]',
#        'MatrixVSsingle[SQ011]', 'MatrixVSsingle[SQ012]',
#        'MatrixVSsingle[SQ013]']
# old_column_names2 = ['MatrixVSsingleDruga[SQ001]', 'MatrixVSsingleDruga[SQ002]',
#        'MatrixVSsingleDruga[SQ003]', 'MatrixVSsingleDruga[SQ004]',
#        'MatrixVSsingleDruga[SQ005]', 'MatrixVSsingleDruga[SQ006]',
#        'MatrixVSsingleDruga[SQ007]', 'MatrixVSsingleDruga[SQ008]',
#        'MatrixVSsingleDruga[SQ009]', 'MatrixVSsingleDruga[SQ010]',
#        'MatrixVSsingleDruga[SQ011]', 'MatrixVSsingleDruga[SQ012]',
#        'MatrixVSsingleDruga[SQ013]']
# new_column_names = ['b[SQ001]','b2[SQ001]','b3[SQ001]','b4[SQ001]',
#       'b5[SQ001]','b6[SQ001]','b7[SQ001]','b8[SQ001]',
#       'b9[SQ001]','b10[SQ001]','b11[SQ001]','b12[SQ001]',
#       'b13[SQ001]']
#
# column_rename_mapping = {old: new for old, new in zip(old_column_names, new_column_names)}
# matrica2a.rename(columns=column_rename_mapping, inplace=True)
#
# column_rename_mapping = {old: new for old, new in zip(old_column_names2, new_column_names)}
# matrica2b.rename(columns=column_rename_mapping, inplace=True)
#
# del old_column_names
# del old_column_names2
# del new_column_names
# del column_rename_mapping
#
# matrica2b = matrica2b[['id', 'submitdate', 'lastpage', 'startlanguage', 'startdate', 'datestamp', 'ipaddr', 'DemografskePol[SQ001]', 'Demografske[SQ001]', 'zaAnalytics1[SQ001]', 'vreme[SQ001]', 'zaAnalzticsMozda[SQ001]', 'hidden', 'b[SQ001]', 'b2[SQ001]', 'b3[SQ001]', 'b4[SQ001]', 'b5[SQ001]', 'b6[SQ001]', 'b7[SQ001]', 'b8[SQ001]', 'b9[SQ001]', 'b10[SQ001]', 'b11[SQ001]', 'b12[SQ001]', 'b13[SQ001]', 'textboxPrvi', 'textboxDrugi', 'textboxTreci', 'textboxPrvimali', 'textboxDrugimali', 'textboxTrecimali', 'interviewtime','formatUp']]
#
# # Spajanje svih matrica
#
# matricaFin = pd.concat([matrica1a, matrica1b, matrica1h, matrica2a, matrica2b], ignore_index=True)
# matricaFin.dropna(subset=['DemografskePol[SQ001]'], inplace=True)
#
# # Kratki nazivi
#
# old_column_names = ['DemografskePol[SQ001]', 'Demografske[SQ001]',
#        'zaAnalytics1[SQ001]', 'vreme[SQ001]', 'zaAnalzticsMozda[SQ001]',
#        'hidden', 'b[SQ001]', 'b2[SQ001]', 'b3[SQ001]', 'b4[SQ001]',
#        'b5[SQ001]', 'b6[SQ001]', 'b7[SQ001]', 'b8[SQ001]', 'b9[SQ001]',
#        'b10[SQ001]', 'b11[SQ001]', 'b12[SQ001]', 'b13[SQ001]', 'textboxPrvi',
#        'textboxDrugi', 'textboxTreci', 'textboxPrvimali', 'textboxDrugimali',
#        'textboxTrecimali', 'interviewtime', 'formatUp']
# new_column_names = ['pol', 'uloga',
#        'cesto', 'vreme', 'opsti',
#        'hidden', 'g01', 'g02', 'g03', 'g04',
#        'g05', 'g06', 'g07', 'g08', 'g09',
#        'g10', 'g11', 'g12', 'g13', 'txt01',
#        'txt02', 'txt03', 'txt01mali', 'txt02mali',
#        'txt03mali', 'traj', 'formatUp']
#
# column_rename_mapping = {old: new for old, new in zip(old_column_names, new_column_names)}
# matricaFin.rename(columns=column_rename_mapping, inplace=True)
#
# # zamene vrednosti str -> num
#
# matricaFin.replace('Uopšte se ne slažem', '1', inplace=True)
# matricaFin.replace('Uglavnom se ne slažem', '2', inplace=True)
# matricaFin.replace('Neodlučan/na sam', '3', inplace=True)
# matricaFin.replace('Nedolučan/na sam', '3', inplace=True)
# matricaFin.replace('Potpuno se slažem', '5', inplace=True)
# matricaFin.replace('Uglavnom se slažem', '4', inplace=True)
# matricaFin.replace('uglavnom se slažem', '4', inplace=True)
# matricaFin.replace('Ugalvnom se ne slažem', '2', inplace=True)
# matricaFin.replace('Uglavanom se ne slažem', '2', inplace=True)
#
#
# matricaFin['pol'] = matricaFin['pol'].str.replace('Ženski', '2')
# matricaFin['pol'] = matricaFin['pol'].str.replace('Muški', '1')
# matricaFin['pol'] = matricaFin['pol'].astype(int)
#
#
# matricaFin['uloga'] = matricaFin['uloga'].str.replace('Budući student Odseka', '1')
# matricaFin['uloga'] = matricaFin['uloga'].str.replace('Student Odseka', '2')
# matricaFin['uloga'] = matricaFin['uloga'].str.replace('Nastavnik Odseka', '3')
# matricaFin['uloga'] = matricaFin['uloga'].str.replace('Drugo', '4')
# # ?
# matricaFin['uloga'] = matricaFin['uloga'].fillna(0)
# matricaFin['uloga'] = matricaFin['uloga'].astype(int)
# matricaFin['uloga'] = matricaFin['uloga'].replace(0, np.nan)
#
#
# matricaFin.replace('Barem jednom nedeljno', '4', inplace=True)
# matricaFin.replace('Nekoliko puta mesečno', '3', inplace=True)
# matricaFin.replace('Nekoliko puta u semestru', '2', inplace=True)
# matricaFin.replace('Ovo mi je prvi put', '1', inplace=True)
#
# matricaFin['cesto'] = matricaFin['cesto'].fillna(0)
# matricaFin['cesto'] = matricaFin['cesto'].astype(int)
# matricaFin['cesto'] = matricaFin['cesto'].replace(0, np.nan)
#
#
# matricaFin['vreme'] = matricaFin['vreme'].str.replace('Manje od 5 minuta', '1')
# matricaFin['vreme'] = matricaFin['vreme'].str.replace('Između 5 i 15 minuta', '2')
# matricaFin['vreme'] = matricaFin['vreme'].str.replace('Između 15 i 30 minuta', '3')
# matricaFin['vreme'] = matricaFin['vreme'].str.replace('Duže od 30 minuta', '4')
#
# matricaFin['vreme'] = matricaFin['vreme'].fillna(0)
# matricaFin['vreme'] = matricaFin['vreme'].astype(int)
# matricaFin['vreme'] = matricaFin['vreme'].replace(0, np.nan)
#
#
# matricaFin['opsti'] = matricaFin['opsti'].str.replace('Veoma negativan', '1')
# matricaFin['opsti'] = matricaFin['opsti'].str.replace('Uglavnom negativan', '2')
# matricaFin['opsti'] = matricaFin['opsti'].str.replace('Neutralan', '3')
# matricaFin['opsti'] = matricaFin['opsti'].str.replace('Uglavnom pozitivan', '4')
# matricaFin['opsti'] = matricaFin['opsti'].str.replace('Veoma pozitivan', '5')
#
# matricaFin['opsti'] = matricaFin['opsti'].fillna(0)
# matricaFin['opsti'] = matricaFin['opsti'].astype(int)
# matricaFin['opsti'] = matricaFin['opsti'].replace(0, np.nan)
#
# matricaFin.drop(['ipaddr', 'submitdate', 'lastpage', 'startlanguage', 'startdate', 'datestamp'], axis = 1, inplace=True)
#
# grid = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12', 'g13']
# for x in grid:
#   matricaFin[x] = matricaFin[x].fillna(0)
#   matricaFin[x] = matricaFin[x].astype(int)
#   matricaFin[x] = matricaFin[x].replace(0, np.nan)

# matricaFin.to_csv(ptnj + '/matricaFin.csv', index=False)




print("#1 - Odavde pocinje analize")
# Odavde pocinje analize

# ptnj = "/Users/emmasarok/Desktop/Emma/FAX/MASTER RAD/Gotovi fajlovi/Matrice/"
# matrica = pd.DataFrame(pd.read_csv( ptnj + "matricaFin.csv"))

matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))


# EFA

from factor_analyzer import FactorAnalyzer
from factor_analyzer import utils
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import calculate_kmo
from tabulate import tabulate
import csv

matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

grid = matrica [['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']


form = []
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)].copy())
for x in range(0, 4):
  form[x].drop(["formatUp", "hidden"], axis = 1, inplace=True)

faktor = grid.copy()
# Izbaci prazna polja
faktor.dropna(inplace=True)
#Compute the Bartlett sphericity test
chi_square_value,p_value=calculate_bartlett_sphericity(faktor)
chi_square_value, p_value
#Calculate the Kaiser-Meyer-Olkin criterion for items and overall
kmo_all,kmo_model=calculate_kmo(faktor)
kmo_model

print("#2")
#Korelacija matrica
#Compute correlation with `other` Series, "kendall" method
kormat = faktor.corr("kendall")
kormat.to_csv('/Users/emmasarok/kormat.csv', index=False)

loadings = []
evs = []
brf = [3, 3, 2, 3]

print("#3")
# Napravi listu rezultata za eigen values
for x in range(0, 4):
  fa = FactorAnalyzer(rotation='varimax', method='principal', impute = "drop", n_factors=brf[x])
  fa.fit(form[x])
  loadings.append(fa.loadings_)
  ev, v = fa.get_eigenvalues()
  evs.append(ev)

evs[0]
evs[1]
evs[2]
evs[3]
fa.loadings_

loadings[2]

print("#4 - PLOT za 4 matrice")
plt.clf()
# PLOT za 4 matrice
for x in range(0, 4):
  plt.plot(range(1,form[x].shape[1]+1),evs[x], '-o')
plt.title('Scree plot')
plt.xlabel('Faktori')
plt.ylabel('Ajgen vrednosti')
plt.grid()
classes = ['Jedna strana - Redosled A', 'Jedna strana - Redosled B', 'Slajdovi - Redosled A', 'Slajdovi - Redosled B']
plt.legend(labels=classes)
plt.show()

x = 3
loadings[x] = loadings[x].astype(str)
loadings[x] = np.insert(loadings[x], 0, vars, axis=1)
#loadings[x] = np.insert(loadings[x], 13, utils.smc(fa.corr_, sort=False), axis=1)
loadings[x] = np.insert(loadings[x], 0, ["", "F1", "F2", "F3"], axis=0)

with open("/Users/emmasarok/loadings"+str(x)+".csv","w+", newline='') as my_csv:
  csvWriter = csv.writer(my_csv,delimiter=',')
  csvWriter.writerows(loadings[x])

# Korelacione matrice
print("#5 Korelacione matrice")

matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

grid = matrica [['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]

form = []
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)].copy())


# def cronbach_alpha(df):
#     # Broj stavki
#     k = df.shape[1]
#     # Variance for each column
#     variances = df.var(axis=0, ddof=1)
#     # Total variance
#     total_variance = df.sum(axis=1).var(ddof=1)
#     # Cronbach's alpha formula
#     alpha = (k / (k - 1)) * (1 - (variances.sum() / total_variance))
#     return alpha
#
# # Izračunajmo Cronbachovu Alfu za svaki format
# alpha_format0 = cronbach_alpha(form[0])
# alpha_format1 = cronbach_alpha(form[1])
# alpha_format2 = cronbach_alpha(form[2])
# alpha_format3 = cronbach_alpha(form[3])
#
# print(f'Cronbachova Alfa za Format o: {alpha_format0}')
# print(f'Cronbachova Alfa za Format 1: {alpha_format1}')
# print(f'Cronbachova Alfa za Format 2: {alpha_format2}')
# print(f'Cronbachova Alfa za Format 3: {alpha_format3}')
# print(fIzgleda da su vrednosti Cronbachove Alfe za vaše formate prilično niske, a za Format 0 čak i negativna. Ovo može ukazivati na nekoliko problema:
# Negativna Alfa (Format 0): Kada je Cronbachova Alfa negativna, to obično ukazuje na vrlo nisku međusobnu povezanost stavki, ili čak na prisustvo negativnih kovarijansi između nekih stavki. Mogući razlozi uključuju loše formulisane stavke, stavke koje ne mere isti konstrukt, ili čak greške u podacima.
# Niske Alfe (Formati 1, 2, i 3): Niske vrednosti Alfe (ispod 0.7) sugerišu da stavke unutar formata možda nisu dobro povezane ili ne mere dosledno isti konstrukt. Ovo može biti posledica nekonzistentnog odgovaranja ispitanika, heterogenih stavki koje mere različite aspekte, ili jednostavno lošeg dizajna upitnika.)

for x in range(0, 4):
  form[x].drop(["formatUp", "hidden"], axis = 1, inplace=True)

def upper(df):
  try:
      assert(type(df)==np.ndarray)
  except:
      if type(df)==pd.DataFrame:
          df = df.values
      else:
          raise TypeError('Must be np.ndarray or pd.DataFrame')
  mask = np.triu_indices(df.shape[0], k=1)
  return df[mask]

print("#6 - Bootstrap")
def bootstrap(mat1, mat2):
  df_1 = mat1.sample(frac = 0.6)
  df_2 = mat2.sample(frac = 0.6)
  corrM1 = df_1.corr(method='kendall')
  corrM2 = df_2.corr(method='kendall')
  res = stats.kendalltau(upper(corrM1), upper(corrM2))
  return res.statistic

for a in range (0, 4):
  for b in range (a, 4):
    bootr = []
    # iterate 1000 times
    for x in range (0, 1000):
    # for x in range (0, 200):
      bootr.append(bootstrap(form[a], form[b]))
    # print(a, b, np.percentile(bootr, 2.5), np.mean(bootr), np.percentile(bootr, 97.5))
    print(a, b,
        f"{np.percentile(bootr, 2.5):.2f}",
        f"{np.mean(bootr):.2f}",
        f"{np.percentile(bootr, 97.5):.2f}")
    bootr = []
    # iterate 1000 times
    # for x in range (0, 1000):
    for x in range (0, 1000):
      bootr.append(bootstrap(form[a], form[b]))
    # print(a, b, np.percentile(bootr, 2.5), np.mean(bootr), np.percentile(bootr, 97.5))
    print(a, b,
        f"{np.percentile(bootr, 2.5):.2f}",
        f"{np.mean(bootr):.2f}",
        f"{np.percentile(bootr, 97.5):.2f}")
    print("***************")

corrM1 = form[0].corr(method='spearman')
corrM2 = form[1].corr(method='spearman')
corrM3 = form[2].corr(method='spearman')
corrM4 = form[3].corr(method='spearman')

print("#7# Heatmaps")
plt.clf()
f,axes = plt.subplots(2,2, figsize=(10,10))
sns.set_style("white")
sns.heatmap(corrM1, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
axes[0, 0].set_title('Jedna strana - Redosled A')
sns.heatmap(corrM2, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
axes[0, 1].set_title('Jedna strana - Redosled B')
sns.heatmap(corrM3, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
axes[1, 0].set_title('Slajdovi - Redosled A')
sns.heatmap(corrM4, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
axes[1, 1].set_title('Slajdovi - Redosled B')

# Concatenate dataframes
matrica_Jedna_strana = pd.concat([corrM1, corrM2], ignore_index=True)
matrica_Slajdovi = pd.concat([corrM3, corrM4], ignore_index=True)
matrica_RedosledA = pd.concat([corrM1, corrM3], ignore_index=True)
matrica_RedosledB = pd.concat([corrM2, corrM4], ignore_index=True)

# # Create a 2x2 grid of subplots
# fig, axes = plt.subplots(2, 2, figsize=(15, 15))
#
# # Plot each heatmap on a different subplot
# sns.heatmap(matrica_Jedna_strana, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
# axes[0, 0].set_title('Jedna strana')
#
# sns.heatmap(matrica_Slajdovi, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
# axes[0, 1].set_title('Slajdovi')
#
# sns.heatmap(matrica_RedosledA, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
# axes[1, 0].set_title('Grid pitanja - Redosled A')
#
# sns.heatmap(matrica_RedosledB, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
# axes[1, 1].set_title('Grid pitanja - Redosle B')

# Adjust layout
plt.tight_layout()
plt.show()


print("#8-Duzina odgovora")


# Sredjivanje martrice

matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))
# autlajieri
matrica = matrica[matrica['traj'] < 3600]

# Razlike u duzini popunjavanja

# Dugi ili kratki tekst
matrica['dugi'] = matrica.apply(lambda x: 1 if x['hidden'] < 51 else 2, axis=1)
# Ukupna duzina unetog teksta
matrica['duzodg'] = 0
matrica['duzodgm'] = 0
matrica['duzodgv'] = 0
matrica['txt01'] = matrica['txt01'].astype(str)
matrica['txt02'] = matrica['txt02'].astype(str)
matrica['txt03'] = matrica['txt03'].astype(str)
matrica['txt01mali'] = matrica['txt01mali'].astype(str)
matrica['txt02mali'] = matrica['txt02mali'].astype(str)
matrica['txt03mali'] = matrica['txt03mali'].astype(str)
matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt01']), axis=1)
matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt02']), axis=1)
matrica['duzodgv'] = matrica.apply(lambda x: x['duzodgv'] + len(x['txt03']), axis=1)
matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt01mali']), axis=1)
matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt02mali']), axis=1)
matrica['duzodgm'] = matrica.apply(lambda x: x['duzodgm'] + len(x['txt03mali']), axis=1)
matrica['duzodgv'] = matrica.apply(lambda x: 0 if x['duzodgv'] < 10 else x['duzodgv'], axis=1)
matrica['duzodgm'] = matrica.apply(lambda x: 0 if x['duzodgm'] < 10 else x['duzodgm'], axis=1)
matrica['duzodg'] = matrica['duzodgv'] + matrica['duzodgm']
# Da li je odgovorio na barem jedan txtbox
matrica['imaot'] = matrica.apply(lambda x: 1 if x['duzodg'] > 0 else 0, axis=1)
# Da li je odgovorio na sve iz grida
matrica['imasve'] = matrica.iloc[:,7:20].sum(axis=1,min_count=13)

# Samo sa svim odogovrima
matrica = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()
samosvi = matrica[(~matrica['imasve'].isna()) & (matrica['vreme'] > 1)].copy()
matrica = matrica[(~matrica['imasve'].isna())].copy()
samosvi = matrica.copy()

samosvi = samosvi[samosvi['duzodg'] > 0]

fig = interaction_plot(samosvi['formatUp'], samosvi['dugi'], samosvi['duzodg'], colors=['red','blue'], markers=['D','^'], ms=10)
plt.show()

model = ols('duzodg ~ C(formatUp) + C(dugi) + C(formatUp):C(dugi)', data=samosvi).fit()
sm.stats.anova_lm(model, typ=3)
model.summary()


matrica[(matrica['formatUp'] == 1) & (matrica['duzodg'] > 0)]['duzodg'].mean()
matrica[(matrica['formatUp'] == 2) & (matrica['duzodg'] > 0)]['duzodg'].mean()

form1 = samosvi[(samosvi['formatUp'] == 1) & (samosvi['duzodg'] > 0)].copy()
form2 = samosvi[(samosvi['formatUp'] == 2) & (samosvi['duzodg'] > 0)].copy()

form1 = samosvi[(samosvi['dugi'] == 2) & (samosvi['formatUp'] == 2)].copy()
form2 = samosvi[(samosvi['dugi'] == 2) & (samosvi['formatUp'] == 1)].copy()

form3 = samosvi[(samosvi['duzodg'] > 0)].copy()

form1['duzodg'].mean()
form2['duzodg'].mean()

U1, p = mannwhitneyu(form1['duzodg'], form2['duzodg'])
print(U1, p)

from scipy import stats
stats.ttest_ind(form1['duzodg'], form2['duzodg'], equal_var=False)

U1, p = mannwhitneyu(form3['duzodgm'], form3['duzodgv'])
print(U1, p)

form1 = matrica[(matrica['imaot'] == 0) & (matrica['formatUp'] == 1)].copy()
form2 = matrica[(matrica['imaot'] == 0) & (matrica['formatUp'] == 2)].copy()

U1, p = mannwhitneyu(form1['interviewtime'], form2['interviewtime'])
print(U1, p)

proba = pd.crosstab(index=samosvi['imaot'], columns=samosvi['zaAnalytics1[SQ001]'])
stat, p, dof, expected = chi2_contingency(proba)
print("p value is " + str(p))


fig = interaction_plot(matrica['formatUp'], matrica['imaot'], matrica['interviewtime'], colors=['red','blue'], markers=['D','^'], ms=10)
plt.show()

nesub = matrica[matrica['submitdate'].isna()].copy()

print("9 -552red")
# CFA

from semopy import Model
from semopy import Optimizer
from semopy import gather_statistics
from semopy.inspector import inspect
from semopy import calc_stats

model_spec = '''
    F1 =~ g02 + g03 + g04 + g11
    F2 =~ g01 + g05 + g12
    F3 =~ g06 + g07 + g08 + g09
'''

model = Model(model_spec)
# model.load_dataset(faktor)
opt = Optimizer(model)
ofv = opt.optimize()
inspect(opt)

stat = gather_statistics(opt)
stat
print(stat.rmsea)


U1, p = mannwhitneyu(form1['interviewtime'], form2['interviewtime'])
print(U1, p)


print("Emma")
# Emma analize za hipoteze druge grupe:

# Ispitivanje inter-ajtem korelacija (IAK) na druga grupa pitanja: Dizajn rešetke/matrične tabele u odnosu na dizajn jedne stvake po stranici
# (H5a) Prosečni (IAK)-ovi su veći u dizajnu matrične tabele nego u dizajnu sa jednom stvakom po stranici.
# (H5b) Svaki pojedinačni (IAK) je veći u dizajnu matrične tabele nego u dizajnu sa jednom stvakom po stranici
# (H5b2) Vizuelne karakteristike formata upitnika (A ili B)  imaju uticaj na sklonosti davanju negativnih/pozitivnih ocena.
# (H5c) Korelacija koeficijenta stavki u matričnom formatu se ne razlikuje značajno kod dva različitog racporeda pitanja (odnosno format a i format b matricnih pitanja).
# Prosečni (IAK)-ovi su veći kod rasporeda pitanja redom u odnosu na raspored pitanja pomešano.
# Vizuelne karakteristike različitog racporeda pitanja imaju uticaj na sklonosti davanju negativnih/pozitivnih ocena.


# IAK za 2 formata (A_1strana i  B_slajdovi)
import pandas as pd

matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Uncomment these lines if you want to filter the data
# matrica = matrica[matrica['uloga'] == 2] # Samo studenti
# matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

# Selecting specific columns for analysis
grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]


# Creating subsets based on formatUp
form_A_1strana = grid[grid['formatUp'] == 1].copy()
form_B_slajdovi = grid[grid['formatUp'] == 2].copy()

# Calculating inter-item correlations for each format
format_A_correlations = form_A_1strana.corr()
format_B_correlations = form_B_slajdovi.corr()

# Calculating average inter-item correlations for each format
average_iak_format_A = format_A_correlations.mean().mean()
average_iak_format_B = format_B_correlations.mean().mean()

print("Prosečni IAK za format A_1strana:", average_iak_format_A)
print("Prosečni IAK za format B_slajdovi:", average_iak_format_B)
print("formatIAK")



# Creating subsets based on Hidden (redom i pomesano)
form_redom = grid[grid['hidden'] <= 50].copy()
form_pomesano = grid[grid['hidden'] > 50].copy()

# Calculating inter-item correlations for each hidden
format_redom_correlations = form_redom.corr()
format_pomesano_correlations = form_pomesano.corr()


# Calculating average inter-item correlations for each hidden
average_iak_format_redom = format_redom_correlations.mean().mean()
average_iak_format_pomesano = format_pomesano_correlations.mean().mean()

print("Prosečni IAK za format redom:", average_iak_format_redom)
print("Prosečni IAK za format pomesano:", average_iak_format_pomesano)
print("hiddenIAK")
print("")


import pandas as pd
from scipy.stats import pearsonr

# Reading the data from CSV
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Uncomment these lines if you want to filter the data
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

# Selecting specific columns for analysis
grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']

# Creating subsets based on hidden
form_redom = grid[grid['hidden'] <= 50].copy()
form_pomesano = grid[grid['hidden'] > 50].copy()

# Izračunavanje korelacija stavki za raspored redom
correlations_redom = form_redom[vars].corr()

# Izračunavanje korelacija stavki za raspored pomesano
correlations_pomesano = form_pomesano[vars].corr()

# Izravnavanje korelacionih matrica u jednovektorske forme
correlations_redom_flat = correlations_redom.values.flatten()
correlations_pomesano_flat = correlations_pomesano.values.flatten()

# Izračunavanje korelacionog koeficijenta Pearson-a između dve korelacije
pearson_corr, _ = pearsonr(correlations_redom_flat, correlations_pomesano_flat)
# for var, results in pearson_corr.items():
#     print(f"p-value: {results['p_val']:.4f}")
#     if results['p_val'] < 0.05:
#         print("P je p < 0.05, smatramo da postoji statistički značajna razlika između grupa za tu stavku.")
#     else:
#         print("P je p >= 0.05, ne možemo zaključiti da postoji statistički značajna razlika.")
# print("p-vrednost < 0.05 ukazuje na statistički značajnu razliku između grupa za tu stavku.\n")
#
print(f"Pearsonov korelacioni koeficijent između dva rasporeda pitanja: {pearson_corr}")

print("")

# #
# import pandas as pd
# from scipy.stats import pearsonr
#
# # Reading the data from CSV
# matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")
#
# matrica = matrica[matrica['uloga'] == 2] # Samo studenti
# matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju
#
# # Selecting specific columns for analysis
# grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
# variables = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']
#
# # Creating subsets based on hidden
# form_redom = grid[grid['hidden'] <= 50].copy()
# form_pomesano = grid[grid['hidden'] > 50].copy()
#
# # Calculating correlations for items in ordered arrangement
# correlations_redom = form_redom[variables].corr()
#
# # Calculating correlations for items in mixed arrangement
# correlations_pomesano = form_pomesano[variables].corr()
#
# # Flattening correlation matrices into one-dimensional arrays
# correlations_redom_flat = correlations_redom.values.flatten()
# correlations_pomesano_flat = correlations_pomesano.values.flatten()
#
# # Calculating Pearson correlation coefficient between two correlation arrays
# pearson_corr, _ = pearsonr(correlations_redom_flat, correlations_pomesano_flat)
#
# # Outputting results
# for p_val in pearson_corr:
#     print(f"p-value: {p_val:.4f}")
#     if p_val < 0.05:
#         print("P je p < 0.05, smatramo da postoji statistički značajna razlika između grupa za tu stavku.")
#     else:
#         print("P je p >= 0.05, ne možemo zaključiti da postoji statistički značajna razlika.")
# print("p-vrednost < 0.05 ukazuje na statistički značajnu razliku između grupa za tu stavku.\n")
#
# print(f"Pearsonov korelacioni koeficijent između dva rasporeda pitanja: {pearson_corr}\n")
#
#
print("T-Test za uporedjenje hidden(grupe redom i pomesano)")

# T-Test za uporedjenje hidden(grupe redom i pomesano)
import pandas as pd
from scipy.stats import ttest_ind

# Reading the data from CSV
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Filtriranje podataka
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

# Selecting specific columns for analysis
grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']

# Creating subsets based on hidden
form_redom = grid[grid['hidden'] <= 50].copy()
form_pomesano = grid[grid['hidden'] > 50].copy()

# Izračunavanje t-testa za svaku stavku između dva rasporeda
t_test_results = {}
for var in vars:
    t_stat, p_val = ttest_ind(form_redom[var].dropna(), form_pomesano[var].dropna())
    t_test_results[var] = {'t_stat': t_stat, 'p_val': p_val}

# Ispisivanje rezultata t-testa

for var, results in t_test_results.items():
    print(f"{var} - t-statistic: {results['t_stat']:.4f}, p-value: {results['p_val']:.4f}")
    if results['p_val'] < 0.05:
        print("P je p < 0.05, smatramo da postoji statistički značajna razlika između grupa za tu stavku.")
    else:
        print("P je p >= 0.05, ne možemo zaključiti da postoji statistički značajna razlika.")

print("\nT-test rezultati su izračunati za svaku stavku između dva rasporeda (redom i pomesano).")
print("p-vrednost < 0.05 ukazuje na statistički značajnu razliku između grupa za tu stavku.\n")
print("T-test")
print("")




# Anova za hidden
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#
# # Reading the data from CSV
# matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")
#
# # Filtriranje podataka
# matrica = matrica[matrica['uloga'] == 2] # Samo studenti
# matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju
#
# # Selecting specific columns for analysis
# grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
# vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']
#
# form_redom = grid[grid['hidden'] <= 50].copy()
# form_pomesano = grid[grid['hidden'] > 50].copy()
#
# # Creating a long-format DataFrame for ANOVA
# long_format_data = grid.melt(id_vars=['form_redom', 'form_pomesano'], value_vars=vars, var_name='question', value_name='score')
#
# # Defining the model
# model = ols('score ~ C(hidden) + C(question) + C(hidden):C(question)', data=long_format_data).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
#
# # Ispisivanje rezultata ANOVA
# print(anova_table)
# print("ANOVA")
print("")


# IAK - za sva cetri formata
import pandas as pd

# Reading the data from CSV
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Uncomment these lines if you want to filter the data
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

# Selecting specific columns for analysis
grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']

# Creating subsets based on conditions
form = []
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] <= 50)][vars].copy())
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)][vars].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] <= 50)][vars].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)][vars].copy())

# Calculating inter-item correlations for each format
correlations = [f.corr() for f in form]

# Calculating average inter-item correlations for each format
average_iak_format_0 = correlations[0].mean().mean()
average_iak_format_1 = correlations[1].mean().mean()
average_iak_format_2= correlations[2].mean().mean()
average_iak_format_3 = correlations[3].mean().mean()

print("Prosečni IAK za format 0:", average_iak_format_0)
print("Prosečni IAK za format 1:", average_iak_format_1)
print("Prosečni IAK za format 2:", average_iak_format_2)
print("Prosečni IAK za format 3:", average_iak_format_3)

print("H01-Ne postoji statistički značajna razlika u vremenu popunjavanja formata upitnika sa svim pitanjima na jednoj stranici formata u kome je svako pitanje na zasebnoj stranici. ")
## # (H01) Ne postoji statistički značajna razlika u vremenu popunjavanja dva formata upitnika: A (sva pitanja na jednoj stranici) i B (svaka pitanja na zasebnoj stranici) .
import pandas as pd
from scipy.stats import ttest_ind

# Učitavanje podataka iz CSV datoteke
df = pd.read_csv('/Users/emmasarok/matricaFin.csv')

# Prikaz prvih nekoliko redova dataframe-a radi provere
print(df.head())

# Razdvajanje podataka na osnovu formata upitnika A i B
format_A = df[df['format_upitnika'] == 'A']['vreme_popunjavanja']
format_B = df[df['format_upitnika'] == 'B']['vreme_popunjavanja']

# Izvršavanje t-testa za nezavisne uzorke
t_statistic, p_value = ttest_ind(format_A, format_B)

# Prikaz rezultata
print("T-statistika:", t_statistic)
print("P-vrednost:", p_value)

# Tumačenje rezultata
alpha = 0.05
if p_value > alpha:
    print("Nema dovoljno dokaza da se odbaci H01.")
else:
    print("Postoje statistički značajne razlike u vremenu popunjavanja između formata upitnika A i B.")