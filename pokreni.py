
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

# Odavde pocinje
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

faktor2 = grid.copy()
faktor = matrica[grid].copy()
# Izbaci prazna polja
faktor.dropna(inplace=True)
#Compute the Bartlett sphericity test
chi_square_value,p_value=calculate_bartlett_sphericity(faktor)
chi_square_value, p_value
#Calculate the Kaiser-Meyer-Olkin criterion for items and overall
kmo_all,kmo_model=calculate_kmo(faktor)
kmo_model

#Korelacija matrica
#Compute correlation with `other` Series, "kendall" method
kormat = faktor.corr("kendall")
kormat.to_csv(ptnj + '/kormat.csv', index=False)

loadings = []
evs = []
brf = [3, 3, 2, 3]

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

plt.clf()
# PLOT za 4 matrice
for x in range(0, 4):
  plt.plot(range(1,form[x].shape[1]+1),evs[x], '-o')
plt.title('Scree plot')
plt.xlabel('Faktori')
plt.ylabel('Ajgen vrednosti')
plt.grid()
classes = ['Jedna strana - redom', 'Jedna strana - pomešano', 'Slajdovi - redom', 'Slajdovi - pomešano']
plt.legend(labels=classes)
plt.show()

x = 3
loadings[x] = loadings[x].astype(str)
loadings[x] = np.insert(loadings[x], 0, vars, axis=1)
#loadings[x] = np.insert(loadings[x], 13, utils.smc(fa.corr_, sort=False), axis=1)
loadings[x] = np.insert(loadings[x], 0, ["", "F1", "F2", "F3"], axis=0)

with open(ptnj + "/loadings"+str(x)+".csv","w+", newline='') as my_csv:
  csvWriter = csv.writer(my_csv,delimiter=',')
  csvWriter.writerows(loadings[x])

# Korelacione matrice

matrica = pd.DataFrame(pd.read_csv(ptnj + "matricaFin.csv"))
matrica = matrica[matrica['uloga'] == 2] # Samo studenti
matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju

grid = matrica [['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g11', 'g10', 'g12']]

form = []
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)].copy())

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
    for x in range (0, 1000):
      bootr.append(bootstrap(form[a], form[b]))
    print(a, b, np.percentile(bootr, 2.5), np.mean(bootr), np.percentile(bootr, 97.5))

corrM1 = form[0].corr(method='spearman')
corrM2 = form[1].corr(method='spearman')
corrM3 = form[2].corr(method='spearman')
corrM4 = form[3].corr(method='spearman')

plt.clf()
f,axes = plt.subplots(2,2, figsize=(10,10))
sns.set_style("white")
sns.heatmap(corrM1, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
sns.heatmap(corrM2, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[0, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
sns.heatmap(corrM3, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 0], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
sns.heatmap(corrM4, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=axes[1, 1], square=True, cbar_kws={"shrink": .5}, xticklabels=True)
plt.show()

# Sredjivanje martrice

matrica = pd.DataFrame(pd.read_csv(ptnj + "matricaFin.csv"))


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
model.load_dataset(faktor)
opt = Optimizer(model)
ofv = opt.optimize()
inspect(opt)

stat = gather_statistics(opt)
stat
print(stat.rmsea)


U1, p = mannwhitneyu(form1['interviewtime'], form2['interviewtime'])
print(U1, p)