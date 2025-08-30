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


loadings = []
evs = []
brf = [3, 3, 3, 3]

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

x = 2
loadings[x] = loadings[x].astype(str)
loadings[x] = np.insert(loadings[x], 0, vars, axis=1)
#loadings[x] = np.insert(loadings[x], 13, utils.smc(fa.corr_, sort=False), axis=1)
loadings[x] = np.insert(loadings[x], 0, ["", "F1", "F2", "F3"], axis=0)

with open("/Users/emmasarok/loadings"+str(x)+".csv","w+", newline='') as my_csv:
  csvWriter = csv.writer(my_csv,delimiter=',')
  csvWriter.writerows(loadings[x])

