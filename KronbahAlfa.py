import csv
import numpy as np
import pandas as pd

# Pretpostavimo da je vaš CSV fajl nazvan 'data.csv'
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")
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

corrM1 = form[0].corr(method='spearman')
corrM2 = form[1].corr(method='spearman')
corrM3 = form[2].corr(method='spearman')
corrM4 = form[3].corr(method='spearman')
def cronbach_alpha(df):
    # Broj stavki
    k = df.shape[1]
    # Variance for each column
    variances = df.var(axis=0, ddof=1)
    # Total variance
    total_variance = df.sum(axis=1).var(ddof=1)
    # Cronbach's alpha formula
    alpha = (k / (k - 1)) * (1 - (variances.sum() / total_variance))
    return alpha

# Izračunajmo Cronbachovu Alfu za svaki format
alpha_format0 = cronbach_alpha(form[0])
alpha_format1 = cronbach_alpha(form[1])
alpha_format2 = cronbach_alpha(form[2])
alpha_format3 = cronbach_alpha(form[3])

print(f'Cronbachova Alfa za Format 1: {alpha_format0}')
print(f'Cronbachova Alfa za Format 2: {alpha_format1}')
print(f'Cronbachova Alfa za Format 3: {alpha_format2}')
print(f'Cronbachova Alfa za Format 4: {alpha_format3}')
