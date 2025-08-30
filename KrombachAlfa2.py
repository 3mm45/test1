# import csv
# import numpy as np
# import pandas as pd
#
# # Pretpostavimo da je vaš CSV fajl nazvan 'data.csv'
# matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")
# matrica = pd.DataFrame(pd.read_csv("/Users/emmasarok/matricaFin.csv"))
#
# matrica = matrica[matrica['uloga'] == 2] # Samo studenti
# matrica = matrica[matrica['cesto'] > 1] # Samo oni koji redovno posecuju
#
# grid = matrica [['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
# vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']
#
# form = []
# form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] < 51)].copy())
# form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)].copy())
# form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] < 51)].copy())
# form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)].copy())
#
# corrM1 = form[0].corr(method='spearman')
# corrM2 = form[1].corr(method='spearman')
# corrM3 = form[2].corr(method='spearman')
# corrM4 = form[3].corr(method='spearman')
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
# print(f'Cronbachova Alfa za Format 1: {alpha_format0}')
# print(f'Cronbachova Alfa za Format 2: {alpha_format1}')
# print(f'Cronbachova Alfa za Format 3: {alpha_format2}')
# print(f'Cronbachova Alfa za Format 4: {alpha_format3}')

import pandas as pd

# Učitaj CSV fajl
matrica = pd.read_csv("/Users/emmasarok/matricaFin.csv")

# Proveri i prikaži broj NaN vrednosti pre čišćenja
print("Broj NaN vrednosti pre čišćenja:")
print(matrica.isna().sum())

# Ukloni redove koji imaju NaN vrednosti u ključnim kolonama
matrica = matrica.dropna(subset=['uloga', 'cesto', 'formatUp', 'hidden'] + ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12'])

# Filtriraj samo studente koji redovno posećuju
matrica = matrica[matrica['uloga'] == 2]  # Samo studenti
matrica = matrica[matrica['cesto'] > 1]   # Samo oni koji redovno posećuju

# Prikaz broj NaN vrednosti nakon čišćenja
print("Broj NaN vrednosti nakon čišćenja:")
print(matrica.isna().sum())

# Kreiraj grid sa potrebnim kolonama
grid = matrica[['formatUp', 'hidden', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']]
vars = ['g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12']

# Proveri varijabilnost podataka za svaku stavku
low_variability_cols = [col for col in vars if grid[col].nunique() <= 1]
print(f'Stakve sa niskom varijabilnošću: {low_variability_cols}')

# Ukloni kolone sa niskom varijabilnošću iz analize
vars = [col for col in vars if col not in low_variability_cols]
grid = grid[vars + ['formatUp', 'hidden']]

# Pripremi forme u skladu sa 'formatUp' i 'hidden' vrednostima
form = []
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 1) & (grid['hidden'] > 50)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] < 51)].copy())
form.append(grid[(grid['formatUp'] == 2) & (grid['hidden'] > 50)].copy())

# Proveri veličinu svake forme i NaN vrednosti
for i, f in enumerate(form):
    print(f'Format {i+1}: {f.shape}, NaN vrednosti: {f.isna().sum().sum()}')

# Ukloni NaN vrednosti iz svake forme pre analize
form = [f.dropna() for f in form]

# Proveri korelacije između stavki
for i, f in enumerate(form):
    print(f"Korelacija za Format {i+1}:")
    corr_matrix = f[vars].corr(method='spearman')
    print(corr_matrix)

    # Proveri negativne korelacije
    neg_corr = corr_matrix[corr_matrix < 0]
    print(f"Negativne korelacije za Format {i+1}:")
    print(neg_corr)

# Funkcija za izračunavanje Cronbachove Alfe
def cronbach_alpha(df):
    k = df.shape[1]  # Broj stavki
    variances = df.var(axis=0, ddof=1)  # Variance for each column
    total_variance = df.sum(axis=1).var(ddof=1)  # Total variance
    alpha = (k / (k - 1)) * (1 - (variances.sum() / total_variance))  # Cronbach's alpha formula
    return alpha

# Izračunaj Cronbachovu Alfu za svaki format
alpha_format0 = cronbach_alpha(form[0][vars])
alpha_format1 = cronbach_alpha(form[1][vars])
alpha_format2 = cronbach_alpha(form[2][vars])
alpha_format3 = cronbach_alpha(form[3][vars])

# Ispiši rezultate
print(f'Cronbachova Alfa za Format 1: {alpha_format0}')
print(f'Cronbachova Alfa za Format 2: {alpha_format1}')
print(f'Cronbachova Alfa za Format 3: {alpha_format2}')
print(f'Cronbachova Alfa za Format 4: {alpha_format3}')

