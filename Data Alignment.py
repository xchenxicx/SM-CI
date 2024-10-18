import pandas as pd
import numpy as np
from scipy import stats
import os
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import sys
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
def average_replicates(df):
    return df.groupby(df.index).mean()
stA = average_replicates(stA)
scA = average_replicates(scA)
common_genes = scA.index.intersection(stA.index)
scA = scA.loc[common_genes]
stA = stA.loc[common_genes]
stA = stA.reindex(scA.index)

col_sums = predictions_assay.sum(axis=0)
cellfrac = col_sums / col_sums.sum()
cellfrac = pd.DataFrame({'Index': cellfrac.index, 'Fraction': cellfrac.values})
cellfrac = cellfrac.T
all_vars = locals().copy()

st = stA.copy()
stY = st.copy()
number = 10
replicated_columns = [stY.iloc[:, col_index]
                      for col_index in range(stY.shape[1]) for _ in range(number)]
result_ST = pd.concat(replicated_columns, axis=1)
result_ST.columns = np.repeat(stY.columns, number)

spotK = result_ST
lo_transposed = lo.T
stY = lo_transposed.copy()
replicated_columns = [stY.iloc[:, col_index]
                      for col_index in range(stY.shape[1]) for _ in range(number)]
result_ST = pd.concat(replicated_columns, axis=1)
result_ST.columns = np.repeat(stY.columns, number)

spotKXY = result_ST
scrna = scA.copy()
celltypes = celltypes1.copy()
celltypes = celltypes.loc[scrna.columns]

celltypes[celltypes.columns[0]] = celltypes[celltypes.columns[0]].str.replace(r'[^\w\s]', ' ', regex=True)
celltypes = celltypes.loc[scrna.columns]
Type = celltypes
unique_Type = Type[Type.columns[0]].unique()
unique_Type_length = len(unique_Type)

positions = {k: list(v) for k, v in Type.groupby(Type[Type.columns[0]]).groups.items()}
positions = {k: v for k, v in positions.items() if k is not None}
lab = celltypes.iloc[:, 0]

scrna = pd.concat([scrna, celltypes.iloc[:, 1:3].T, pd.DataFrame(lab).T], ignore_index=False, axis=0)
cell_type_counts = celltypes.iloc[:, 0].value_counts()

weights = pd.DataFrame(cellfrac)
weights = weights.drop(weights.index[0])
spot = spotKXY
n = spot.shape[1]

new_row = np.round(weights.iloc[0, :].astype(float) * n)
weights = pd.concat([weights, pd.DataFrame([new_row], columns=weights.columns)], ignore_index=True)

weights = weights.apply(pd.to_numeric)
max_value_index = weights.iloc[1, :].idxmax()

max_value = weights.iloc[1, max_value_index] + n - weights.iloc[1, :].sum()
weights.iloc[1, max_value_index] = max_value
unique_Type1 = pd.DataFrame(unique_Type).transpose()

unique_Type1 = pd.concat([unique_Type1, unique_Type1], ignore_index=True)
unique_Type1.columns = unique_Type1.iloc[0, :]
unique_Type1 = unique_Type1.drop(0).reset_index(drop=True)
weights.columns = cellfrac.iloc[0]

weights.columns = weights.columns.astype(str).str.replace(r'[^\w\s]', ' ', regex=True)
unique_Type1.columns = unique_Type1.columns.astype(str).str.replace(r'[^\w\s]', ' ', regex=True)
print(weights.head())

extra_cols = list(set(unique_Type1.columns) - set(weights.columns))
common_cols = [col for col in weights.columns if col in unique_Type1.columns]
unique_Type1 = unique_Type1[common_cols + extra_cols]

extra_data = pd.DataFrame(0, index=weights.index, columns=extra_cols)
weights = pd.concat([weights, extra_data], axis=1)

target_counts = weights.iloc[1, :]
extracted_cells = {}

np.random.seed(1)
for cell_type in target_counts.keys():
    cells_indices = np.where(scrna.iloc[-1, :] == cell_type)[0]
    m = len(cells_indices)
    n = int(target_counts[cell_type])

    if n <= m:
        selected_indices = np.random.choice(cells_indices, n, replace=False)

    else:
        selected_indices = np.concatenate((cells_indices, np.random.choice(cells_indices, n - m, replace=True)))

    extracted_cells[cell_type] = scrna.iloc[:, selected_indices]
result_matrix = pd.concat(extracted_cells.values(), axis=1)
result_matrix.index = scrna.index
print(result_matrix.columns)

result_matrix = pd.DataFrame(result_matrix)
label = result_matrix.iloc[-1, :].to_frame()

weizhi = result_matrix.iloc[-3:-1, :].T
result_matrix = result_matrix.iloc[:-3, :]

cellK = result_matrix
lableK = label
print("\nspotK:")
print(spotK)
print("\ncellK:")
print(cellK)
print("\nlableK:")
print(lableK)

sc = cellK
st = spotK
sc_matrix = sc.to_numpy()
st_matrix = st.to_numpy()
sc_matrix = sc_matrix.astype(np.float64)
st_matrix = st_matrix.astype(np.float64)
sc_matrix = sc_matrix / (sc_matrix.sum(axis=0) / 1e6)
sc_matrix = np.log2(sc_matrix + 1)
st_matrix = st_matrix / (st_matrix.sum(axis=0) / 1e6)
st_matrix = np.log2(st_matrix + 1)
sc = pd.DataFrame(sc_matrix, index=sc.index, columns=sc.columns)
st = pd.DataFrame(st_matrix, index=st.index, columns=st.columns)
JBspotK=st
JBcellK=sc
