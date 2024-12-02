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
os.chdir("Path 1")
dt_dir = "Path 2"
scrna = pd.read_csv(dt_dir+"Single_cell_gene_expression_data.txt", sep='\t', header=0, index_col=0)
st = pd.read_csv(dt_dir+"Spatial_transcriptome_gene_expression_count_data.txt", sep='\t', header=0, index_col=0)
lo = pd.read_csv(dt_dir+"Spatial_transcriptom_coordinates.csv", header=0, index_col=0)
celltypes1 = pd.read_csv(dt_dir+"Single_cell_celllable.csv", header=0, index_col=0)
predictions_assay = pd.read_csv(dt_dir+"Estimation_of_Cell_Types_in_Spots.csv", header=0, index_col=0)
print("\nST:")
print(st)
print("\nscrna :")
print(scrna)
print("\nlo :")
print(lo )
print("\ncelltypes1:")
print(celltypes1)
print("\npredictions_assay:")
print(predictions_assay)

#################Dropout Handling##############
data = scrna
data.replace(0, np.nan, inplace=True)
gene_expression_ratio = data.notna().mean(axis=1)
filtered_data = data[gene_expression_ratio >= 0.1]

threshold_cell = 0.1 * filtered_data.shape[0]
filtered_data = filtered_data.loc[:, filtered_data.gt(0).sum(axis=0) >= threshold_cell]

cell_expression_ratio = filtered_data.notna().mean(axis=0)
scrna=filtered_data

celltypes = celltypes1.copy()
celltypes = celltypes.loc[scrna.columns]

celltypes[celltypes.columns[0]] = celltypes[celltypes.columns[0]].str.replace(r'[^\w\s]', ' ', regex=True)
celltypes1 = celltypes.loc[scrna.columns]

imputed_data_df = pd.DataFrame(index=filtered_data.index)
for cell_type in celltypes.iloc[:, 0].unique():
    same_type_cells = celltypes[celltypes.iloc[:, 0] == cell_type].index
    same_type_data = filtered_data[same_type_cells]

    cell_expression_ratio = same_type_data.notna().mean(axis=0)
    bins = np.arange(0, 1.1, 0.1)
    hist, bin_edges = np.histogram(cell_expression_ratio, bins=bins)
    max_index = np.argmax(hist)

    max_interval_start2 = bin_edges[max_index - 3]
    max_interval_end2 = bin_edges[max_index - 2]
    max_interval_start1 = bin_edges[max_index - 2]
    max_interval_end1 = bin_edges[max_index - 1]

    low_expression_cells = ((cell_expression_ratio > max_interval_start2)
                            & (cell_expression_ratio < max_interval_end2)
                            | (cell_expression_ratio > max_interval_start1)
                            & (cell_expression_ratio < max_interval_end1))
    high_expression_data = same_type_data.loc[:, ~low_expression_cells]
    low_expression_data = same_type_data.loc[:, low_expression_cells]
    low_expression_data = low_expression_data.dropna(axis=1, how='all')
    sc_data = low_expression_data

    filled_data_0 = same_type_data.copy()
    filled_data_0.fillna(0, inplace=True)
    filtered_data_T = filled_data_0.T
    knn_expression = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn_expression.fit(filtered_data_T.values)
    knn_expression1 = NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn_expression1.fit(filtered_data_T.values)

    filled_data = sc_data.copy()
    if not low_expression_data.empty:

        for cell in sc_data.columns:
            gene_expression = filtered_data_T.loc[cell].values.reshape(1, -1)
            num_samples = same_type_data.shape[1]

            if num_samples >= 5:
                distances1, expression_neighbors = knn_expression.kneighbors(gene_expression, n_neighbors=5)
                common_neighbors = [item for sublist in expression_neighbors for item in sublist]

                gene_expression = sc_data[cell].values
                nan_indices = np.where(np.isnan(gene_expression))[0]

                np.random.seed(1)
                selected_nan_indices = np.random.choice(nan_indices, size=int(0.5 * len(nan_indices)), replace=False)

                if len(selected_nan_indices) > 0:

                    for i, index in enumerate(selected_nan_indices):
                        predicted_value = np.mean(filled_data_0.iloc[:, common_neighbors].iloc[index, :])
                        gene_expression[index] = predicted_value
                filled_data[cell] = gene_expression

            else:
                distances1, expression_neighbors = knn_expression1.kneighbors(gene_expression, n_neighbors=1)
                nearest_neighbors1 = [item for sublist in expression_neighbors for item in sublist]

                gene_expression = sc_data[cell].values
                nan_indices = np.where(np.isnan(gene_expression))[0]
                np.random.seed(1)
                selected_nan_indices = np.random.choice(nan_indices, size=int(0.5 * len(nan_indices)), replace=False)

                if len(selected_nan_indices) > 0:

                    for i, index in enumerate(selected_nan_indices):
                        predicted_value = np.mean(filled_data_0.iloc[:, nearest_neighbors1].iloc[index, :])
                        gene_expression[index] = predicted_value
                filled_data[cell] = gene_expression

            imputed_same_type_data = pd.concat([high_expression_data, filled_data], axis=1)

    else:
        imputed_same_type_data = high_expression_data
    imputed_data_df = pd.concat([imputed_data_df, imputed_same_type_data], axis=1)
imputed_data_df.fillna(0, inplace=True)
scrna = imputed_data_df
data = st
coordinates = lo

data.replace(0, np.nan, inplace=True)
gene_expression_ratio = data.notna().mean(axis=1)
filtered_data = data[gene_expression_ratio >= 0.1]
cell_expression_ratio = filtered_data.notna().mean(axis=0)

bins = np.arange(0, 1.1, 0.1)
hist, bin_edges = np.histogram(cell_expression_ratio, bins=bins)
max_index = np.argmax(hist)
max_interval_start2 = bin_edges[max_index - 3]
max_interval_end2 = bin_edges[max_index - 2]
max_interval_start1 = bin_edges[max_index - 2]
max_interval_end1 = bin_edges[max_index - 1]

low_expression_cells = ((cell_expression_ratio > max_interval_start2)
                        & (cell_expression_ratio < max_interval_end2)
                        | (cell_expression_ratio > max_interval_start1)
                        & (cell_expression_ratio < max_interval_end1))
high_expression_data = filtered_data.loc[:, ~low_expression_cells]
low_expression_data = filtered_data.loc[:, low_expression_cells]
st_data = low_expression_data
spot_coordinates = coordinates

knn_spatial = NearestNeighbors(n_neighbors=10, algorithm='auto')
knn_spatial.fit(spot_coordinates.values)
knn_expression = KNeighborsRegressor(n_neighbors=10, weights='distance')
filled_data = st_data.copy()

for spot in st_data.columns:
    gene_expression = st_data[spot].values
    nan_indices = np.where(np.isnan(gene_expression))[0]
    np.random.seed(1)
    selected_nan_indices = np.random.choice(nan_indices,
                                            size=int(0.5 * len(nan_indices)), replace=False)

    if len(selected_nan_indices) > 0:
        non_nan_indices = np.where(~np.isnan(gene_expression))[0]
        knn_expression.fit(np.arange(len(filtered_data))[non_nan_indices].reshape(-1, 1),
                           filtered_data[spot].values[non_nan_indices])
        gene_coordinate = spot_coordinates.loc[spot].values.reshape(1, -1)
        distances, spatial_neighbors = knn_spatial.kneighbors(gene_coordinate, n_neighbors=10)
        expression_neighbors = knn_expression.kneighbors(
            np.arange(len(filtered_data))[selected_nan_indices].reshape(-1, 1), return_distance=False)
        common_neighbors = []

        for i in range(len(spatial_neighbors)):
            common = list(set(spatial_neighbors[i]) & set(expression_neighbors[i]))
            common_neighbors.append(common)
            common_neighbors = [item for sublist in common_neighbors for item in sublist]

        for i, index in enumerate(selected_nan_indices):

            if len(common_neighbors) > 0:
                predicted_value = np.mean(gene_expression[common_neighbors])

            else:
                nearest_neighbors = expression_neighbors[:5]

                if len(nearest_neighbors) > 0:
                    predicted_value = np.mean(gene_expression[nearest_neighbors])

                else:
                    predicted_value = 0
            gene_expression[index] = predicted_value

    filled_data[spot] = gene_expression

imputed_data_df = pd.concat([high_expression_data, filled_data], axis=1)
imputed_data_df.fillna(0, inplace=True)
st = imputed_data_df
stA =st
scA = scrna
