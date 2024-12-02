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
B = cellpearson_row_l1_norms.values @ spotEuclidean_col_l1_norms.values
B = pd.DataFrame(B,
                    index=cellpearson.index,
                    columns=spotEuclidean.columns)

STSCpearson = STSCpearson.apply(normalize_0_1, axis=0)
B = B.apply(normalize_0_1, axis=0)

A = 0.5 * STSCpearson.values + 0.5 * STSCpearson.values * (1 / (B.values + 1))

A = pd.DataFrame(A, index=STSCpearson.index, columns=STSCpearson.columns)
row_ind, col_ind = linear_sum_assignment(-A)

assignment = np.vstack([row_ind, col_ind]).T
paixu = assignment[:, 1]
assignment = pd.DataFrame(assignment, index=A.index)

col = np.array(A.columns)
col_sorted = col[paixu]
col_sorted_reshaped = col_sorted.reshape(-1, 1)

assignment_with_col = np.hstack((assignment, col_sorted_reshaped))
assignment_with_col=assignment_with_col[:, 1:]

assignment = pd.DataFrame(assignment_with_col, index=A.index, columns=['Value', 'Column'])
assignment = assignment.iloc[:, 1:]

assignment = pd.concat([assignment, lableK.iloc[:, 0]], axis=1)
position = spotKXY.T

position_ID = position.index.to_list()
order_assignment = [position_ID.index(i) for i in assignment.iloc[:, 0]]

position = position.iloc[order_assignment, :]
assignment_index = assignment.index

assignment = pd.concat([assignment.reset_index(drop=True), position.reset_index(drop=True)], axis=1)
assignment['spot_ID'] = assignment.iloc[:, 0].str.replace(r'\.\d+$', '', regex=True)

assignment.insert(0, 'spot_ID', assignment.pop('spot_ID'))
assignment.index = assignment_index

spot_type = assignment.iloc[:, [0, 2]]
matrix_spot_type = pd.crosstab(spot_type.iloc[:, 0], spot_type.iloc[:, 1])

weights = matrix_spot_type.reset_index().set_index(spot_type.columns[0])
Type = pd.DataFrame(celltypes1)

unique_Type = Type.iloc[:, 0].unique()
print(len(unique_Type))

unique_Type1 = pd.DataFrame([unique_Type, unique_Type])
unique_Type1.columns = unique_Type1.iloc[0]

weights.columns = weights.columns.str.replace(r'[^\w\s]', ' ', regex=True)
unique_Type1.columns = unique_Type1.columns.str.replace(r'[^\w\s]', ' ', regex=True)
unique_Type1 = unique_Type1.iloc[1:].reset_index(drop=True)

extra_cols = [col for col in unique_Type1.columns if col not in weights.columns]
extra_data = pd.DataFrame(0, index=weights.index, columns=extra_cols)

weights = pd.concat([weights, extra_data], axis=1)
matrix_spot_type = weights.values
prop_data = weights.div(weights.sum(axis=1), axis=0)
matrix_spot_type_df = pd.DataFrame(matrix_spot_type, index=weights.index, columns=weights.columns)

print("\nCell_position")
print(assignment)
print("\nProportion_cell_types_spots")
print(prop_data)
print("\nNumber_cell_types_spots")
print(matrix_spot_type_df)
assignment.to_csv("Cell_position.csv")
prop_data.to_csv("Proportion_cell_types_spots.csv")
matrix_spot_type_df.to_csv("Number_cell_types_spots.csv")
