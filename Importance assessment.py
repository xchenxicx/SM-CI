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

sc = JBcellK.copy()
st = JBspotK.copy()
sc = sc.apply(pd.to_numeric)
st = st.apply(pd.to_numeric)
sc.index = JBcellK.index
st.index = JBspotK.index

cor_matrix = np.corrcoef(sc.T, st.T)[:sc.shape[1], sc.shape[1]:]
STSCpearson = cor_matrix
cellpearson = np.corrcoef(sc.T)
spotpearson = np.corrcoef(st.T)

STSCpearson = pd.DataFrame(STSCpearson, index=sc.columns, columns=st.columns)
cellpearson = pd.DataFrame(cellpearson, index=sc.columns, columns=sc.columns)
spotpearson = pd.DataFrame(spotpearson, index=st.columns, columns=st.columns)

loc = spotKXY.values
spot = spotKXY.columns

A_t = loc.T
B_t = loc.T
dist_matrix = distance_matrix(A_t, B_t)
dist_matrix1 = dist_matrix

spotEuclidean = 1 / (1 + dist_matrix1)
spotEuclidean = pd.DataFrame(spotEuclidean, index=spot, columns=spot)

spotEuclidean = np.dot(spotEuclidean, spotpearson)
spotEuclidean = pd.DataFrame(spotEuclidean, index=spot, columns=spot)

def normalize_0_1(x):
    range_val = np.max(x) - np.min(x)
    if range_val == 0:
        return np.zeros_like(x)
    else:
        return (x - np.min(x)) / range_val

STSCpearson = STSCpearson.apply(normalize_0_1, axis=0)
cellpearson = cellpearson.apply(normalize_0_1, axis=0)
spotEuclidean = spotEuclidean.apply(normalize_0_1, axis=0)

print("\nspotEuclidean:")
print(spotEuclidean)
print("\nSTSCpearson:")
print(STSCpearson)
print("\ncellpearson:")
print(cellpearson)

cellpearson_row_l1_norms = cellpearson.abs().sum(axis=1).to_frame(name='L1_norm')
spotEuclidean_col_l1_norms = spotEuclidean.abs().sum(axis=0).to_frame().T
