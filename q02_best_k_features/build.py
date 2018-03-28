# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k = 20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    Selector_f = SelectPercentile(f_regression, percentile=k)
    Selector_f.fit_transform(X,y)
    columns = np.asarray(X.columns.values)
    support = np.asarray(Selector_f.get_support())
    columns_with_support = columns[support]
    list_fea = list(columns_with_support)
    list1 = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return list1


#print(percentile_k_features(data))
