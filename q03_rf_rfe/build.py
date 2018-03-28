# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


#Your solution code here
def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    model= RandomForestClassifier()
    n = df.shape[1]/2

    
    rfe = RFE(RandomForestClassifier(),n_features_to_select= (X.shape[1]/2))
    rfe.fit(X,y)
    

    column  = X.columns.values
    support = rfe.support_

    
    top_features = np.ndarray.tolist(column[support])
    return top_features

#print(rf_rfe(data))




