import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt
from data import preprocess_data


data = pd.read_csv('data/case1Data.csv').values
y = data[:, 0]
X = data[:, 1:]

X_train, y_train, preprocessor = preprocess_data(data)

CV = 5
kf = KFold(n_splits=CV)

[n,p] = X_train.shape
lambdas = np.logspace(-3, 1, num=20)
alpha = 0.001

coefs = np.zeros((CV, len(lambdas), p))

    
for i, (train_index, test_index) in enumerate(kf.split(X)):
    #ytrain = y[train_index].ravel() 
    #Xtrain = X[train_index]
    
    df_train = data.iloc[train_index]
    
    X_train, y_train, preprocessor = preprocess_data(df_train)
    
    for j, lambda_ in enumerate(lambdas):
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")

            model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambda_, normalize=False).fit(X_train, y_train)
            coefs[i,j,:] = model.coef_
        
trace = np.sum(coefs, axis=0)
plt.figure()
plt.semilogx(lambdas, trace)
plt.xlabel(r'$\lambda$')
plt.ylabel('Sum of coefficients')
plt.title('Sum of coefficients of Elastic Net Fit Alpha = %.2f' % alpha)
plt.show()