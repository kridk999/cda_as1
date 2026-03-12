import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt


data = pd.read_csv('data/case1Data.csv').values
y = data[:, 0]
X = data[:, 1:]

CV = 5
kf = KFold(n_splits=CV)

[n,p] = X.shape
lambdas = np.logspace(-3, 1, num=20)
alpha = 0.001

coefs = np.zeros((CV, len(lambdas), p))


for i, (train_index, test_index) in enumerate(kf.split(X)):
    ytrain = y[train_index].ravel() 
    Xtrain = X[train_index]
    
    # <-- Apply Custom Imputation HERE
    # Ensure my_custom_imputer returns the imputed numpy array
    #Xtrain = my_custom_imputer(Xtrain)
    
    for j, lambda_ in enumerate(lambdas):
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")

            model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambda_, normalize=False).fit(Xtrain, ytrain)
            coefs[i,j,:] = model.coef_
        
trace = np.sum(coefs, axis=0)
plt.figure()
plt.semilogx(lambdas, trace)
plt.xlabel(r'$\lambda$')
plt.ylabel('Sum of coefficients')
plt.title('Sum of coefficients of Elastic Net Fit Alpha = %.2f' % alpha)
plt.show()