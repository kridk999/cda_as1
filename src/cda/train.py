import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt
from data import preprocess_data

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

data = pd.read_csv('data/case1Data.csv')
data_np = data.values
y = data_np[:, 0]
X = data_np[:, 1:]

X_train, y_train, preprocessor = preprocess_data(data)

CV = 5
kf = KFold(n_splits=CV)

[n,p] = X_train.shape
lambdas = np.logspace(-3, 1, num=20)
alpha = 0.001

coefs = np.zeros((CV, len(lambdas), p))
cv_means = []
cv_ses = []

errors = np.zeros((CV, len(lambdas)))

for i, (train_index, test_index) in enumerate(kf.split(data)):
    #ytrain = y[train_index].ravel() 
    #Xtrain = X[train_index]
    fold_errors = []
    
    df_train = data.iloc[train_index]
    df_val = data.iloc[test_index]
    
    X_train_fold, y_train_fold, preprocessor = preprocess_data(df_train)
    X_val_fold, y_val_fold, _ = preprocess_data(df_val)
    X_val, y_val = X_train.iloc[test_index], y_train.iloc[test_index]

    
    for j, lambda_ in enumerate(lambdas):
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")

            model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambda_).fit(X_train_fold, y_train_fold)
            coefs[i,j,:] = model.coef_
            preds = model.predict(X_val)
            errors[i, j] = mean_squared_error(y_val, preds)


            
cv_means = np.mean(errors, axis=0)
cv_ses = np.std(errors, axis=0) / np.sqrt(CV)
        
        
 # 1. Find lambda_min
idx_min = np.argmin(cv_means)
l_min = lambdas[idx_min]
err_min = cv_means[idx_min]
se_min = cv_ses[idx_min]

# 2. Apply One-SE Rule
threshold = err_min + se_min
# We want the largest lambda (simplest model) that is below the threshold
# Note: For Ridge, larger alpha = simpler model
possible_lambdas_idx = np.where(cv_means <= threshold)[0]
idx_1se = np.max(possible_lambdas_idx)
l_1se = lambdas[idx_1se]

print(f'Lambda Min: {l_min:.2f} (Error: {err_min:.4f})')
print(f'Lambda 1-SE: {l_1se:.2f} (Error: {cv_means[idx_1se]:.4f})')        

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.errorbar(lambdas, cv_means, yerr=cv_ses, fmt='o-', 
                color=DTU_RED, ecolor='lightgray', capsize=3, label='CV Mean Error')

plt.axhline(threshold, color=DTU_NAVY, linestyle='--', label='One-SE Threshold')
plt.axvline(l_min, color='gray', linestyle=':', label='Min Lambda')
plt.axvline(l_1se, color=DTU_NAVY, linestyle='-', label='Selected (One-SE)')

plt.xscale('log')
plt.xlabel('Complexity Penalty (Lambda)', color=DTU_NAVY, fontsize=12)
plt.ylabel('Mean Squared Error', color=DTU_NAVY, fontsize=12)
plt.title('Wine Quality Audit: The One-SE Rule', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
#plt.show()

print('\nVERDICT:')
print(f'The auditor selects lambda = {l_1se:.2f}.')
print('While its error is slightly higher than the minimum, it is within')
print('one standard error, meaning the difference is likely noise.')
print('The larger lambda results in a more parsimonious, robust model.')

trace = np.sum(coefs, axis=0)
plt.figure()
plt.semilogx(lambdas, trace)
plt.xlabel(r'$\lambda$')
plt.ylabel('Sum of coefficients')
plt.title('Sum of coefficients of Elastic Net Fit Alpha = %.2f' % alpha)
plt.show()