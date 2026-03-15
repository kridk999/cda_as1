import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.utils import resample
import warnings

# Import the preprocessor used in train.py
from data import preprocess_data

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

def bootstrap_predictions(B=1000):
    '''
    Use Bootstrap to generate predictions and confidence intervals for a test set.
    '''
    df_train = pd.read_csv('data/case1Data.csv')
    df_test = pd.read_csv('data/case1Data_Xnew.csv') 
    
    alpha_ratio = 0.001       
    lambda_val = 1.44
    
    N_train = len(df_train)
    N_test = len(df_test)
    
    boot_preds = np.zeros((B, N_test))

    print(f'--- Starting Bootstrapped Predictions (B={B}) ---')

    for b in range(B):
        df_resampled = resample(df_train, replace=True, n_samples=N_train, random_state=b)
        
        X_train, y_train, preprocessor = preprocess_data(df_resampled)

        X_test, _ = preprocessor.transform(df_test)
        
        
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            model = linear_model.ElasticNet(l1_ratio=alpha_ratio, alpha=lambda_val).fit(X_train, y_train)
            
        boot_preds[b, :] = model.predict(X_test)

    final_preds = np.mean(boot_preds, axis=0)
    

    lower_bounds = np.percentile(boot_preds, 2.5, axis=0)
    upper_bounds = np.percentile(boot_preds, 97.5, axis=0)

    results_df = pd.DataFrame({
        'Predicted_Y': final_preds,
        'Lower_95_CI': lower_bounds,
        'Upper_95_CI': upper_bounds,
        'Uncertainty_Spread': upper_bounds - lower_bounds
    })

    return results_df

if __name__ == '__main__':
    bootstrap_predictions()