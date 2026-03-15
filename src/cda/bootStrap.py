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

def bootstrap_predictions():
    '''
    Use Bootstrap to generate predictions and confidence intervals for a test set.
    '''
    # 1. Load Training Data (with y) and Test Data (without y)
    df_train = pd.read_csv('data/case1Data.csv')
    df_test = pd.read_csv('data/case1Data_Xnew.csv') # --> UPDATE THIS PATH!
    
    # Parameters based on train.py selection
    alpha_ratio = 0.001       
    lambda_val = 1.44
    
    B = 1000
    N_train = len(df_train)
    N_test = len(df_test)
    
    # Store predictions instead of coefficients. 
    # Shape: (Bootstrap iterations vs Number of test samples)
    boot_preds = np.zeros((B, N_test))

    print(f'--- Starting Bootstrapped Predictions (B={B}) ---')

    for b in range(B):
        # 2. Resample the training data
        df_resampled = resample(df_train, replace=True, n_samples=N_train, random_state=b)
        
        # 3. Preprocess training data (learn median/categories)
        X_train_resamp, y_train_resamp, preprocessor = preprocess_data(df_resampled)
        
        # 4. Preprocess test data (apply SAME median/categories!)
        X_test_unprocessed = df_test.drop(columns=['y'], errors='ignore')
        
        # Ensure categorical matching like in train.py validation fix
        c_cols = [col for col in X_test_unprocessed.columns if col.startswith("C_")]
        X_test_unprocessed[c_cols] = X_test_unprocessed[c_cols].astype("object")
        
        X_test_processed_array = preprocessor.transform(X_test_unprocessed) 
        feature_names = preprocessor.get_feature_names_out()
        X_test_processed_df = pd.DataFrame(
            X_test_processed_array, 
            columns=feature_names, 
            index=X_test_unprocessed.index
        )
        
        # 5. Fit model on this bootstrap sample
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            model = linear_model.ElasticNet(l1_ratio=alpha_ratio, alpha=lambda_val).fit(X_train_resamp, y_train_resamp)
            
        # 6. Predict on the test set and store
        boot_preds[b, :] = model.predict(X_test_processed_df)

    # 7. Calculate final predictions and uncertainty
    # We take the median or mean of all bootstrap predictions as 
    # the final estimate
    final_preds = np.mean(boot_preds, axis=0)
    
    # Calculate 95% Confidence Intervals for predictions
    lower_bounds = np.percentile(boot_preds, 2.5, axis=0)
    upper_bounds = np.percentile(boot_preds, 97.5, axis=0)

    # Combine into a final dataframe that you can save
    results_df = pd.DataFrame({
        'Predicted_Y': final_preds,
        'Lower_95_CI': lower_bounds,
        'Upper_95_CI': upper_bounds,
        'Uncertainty_Spread': upper_bounds - lower_bounds
    })

    print("\nFirst 5 predictions with uncertainty:")
    print(results_df.head())
    
    # Optional: Save to CSV
    # results_df.to_csv("data/predictions_with_intervals.csv", index=False)
    
    # --- VISUALIZATION: Show uncertainty of first 50 predictions ---
    plt.figure(figsize=(12, 6))
    
    # How many test samples to plot (plotting all might be too messy)
    plot_num = min(50, N_test)
    x_axis = range(plot_num)
    y_err = np.array([final_preds[:plot_num] - lower_bounds[:plot_num], 
                      upper_bounds[:plot_num] - final_preds[:plot_num]])

    plt.errorbar(x_axis, final_preds[:plot_num], yerr=y_err, fmt='o',
                 color=DTU_NAVY, ecolor=DTU_RED, capsize=4, markersize=5,
                 label='Prediction with 95% CI')
                 
    plt.title(f'Test Set Predictions and Bootstrap Uncertainty (First {plot_num} samples)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Predicted y value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    bootstrap_predictions()