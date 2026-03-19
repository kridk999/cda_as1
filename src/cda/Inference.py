import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt
from data import preprocess_data
from bootStrap import bootstrap_predictions
import os
from tqdm import tqdm

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

data = pd.read_csv('data/case1Data.csv')
data_np = data.values
y = data_np[:, 0]
X = data_np[:, 1:]

X_train, y_train, preprocessor = preprocess_data(data)

new_data = pd.read_csv('data/case1Data_Xnew.csv')

def predictions(new_x, alpha, lambdas, train_data_path='data/case1Data.csv'):
    """
    Make predictions given new observations (new_x) and tuned hyperparameters.
    new_x should be a pandas DataFrame without the target variable 'y'.
    """
    
    # Train the model using the provided optimal hyperparameters
    model = linear_model.ElasticNet(
        l1_ratio=alpha, 
        alpha=lambdas, 
        max_iter=10000
    ).fit(X_train, y_train)

    # 4. Transform the new data using the ALREADY FITTED preprocessor
    # Since new_x lacks 'y', the preprocessor's transform method will return None for y_new
    X_new_processed, _ = preprocessor.transform(new_x)
    
    # Ensure missing categorical dummy columns are filled with 0 so dimensions match
    X_new_processed = X_new_processed.reindex(columns=X_train.columns, fill_value=0)
    
    # 5. Predict and return
    y_pred = model.predict(X_new_processed)
    
    return y_pred
    
def estimate_single_rmse_bootstrap(old_data, alpha, lambdas, B=100):
    """
    Calculates a single estimate of the expected test RMSE using 
    the Out-Of-Bag (OOB) Bootstrap method. 
    """
    # Preprocess once outside the loop for speed
    X_train_proc, y_train_proc, _ = preprocess_data(old_data)
    
    n_samples = len(old_data)
    indices = np.arange(n_samples)
    
    oob_mses = []
    
    for _ in tqdm(range(B), desc="Bootstrapping OOB Error"):
        # 1. Bootstrap sample (with replacement)
        boot_indices = np.random.choice(indices, size=n_samples, replace=True)
        
        # 2. Out-of-bag sample (the unused rows)
        oob_indices = np.setdiff1d(indices, boot_indices)
        
        if len(oob_indices) == 0:
            continue
            
        # 3. Split preprocessed data
        X_boot = X_train_proc.iloc[boot_indices]
        y_boot = y_train_proc.iloc[boot_indices]
        
        X_oob = X_train_proc.iloc[oob_indices]
        y_oob = y_train_proc.iloc[oob_indices]
        
        # 4. Train the model on the bootstrap sample
        model = linear_model.ElasticNet(
            l1_ratio=alpha, 
            alpha=lambdas, 
            max_iter=10000
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_boot, y_boot)
        
        # 5. Predict on out-of-bag data 
        preds = model.predict(X_oob)
        
        # 6. Calculate MSE for this OOB set
        mse = mean_squared_error(y_oob, preds)
        oob_mses.append(mse)
        
    # Average the MSEs to estimate the unseen test MSE
    expected_mse = np.mean(oob_mses)
    
    # Square root to get your final expected RMSE
    single_estimated_rmse = np.sqrt(expected_mse)
    
    print(f"\nEstimated Generalization RMSE (OOB Bootstrap): {single_estimated_rmse:.4f}")
    
    return single_estimated_rmse

def save_predictions_csv(predictions, filename='predictions.csv'):
    """
    Saves the given array of predictions to a CSV file.
    """
    # Create the reports folder if it doesn't exist 
    os.makedirs('reports', exist_ok=True)
    filepath = f"reports/{filename}"
    6
    df = pd.DataFrame({
        'Predicted_Y': predictions
    })
    df.to_csv(filepath, index=False)
    print(f"Saved {len(predictions)} predictions to {filepath}")


if __name__ == '__main__':
    new_data = pd.read_csv('data/case1Data_Xnew.csv')
    
    alpha_optimal = 1.0
    lambda_optimal = 6.1359
    
    # 1. Get predictions using tuned parameters
    y_new = predictions(new_data, alpha=alpha_optimal, lambdas=lambda_optimal)
    
    # 2. Save the predictions 
    save_predictions_csv(y_new, 'case1Data_Ynew_predictions.csv')
    
    # 3. Calculate the single best estimate for the RMSE using 100 bootstraps
    estimated_rmse = estimate_single_rmse_bootstrap(data, alpha=alpha_optimal, lambdas=lambda_optimal, B=2000)
    
    # 4. Save your estimated RMSE to a CSV
    os.makedirs('reports', exist_ok=True)
    pd.DataFrame({'Estimated_RMSE': [estimated_rmse]}).to_csv('reports/estimated_rmse.csv', index=False)
    
