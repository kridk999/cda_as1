import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import linear_model
import warnings
import matplotlib.pyplot as plt
from data import preprocess_data
from bootStrap import bootstrap_predictions

# DTU Colors
DTU_RED = '#990000'
DTU_NAVY = '#00213E'

data = pd.read_csv('data/case1Data.csv')
data_np = data.values
y = data_np[:, 0]
X = data_np[:, 1:]

X_train, y_train, preprocessor = preprocess_data(data)

new_data = pd.read_csv('data/case1Data_Xnew.csv')

def inference(new_data, old_data=data):
    # 1. Get bootstrap predictions for comparison later
    
    lambdas = 6.1359
    alpha = 1.0
    
    new_df = bootstrap_predictions(B=100, alpha_ratio=alpha, lambda_val=lambdas) 
    bootstrap_y = new_df["Predicted_Y"].values

    # 2. Set the hyperparameters you found during tuning
  

    # 3. Fit preprocessor and model on ALL of the old data (No KFold!)
    print("Fitting final model on entire old_data...")
    X_train_full, y_train_full, final_preprocessor = preprocess_data(old_data)
    
    final_model = linear_model.ElasticNet(
        l1_ratio=alpha, 
        alpha=lambdas, 
        max_iter=10000
    ).fit(X_train_full, y_train_full)

    # 4. Transform new_data using the final_preprocessor
    X_test, _ = final_preprocessor.transform(new_data)

    # 5. Make predictions
    predictions = final_model.predict(X_test)
    
    print("\nPredictions for new data:")
    print(predictions)

    # 6. Evaluate against bootstrap
    rmse = np.sqrt(mean_squared_error(predictions, bootstrap_y))
    print(f"\nRMSE between ElasticNet predictions and Bootstrap predictions: {rmse:.4f}")

    return predictions

if __name__ == '__main__':
    new_data = pd.read_csv('data/case1Data_Xnew.csv')
    inference(new_data)