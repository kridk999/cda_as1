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
    new_df = bootstrap_predictions(B=100) 
    bootstrap_y = new_df["Predicted_Y"].values


    CV = 5
    kf = KFold(n_splits=CV)

    lambdas = 1.44
    alpha = 0.001


    for i, (train_index, test_index) in enumerate(kf.split(data)):

        df_train = data.iloc[train_index]
        
        X_train, y_train, fitted_preprocessor = preprocess_data(df_train)

        model = linear_model.ElasticNet(l1_ratio=alpha, alpha=lambdas).fit(X_train, y_train)

    X_test, _ = fitted_preprocessor.transform(new_data)

    predictions = model.predict(X_test)
    print("Predictions for new data:")
    print(predictions)

    bootstrap_y = new_df["Predicted_Y"].values
    
    
    rmse = np.sqrt(mean_squared_error(predictions, bootstrap_y))
    print(f"RMSE between ElasticNet predictions and Bootstrap predictions: {rmse:.4f}")



    return predictions

if __name__ == '__main__':
    new_data = pd.read_csv('data/case1Data_Xnew.csv')
    inference(new_data)