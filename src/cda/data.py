import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def get_column_types(df: pd.DataFrame):
    x_cols = [col for col in df.columns if col.startswith("x_")]
    c_cols = [col for col in df.columns if col.startswith("C_")]
    return x_cols, c_cols

class CustomPreprocessor:
    """
    A wrapper around ColumnTransformer that handles dataset splitting,
    type casting, and returning pandas DataFrames for a clean interface.
    """
    def __init__(self, target_col="y"):
        self.target_col = target_col
        self.transformer = None
        self.c_cols = []
        
    def fit_transform(self, df: pd.DataFrame):

        X = df.drop(columns=[self.target_col], errors="ignore").copy()
        y = df[self.target_col] if self.target_col in df.columns else None
        

        x_cols, self.c_cols = get_column_types(X)

        X[self.c_cols] = X[self.c_cols].astype("object")

        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, x_cols),
                ("cat", categorical_pipeline, self.c_cols),
            ],
            verbose_feature_names_out=False
        )

        X_processed_array = self.transformer.fit_transform(X)
 
        feature_names = self.transformer.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=X.index)
        
        return X_processed, y

    def transform(self, df: pd.DataFrame):
        
        X = df.drop(columns=[self.target_col], errors="ignore").copy()
        y = df[self.target_col] if self.target_col in df.columns else None
        
        
        X[self.c_cols] = X[self.c_cols].astype("object")
        
        
        X_processed_array = self.transformer.transform(X)

        feature_names = self.transformer.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=X.index)
        
        return X_processed, y
    
    def get_feature_names_out(self):
        """Pass through method to get feature names from the underlying transformer"""
        if self.transformer is None:
            raise ValueError("The transformer has not been fitted yet.")
        return self.transformer.get_feature_names_out()

def preprocess_data(df: pd.DataFrame, target_col: str = "y"):
    """
    Main entry point for training data. 
    Initialize the custom wrapper and fit it to the dataframe.
    """
    preprocessor = CustomPreprocessor(target_col=target_col)
    X_processed, y = preprocessor.fit_transform(df)
    
    return X_processed, y, preprocessor

if __name__ == '__main__':
    print("hello")
    df = pd.read_csv("data/case1Data.csv")

    X_processed, y, preprocessor = preprocess_data(df)

    print(X_processed.head())
    print(y.head())