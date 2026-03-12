import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# 

def get_column_types(df: pd.DataFrame):
    x_cols = [col for col in df.columns if col.startswith("x_")]
    c_cols = [col for col in df.columns if col.startswith("C_")]
    return x_cols, c_cols


def make_preprocessor(df: pd.DataFrame):
    x_cols, c_cols = get_column_types(df)

    # convert categoricals to string
    df[c_cols] = df[c_cols].astype("object")

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),])

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, x_cols),
            ("cat", categorical_pipeline, c_cols),
        ],verbose_feature_names_out=False) # gør bare at navnene ikke bliver random til sidst


def preprocess_data(df: pd.DataFrame, target_col: str = "y"):
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col] if target_col in df.columns else None

    preprocessor = make_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    return X_processed, y, preprocessor

if __name__ == '__main__':
    print("hello")
    df = pd.read_csv("data/case1Data.csv")

    X_processed, y, preprocessor = preprocess_data(df)

    print(X_processed.head())
    print(y.head())