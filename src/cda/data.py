import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_column_types(df: pd.DataFrame):
    x_cols = [col for col in df.columns if col.startswith("x_")]
    c_cols = [col for col in df.columns if col.startswith("C_")]
    return x_cols, c_cols


def make_preprocessor(df: pd.DataFrame):
    x_cols, c_cols = get_column_types(df)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),])

    categorical_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="constant", fill_value="missing")), #den her skal lige tjekkes for tror ik det er rigtigt!
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, x_cols),
            ("cat", categorical_pipeline, c_cols),])


def preprocess_data(df: pd.DataFrame, target_col: str = "y"):
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col] if target_col in df.columns else None

    preprocessor = make_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    return X_processed, y, preprocessor

if __name__ == '__main__':

    df = pd.read_csv("../data/case1Data.csv")

    X_processed, y, preprocessor = preprocess_data(df)

    print(X_processed.head())
    print(y.head())