import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = "data/raw_data.csv"
PROCESSED = "data/processed_data.csv"
PREPROCESSOR_OUT = "models/preprocessor.pkl"

def build_preprocessor(df: pd.DataFrame):
    # detect columns
    drop_cols = ["Unnamed: 0"] if "Unnamed: 0" in df.columns else []
    df = df.drop(columns=drop_cols, errors='ignore')

    y_col = "y"
    if y_col in df.columns:
        X = df.drop(columns=[y_col])
    else:
        X = df.copy()

    # types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # simple fixes
    # replace 'pdays' 999 with np.nan or -1 (choose -1 to preserve info)
    if "pdays" in numeric_cols:
        X["pdays"] = X["pdays"].replace(999, -1)

    # pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols, df

def run_etl():
    df = pd.read_csv(RAW)
    preprocessor, num_cols, cat_cols, df_orig = build_preprocessor(df)

    # Fit preprocessor on X
    X = df_orig.drop(columns=["y"]) if "y" in df_orig.columns else df_orig.copy()
    preprocessor.fit(X)

    # transform and build processed dataframe (columns names for OHE)
    X_trans = preprocessor.transform(X)
    # Create feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    ohe_names = []
    if hasattr(ohe, 'get_feature_names_out'):
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
    else:
        # sklearn <1.0 fallback
        for i, col in enumerate(cat_cols):
            cats = preprocessor.named_transformers_['cat'].named_steps['ohe'].categories_[i]
            ohe_names += [f"{col}_{c}" for c in cats]

    feature_names = num_cols + ohe_names
    processed = pd.DataFrame(X_trans, columns=feature_names, index=df.index)

    if "y" in df.columns:
        processed["y"] = df["y"].map({"no": 0, "yes": 1})

    # save
    processed.to_csv(PROCESSED, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_OUT)
    print(f"Saved processed data -> {PROCESSED}")
    print(f"Saved preprocessor -> {PREPROCESSOR_OUT}")

if __name__ == "__main__":
    run_etl()
