import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

PROCESSED = "data/processed_data.csv"
OUT_DIR = "results"

def run_eda():
    df = pd.read_csv(PROCESSED)
    # simple class balance
    bal = df['y'].value_counts().to_dict()

    # correlation heatmap (numeric part)
    num = df.select_dtypes(include=['float64','int64'])
    corr = num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/corr_matrix.png")
    plt.close()

    # distribution of important features
    for col in ['age','duration','campaign']:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col].dropna(), bins=30)
            plt.title(col)
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/{col}_dist.png")
            plt.close()

    # top categories if any (take first 10 object-like columns from raw data)
    # Save simple report
    report = {"class_balance": bal, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
    with open(f"{OUT_DIR}/eda_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print("EDA done. Files saved to results/")

if __name__ == "__main__":
    run_eda()
