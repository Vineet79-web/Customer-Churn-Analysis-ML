"""
Customer Churn Analysis - Telco dataset (7,043 customers)

Run:
    python churn_analysis.py
    python churn_analysis.py --data path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv

What it does:
    1. Cleans the data
    2. Engineers features (tenure bands, contract buckets, charge bands)
    3. Prints the key churn statistics (the numbers you quote on your resume
       must come from THIS output, not from anywhere else)
    4. Saves 11 charts to outputs/figures/
    5. Trains Logistic Regression and Random Forest, saves metrics
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# 1. Load and clean
# --------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # TotalCharges has blank strings for customers with tenure 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna().sum()
    print(f"Blank TotalCharges fixed: {missing}")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop_duplicates(subset="customerID")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=[-1, 3, 12, 24, 48, 72],
        labels=["0-3 mo", "4-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    df["ContractBucket"] = df["Contract"].map(
        {"Month-to-month": "Monthly", "One year": "Annual", "Two year": "Annual"}
    )
    df["ChargeBand"] = pd.qcut(
        df["MonthlyCharges"], q=3, labels=["Low", "Medium", "High"]
    )
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
    return df


# --------------------------------------------------------------------------
# 3. Key statistics
# --------------------------------------------------------------------------
def print_key_stats(df: pd.DataFrame) -> dict:
    churned = df[df["Churn"] == 1]
    stats = {
        "customers": len(df),
        "overall_churn_rate_pct": round(df["Churn"].mean() * 100, 1),
        "churned_on_monthly_contract_pct": round(
            (churned["Contract"] == "Month-to-month").mean() * 100, 1
        ),
        "churned_with_tenure_under_3_months_pct": round(
            (churned["tenure"] < 3).mean() * 100, 1
        ),
        "churned_monthly_AND_tenure_under_3_months_pct": round(
            ((churned["Contract"] == "Month-to-month") & (churned["tenure"] < 3)).mean()
            * 100, 1
        ),
        "churn_rate_by_contract_pct": (
            df.groupby("Contract")["Churn"].mean().mul(100).round(1).to_dict()
        ),
    }
    print("\n=== KEY STATISTICS (use these exact numbers in your resume) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    pd.Series(stats).to_json(OUT_DIR / "key_stats.json", indent=2)
    return stats


# --------------------------------------------------------------------------
# 4. Visualisations (11 figures)
# --------------------------------------------------------------------------
def save(fig_name: str):
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close()


def make_figures(df: pd.DataFrame, feature_importance: pd.Series | None = None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1 churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn")
    plt.xticks([0, 1], ["Stayed", "Churned"])
    plt.title("Churn distribution")
    save("01_churn_distribution.png")

    # 2 tenure histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
    plt.title("Tenure distribution by churn")
    save("02_tenure_histogram.png")

    # 3 monthly charges histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, multiple="stack")
    plt.title("Monthly charges distribution by churn")
    save("03_monthly_charges_histogram.png")

    # 4 tenure box plot
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Churn", y="tenure")
    plt.xticks([0, 1], ["Stayed", "Churned"])
    plt.title("Tenure vs churn")
    save("04_tenure_boxplot.png")

    # 5 monthly charges box plot
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.xticks([0, 1], ["Stayed", "Churned"])
    plt.title("Monthly charges vs churn")
    save("05_monthly_charges_boxplot.png")

    # 6 churn rate by contract
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Contract", y="Churn", errorbar=None)
    plt.ylabel("Churn rate")
    plt.title("Churn rate by contract type")
    save("06_churn_by_contract.png")

    # 7 churn rate by tenure band
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="TenureBand", y="Churn", errorbar=None)
    plt.ylabel("Churn rate")
    plt.title("Churn rate by tenure band")
    save("07_churn_by_tenure_band.png")

    # 8 churn rate by internet service
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="InternetService", y="Churn", errorbar=None)
    plt.ylabel("Churn rate")
    plt.title("Churn rate by internet service")
    save("08_churn_by_internet_service.png")

    # 9 churn rate by payment method
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, y="PaymentMethod", x="Churn", errorbar=None)
    plt.xlabel("Churn rate")
    plt.title("Churn rate by payment method")
    save("09_churn_by_payment_method.png")

    # 10 heatmap: churn rate by contract x tenure band
    pivot = df.pivot_table(
        index="Contract", columns="TenureBand", values="Churn",
        aggfunc="mean", observed=False,
    )
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds")
    plt.title("Churn rate: contract type x tenure band")
    save("10_heatmap_contract_tenure.png")

    # 11 correlation heatmap (numeric features)
    num = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
              "NumServices", "Churn"]]
    plt.figure(figsize=(7, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    save("11_correlation_heatmap.png")

    if feature_importance is not None:
        top = feature_importance.sort_values(ascending=False).head(12)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top.values, y=top.index)
        plt.title("Top features (Random Forest)")
        save("12_feature_importance.png")


# --------------------------------------------------------------------------
# 5. Modelling
# --------------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> pd.Series:
    target = "Churn"
    drop_cols = ["customerID", target]
    X = df.drop(columns=drop_cols)
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def make_pre(scale: bool):
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])

    models = {
        "Logistic Regression": Pipeline([
            ("pre", make_pre(True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("pre", make_pre(False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
    }

    rows, importance = [], None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, pred, output_dict=True)
        auc = roc_auc_score(y_test, proba)
        rows.append({
            "model": name,
            "accuracy": round(report["accuracy"], 3),
            "precision_churn": round(report["1"]["precision"], 3),
            "recall_churn": round(report["1"]["recall"], 3),
            "f1_churn": round(report["1"]["f1-score"], 3),
            "roc_auc": round(auc, 3),
        })
        if name == "Random Forest":
            names = model.named_steps["pre"].get_feature_names_out()
            importance = pd.Series(
                model.named_steps["clf"].feature_importances_,
                index=[n.split("__", 1)[1] for n in names],
            )

    results = pd.DataFrame(rows)
    print("\n=== MODEL RESULTS (test set, 20% hold-out) ===")
    print(results.to_string(index=False))
    results.to_csv(OUT_DIR / "model_results.csv", index=False)
    return importance


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = engineer_features(load_and_clean(args.data))
    print_key_stats(df)
    importance = train_models(df)
    make_figures(df, importance)
    print(f"\nDone. Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
