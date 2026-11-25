import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/dataset.csv")
if len(df) > 80000:
    df = df.sample(n=80000, random_state=42)

target_col = "Class" if "Class" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        4,
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [8, 12, None], "min_samples_split": [2, 5]},
        5,
    ),
    "GaussianNB": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        3,
    ),
}

best_score = -1
best_model = None
best_name = ""

for name, (estimator, param_dist, n_iter) in models.items():
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    preds = search.predict(X_test)
    score = f1_score(y_test, preds, zero_division=0)

    proba = search.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(rec, prec)

    print(f"\n=== {name} ===")
    print("Best Params:", search.best_params_)
    print(f"F1-Score (Fraud): {score:.4f} | PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    if pr_auc > best_score:
        best_score = pr_auc
        best_model = search.best_estimator_
        best_name = name

joblib.dump(
    {"model": best_model, "features": X.columns.tolist(), "target": target_col, "best_name": best_name},
    "app/best_model.pkl",
)
print(f"\nModel terbaik tersimpan ke app/best_model.pkl: {best_name} (PR-AUC = {best_score:.4f})")