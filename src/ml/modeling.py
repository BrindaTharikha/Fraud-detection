import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import XGBClassifier

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, num_boost_round=1000, early_stopping_rounds=50, nfold=5, seed=42):
    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.1,
        "seed": seed
    }

    # Cross-validation
    print(" Running internal XGBoost CV with early stopping...")
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        nfold=nfold,
        stratified=True,
        seed=seed,
        verbose_eval=False
    )

    best_trees = len(cv_results)
    print(f" Optimal number of trees: {best_trees}")

    # Final model with best trees
    model = XGBClassifier(
        n_estimators=best_trees,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed
    )
    model.fit(X_train, y_train)
    return model

def cross_validate_model(model, X_train, y_train, model_name="Model"):
    print(f"\n Cross-validation for {model_name}")
    f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"F1 Scores: {f1_scores}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print("\n Evaluation Results")
    print(f"             |  Not Fraud |    Fraud")
    print(f"-------------------------------------")
    print(f"   Not Fraud |   {cm[0,0]:9} | {cm[0,1]:7}")
    print(f"       Fraud |   {cm[1,0]:9} | {cm[1,1]:7}\n")
    print(classification_report(y_test, preds, digits=4))

def train_isolation_forest(X_train, contamination=0.32):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_train)
    return model

def evaluate_isolation_forest(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.where(preds == 1, 0, 1)
    cm = confusion_matrix(y_test, preds)
    print("\n Evaluation Results (Isolation Forest)")
    print(f"             |  Not Fraud |    Fraud")
    print(f"-------------------------------------")
    print(f"   Not Fraud |   {cm[0,0]:9} | {cm[0,1]:7}")
    print(f"       Fraud |   {cm[1,0]:9} | {cm[1,1]:7}\n")
    print(classification_report(y_test, preds, digits=4))
