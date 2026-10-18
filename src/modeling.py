"""Regression and classification: baseline vs candidates, CV-based selection."""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

r4 = lambda x: round(float(x), 4)  # noqa: E731


def _pipe(est):
    return Pipeline([("prep", build_preprocessor()), ("model", est)])


def train_regression(Xtr, ytr, Xte, yte, seed):
    """Return (metrics dict, fitted pipeline, predictions on test)."""
    cands = {"mean_baseline": DummyRegressor(strategy="mean"),
             "linear_regression": LinearRegression(),
             "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                    random_state=seed, n_jobs=1),
             "gradient_boosting": GradientBoostingRegressor(random_state=seed)}
    cv = KFold(5, shuffle=True, random_state=seed)
    results, fitted, preds = {}, {}, {}
    for name, est in cands.items():
        pipe = _pipe(est)
        cv_rmse = -cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        results[name] = {"cv_rmse_train": r4(cv_rmse), "MAE": r4(mean_absolute_error(yte, p)),
                         "RMSE": r4(np.sqrt(mean_squared_error(yte, p))), "R2": r4(r2_score(yte, p))}
        fitted[name], preds[name] = pipe, p
    sel = min((n for n in cands if n != "mean_baseline"), key=lambda n: results[n]["cv_rmse_train"])
    reason = (f"{sel} has the lowest 5-fold cross-validated RMSE on the training set "
              f"({results[sel]['cv_rmse_train']}); test metrics were not used for selection.")
    return {"models": results, "selected_model": sel, "selection_reason": reason}, fitted[sel], preds[sel]


def train_classification(Xtr, ytr, Xte, yte, seed):
    """Return (metrics dict, fitted pipeline, predicted labels, P(high_risk))."""
    cands = {"majority_baseline": DummyClassifier(strategy="most_frequent"),
             "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
             "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                     random_state=seed, n_jobs=1),
             "gradient_boosting": GradientBoostingClassifier(random_state=seed)}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    results, fitted, outs = {}, {}, {}
    for name, est in cands.items():
        pipe = _pipe(est)
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean()
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        auc = r4(roc_auc_score(yte, proba)) if yte.nunique() == 2 else None
        results[name] = {"cv_f1_train": r4(cv_f1),
                         "precision": r4(precision_score(yte, pred, zero_division=0)),
                         "recall": r4(recall_score(yte, pred, zero_division=0)),
                         "f1": r4(f1_score(yte, pred, zero_division=0)),
                         "roc_auc": auc,
                         "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist()}
        fitted[name], outs[name] = pipe, (pred, proba)
    sel = max((n for n in cands if n != "majority_baseline"), key=lambda n: results[n]["cv_f1_train"])
    reason = (f"{sel} has the highest 5-fold cross-validated F1 on the training set "
              f"({results[sel]['cv_f1_train']}); positive class = high_risk (1).")
    return ({"models": results, "selected_model": sel, "selection_reason": reason,
             "confusion_matrix_labels": "rows=actual [0,1], columns=predicted [0,1]"},
            fitted[sel], *outs[sel])
