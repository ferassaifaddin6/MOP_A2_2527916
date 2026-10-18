"""One-command pipeline: data -> validation -> training -> artifacts.

Example: python run_all.py --student-id ug2527916@ines.ac.rw --seed 7916
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.clustering import run_clustering
from src.config import FEATURES, ROOT, TARGET_CLS, TARGET_REG, Paths
from src.data_generation import write_dataset
from src.modeling import train_classification, train_regression
from src.preprocessing import split_data
from src.reporting import plot_clusters, plot_confusion, plot_residuals, write_json
from src.validation import DataValidationError, clean, parse_identity, validate_raw


def main(student_id: str, seed: int, root: Path = ROOT) -> dict:
    """Run the whole pipeline and return a summary dict."""
    ident = parse_identity(student_id, seed)
    code, seed = ident["verification_code"], ident["seed"]
    paths = Paths(Path(root), ident["regdigits"])
    paths.ensure()
    print(f"Verification code: {code}")

    # 1) data
    digest = write_dataset(seed, paths.csv, paths.fingerprint, code)
    print(f"SHA-256 fingerprint: {digest}")
    raw = pd.read_csv(paths.csv)
    quality = validate_raw(raw)
    df = clean(raw)
    quality.update({"rows_after_cleaning": int(len(df)), "duplicates_removed": int(len(raw) - len(df))})
    print("Data quality:", json.dumps(quality))

    # 2) split (fixed, before any fitting)
    Xtr, Xte, ytr, yte, ctr, cte = split_data(df, seed)
    head = {"verification_code": code, "student_id": student_id, "seed": seed,
            "dataset": "SYNTHETIC", "n_train": int(len(Xtr)), "n_test": int(len(Xte))}

    # 3) models
    reg_metrics, reg_model, reg_pred = train_regression(Xtr, ytr, Xte, yte, seed)
    cls_metrics, cls_model, cls_pred, cls_prob = train_classification(Xtr, ctr, Xte, cte, seed)
    clu_report, clu_model, Ztr, ltr = run_clustering(Xtr, ytr, ctr, seed)
    test_clusters = clu_model.predict(Xte)

    # 4) error examples (for the student's own written interpretation)
    abs_err = np.abs(yte.values - reg_pred)
    i = int(abs_err.argmax())
    reg_metrics["largest_error_example"] = {
        "row_id": int(Xte.index[i]), "actual": round(float(yte.iloc[i]), 4),
        "predicted": round(float(reg_pred[i]), 4), "features": Xte.iloc[i].to_dict()}
    wrong = np.where(cls_pred != cte.values)[0]
    if len(wrong):
        j = int(wrong[np.abs(cls_prob[wrong] - cte.values[wrong]).argmax()])
        cls_metrics["most_confident_error_example"] = {
            "row_id": int(Xte.index[j]), "actual": int(cte.iloc[j]), "predicted": int(cls_pred[j]),
            "probability_high_risk": round(float(cls_prob[j]), 4), "features": Xte.iloc[j].to_dict()}

    # 5) artifacts
    write_json(paths.artifacts / "regression_metrics.json", {**head, **reg_metrics})
    write_json(paths.artifacts / "classification_metrics.json", {**head, **cls_metrics})
    write_json(paths.artifacts / "clustering_report.json", {**head, **clu_report})
    write_json(paths.artifacts / "data_quality.json", {**head, **quality})
    pd.DataFrame({"row_id": Xte.index, "actual_yield_tons": yte.values.round(4),
                  "predicted_yield_tons": reg_pred.round(4),
                  "yield_error": (yte.values - reg_pred).round(4),
                  "abs_yield_error": abs_err.round(4),
                  "actual_high_risk": cte.values, "predicted_high_risk": cls_pred,
                  "probability_high_risk": cls_prob.round(4),
                  "cluster": test_clusters}).to_csv(paths.artifacts / "test_predictions.csv",
                                                    index=False, lineterminator="\n")
    plot_residuals(paths.plots / "residuals.png", yte.values, reg_pred, code)
    plot_confusion(paths.plots / "confusion_matrix.png", cte.values, cls_pred, code)
    plot_clusters(paths.plots / "clusters.png", Ztr, ltr, code)

    # 6) models
    joblib.dump(reg_model, paths.models / "regressor.joblib")
    joblib.dump(cls_model, paths.models / "classifier.joblib")
    joblib.dump(clu_model, paths.models / "clusterer.joblib")
    write_json(paths.models / "model_meta.json", {
        **head, "features": FEATURES, "sha256": digest,
        "regression_model": reg_metrics["selected_model"],
        "classification_model": cls_metrics["selected_model"],
        "selected_k": clu_report["selected_k"],
        "regression_test": reg_metrics["models"][reg_metrics["selected_model"]],
        "classification_test": {k: v for k, v in
                                cls_metrics["models"][cls_metrics["selected_model"]].items()}})

    produced = sorted(str(p.relative_to(paths.root)) for p in
                      list(paths.artifacts.rglob("*.*")) + list(paths.models.glob("*.*")))
    print("Artifacts:\n  " + "\n  ".join(produced))
    print(f"Regression: {reg_metrics['selected_model']} {reg_metrics['models'][reg_metrics['selected_model']]}")
    print(f"Classification: {cls_metrics['selected_model']} "
          f"{cls_metrics['models'][cls_metrics['selected_model']]}")
    print(f"Clustering: K={clu_report['selected_k']}")
    return {"identity": ident, "sha256": digest, "regression": reg_metrics, "classification": cls_metrics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    try:
        main(a.student_id, a.seed)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
