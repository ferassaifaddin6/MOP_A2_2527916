"""KMeans clustering on scaled features only (targets never used for fitting)."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from src.config import K_RANGE, NUMERIC
from src.preprocessing import build_preprocessor

r4 = lambda x: round(float(x), 4)  # noqa: E731


def run_clustering(Xtr: pd.DataFrame, train_yield, train_risk, seed):
    """Pick K in 2..6 by silhouette; return (report, fitted pipeline, Z_train, labels_train)."""
    prep = build_preprocessor()
    Z = prep.fit_transform(Xtr)
    scores, models = {}, {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        scores[k] = r4(silhouette_score(Z, km.labels_))
        models[k] = km
    best_k = max(scores, key=scores.get)
    km = models[best_k]
    labels = km.labels_
    pipe = Pipeline([("prep", prep), ("kmeans", km)])

    overall_mean, overall_std = Xtr[NUMERIC].mean(), Xtr[NUMERIC].std()
    profiles = {}
    for c in range(best_k):
        mask = labels == c
        sub = Xtr[mask]
        z = (sub[NUMERIC].mean() - overall_mean) / overall_std
        traits = [f"{'high' if v > 0 else 'low'} {col}" for col, v in z.items() if abs(v) >= 0.4]
        traits.append("mostly season " + ("A" if (sub["season"] == "A").mean() >= 0.5 else "B"))
        traits.append("mostly supported by the cooperative" if sub["cooperative_support"].mean() >= 0.5
                      else "mostly without cooperative support")
        profiles[f"cluster_{c}"] = {
            "size": int(mask.sum()),
            "feature_means": {k: r4(v) for k, v in sub[NUMERIC].mean().items()},
            "share_season_A": r4((sub["season"] == "A").mean()),
            "share_cooperative_support": r4(sub["cooperative_support"].mean()),
            "post_hoc_mean_yield_tons": r4(train_yield[mask].mean()),
            "post_hoc_high_risk_rate": r4(train_risk[mask].mean()),
            "plain_language_profile": "Farms with " + ", ".join(traits) + "."}
    report = {"silhouette_by_k": {str(k): v for k, v in scores.items()}, "selected_k": best_k,
              "selection_reason": f"K={best_k} has the highest silhouette score ({scores[best_k]}).",
              "profiles": profiles,
              "interpretation": ("Clusters are fitted on scaled inputs only; yield and risk are shown "
                                 "post hoc. Because inputs are independently simulated, silhouette values "
                                 "are low and clusters are loose groupings, not natural farm types.")}
    return report, pipe, Z, labels
