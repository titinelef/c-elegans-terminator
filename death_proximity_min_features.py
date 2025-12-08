# ================================================================
#   Death Proximity Classification — Minimal Feature Model
#   Author: (Ton Nom)
#   Goal: classifier les segments en "close to death" (1) ou non (0)
#         en utilisant un nombre MINIMAL de features
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


# ================================================================
# 1. Charger le CSV des features par segment
# ================================================================

FEATURE_FILE = "feature_data/segments_features.csv"

df = pd.read_csv(FEATURE_FILE)
print(f"Loaded segments: {len(df)}")


# ================================================================
# 2. Recréer la colonne segment_index (si absente)
# ================================================================

if "segment_index" not in df.columns or df["segment_index"].isna().all():
    df["segment_index"] = df["filename"].str.extract(
        r"segment(\d+(?:\.\d+)?)", expand=False
    ).astype(float)

# Faire en sorte que les noms de worm soient corrects
df["original_file"] = df["original_file"].astype(str)


# ================================================================
# 3. Calculer la distance à la mort en segments
# ================================================================

max_seg = df.groupby("original_file")["segment_index"].max()

df["segments_from_end"] = df.apply(
    lambda row: max_seg[row["original_file"]] - row["segment_index"], axis=1
)


# ================================================================
# 4. Définir le label binaire "close_to_death"
#    Ici: segments situés dans les 20 derniers (modifiable)
# ================================================================

PROXIMITY_THRESHOLD = 20

df["close_to_death"] = (df["segments_from_end"] <= PROXIMITY_THRESHOLD).astype(int)

y = df["close_to_death"]
groups = df["original_file"]


# ================================================================
# 5. Sélection des features
#    On définit d'abord les top aging features fournis par l'assistant
# ================================================================

top_aging_features = [
    "high_activity_fraction", "mixed_activity_fraction", "low_activity_fraction",
    "mean_speed", "std_speed", "max_speed", "speed_entropy",
    "mean_roaming_score", "std_roaming_score", "fraction_roaming",
    "movement_efficiency", "fraction_efficient_movement",
    "time_paused", "fraction_paused",
    "mean_jerk", "max_jerk", "kinetic_energy_proxy",
    "mean_meandering_ratio", "std_meandering_ratio",
    "wavelet_speed_level0", "wavelet_speed_level1", "wavelet_speed_level2", "wavelet_speed_level3",
    "mean_frenetic_score", "std_frenetic_score",
    "speed_persistence", "activity_level", "speed_skewness", "speed_kurtosis"
]

# Garder seulement les features présentes dans le CSV
available_features = [f for f in top_aging_features if f in df.columns]


# Un sous-ensemble ordonné de features (basé sur leur importance connue)
ranked_core = [
    "mean_speed",
    "speed_entropy",
    "mean_roaming_score",
    "fraction_paused",
    "movement_efficiency",
    "high_activity_fraction"
]

feature_subsets = {
    "1_feature": ranked_core[:1],
    "2_features": ranked_core[:2],
    "3_features": ranked_core[:3],
    "4_features": ranked_core[:4],
    "6_features": ranked_core[:6],
    "all_top_aging": available_features,
}


# ================================================================
# 6. Boucle de cross-val par nombre de features
# ================================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

results = []
print("\nRunning cross-validation...\n")

for name, feats in feature_subsets.items():
    print(f"--- Testing subset: {name} ({len(feats)} features) ---")

    # X = features de ce subset, NaN -> mediane
    X = df[feats].fillna(df[feats].median())

    # Modèle compact mais performant
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ))
    ])

    f1s, accs, aucs = [], [], []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)

        f1s.append(f1_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))
        aucs.append(roc_auc_score(y_te, proba))

    results.append({
        "subset": name,
        "n_features": len(feats),
        "F1_mean": np.mean(f1s),
        "ACC_mean": np.mean(accs),
        "AUC_mean": np.mean(aucs),
    })

print("\n=== RESULTS (Minimal Features Model) ===\n")
for r in results:
    print(
        f"{r['subset']:15s} | "
        f"{r['n_features']:2d} feats | "
        f"F1={r['F1_mean']:.3f} | "
        f"AUC={r['AUC_mean']:.3f} | "
        f"ACC={r['ACC_mean']:.3f}"
    )
