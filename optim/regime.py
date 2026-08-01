import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_regimes(df, k=5, seed=42):
    feats = df[["C_in_gNm3", "Temp_C", "Q_Nm3h"]].values
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)

    while k >= 2:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(feats_s)
        counts = np.bincount(labels)
        min_ratio = counts.min() / len(df)
        if min_ratio >= 0.03:
            break
        print(f"[WARN] k={k} 最小工况样本占比 {min_ratio:.3f}<3%, 减少k")
        k -= 1

    regimes = []
    for i in range(k):
        mask = labels == i
        sub = df[mask]
        regimes.append({
            "id": i,
            "n": int(mask.sum()),
            "mean": {
                "C_in": float(sub["C_in_gNm3"].mean()),
                "Temp": float(sub["Temp_C"].mean()),
                "Q": float(sub["Q_Nm3h"].mean()),
            },
            "var": {
                "C_in": float(sub["C_in_gNm3"].var()),
                "Temp": float(sub["Temp_C"].var()),
                "Q": float(sub["Q_Nm3h"].var()),
            },
            "mask": mask,
        })
        print(f"[INFO] 工况{i}: n={mask.sum()}, C_in={sub['C_in_gNm3'].mean():.2f}, Temp={sub['Temp_C'].mean():.2f}")

    return {"regimes": regimes, "labels": labels.tolist(), "k": k, "scaler": scaler, "centers": km.cluster_centers_}