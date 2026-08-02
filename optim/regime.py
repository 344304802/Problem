import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def cluster_regimes(df, k = 5, seed = 42) :
    feats = df[["C_in_gNm3", "Temp_C", "Q_Nm3h"]].values
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)

    # 用轮廓系数在 [2, 8] 选 K, 同时要求每工况样本 >= 3% 总样本
    best_k, best_sil = None, -1.0
    print("[INFO] 轮廓系数选K:")
    for k_try in range(2, 9) :
        km_try = KMeans(n_clusters = k_try, random_state = seed, n_init = 10)
        labels_try = km_try.fit_predict(feats_s)
        counts = np.bincount(labels_try)
        if counts.min() / len(df) < 0.03 :
            print(f"  k = {k_try} : 最小工况占比 {counts.min()/len(df):.3f}<3%, 跳过")
            continue
        sil = silhouette_score(feats_s, labels_try)
        print(f"  k = {k_try} : silhouette = {sil:.4f}")
        if sil > best_sil :
            best_sil = sil
            best_k = k_try
    if best_k is None :
        best_k = k
        best_sil = float("nan")
    k = best_k
    print(f"[INFO] 选定 k = {k} (silhouette = {best_sil:.4f})")

    km = KMeans(n_clusters = k, random_state = seed, n_init = 10)
    labels = km.fit_predict(feats_s)

    regimes = []
    for i in range(k) :
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

    return {"regimes": regimes, "labels": labels.tolist(), "k": k, "scaler": scaler, "centers": km.cluster_centers_, "silhouette": float(best_sil)}