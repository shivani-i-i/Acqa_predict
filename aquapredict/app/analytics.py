"""Exploratory unsupervised analytics for synthetic households."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "daily_usage_mean",
    "daily_usage_std",
    "flow_mean",
    "output_tds_mean",
    "pressure_drop_mean",
]


def household_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one synthetic or uploaded telemetry frame by household."""
    if "household_id" not in df or df.empty:
        return pd.DataFrame(columns=["household_id", *FEATURES])

    def mean_or_nan(column):
        return float(df[column].mean()) if column in df else float("nan")

    row = {
        "household_id": str(df["household_id"].iloc[-1]),
        "daily_usage_mean": mean_or_nan("usage_L"),
        "daily_usage_std": float(df["usage_L"].std(ddof=0)) if "usage_L" in df and len(df) else 0.0,
        "flow_mean": mean_or_nan("flow_Lpm"),
        "output_tds_mean": mean_or_nan("out_tds_ppm"),
        "pressure_drop_mean": mean_or_nan("dp_sediment_bar"),
    }
    return pd.DataFrame([row])


def population_features(df_population: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a telemetry table into one row per household."""
    if df_population.empty or "household_id" not in df_population:
        return pd.DataFrame(columns=["household_id", *FEATURES])
    frames = [household_features(group) for _, group in df_population.groupby("household_id")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["household_id", *FEATURES])


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    filled = np.where(np.isfinite(values), values, mean)
    return (filled - mean) / std, mean, std


def _lloyd_kmeans(values: np.ndarray, k: int, seed: int = 4242) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic NumPy Lloyd KMeans without native scikit-learn modules."""
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(values)))
    centres = [values[first].copy()]
    for _ in range(k - 1):
        distances = np.min(
            np.asarray([[np.sum((row - centre) ** 2) for centre in centres] for row in values]),
            axis=1,
        )
        total = float(distances.sum())
        index = int(rng.choice(len(values), p=distances / total)) if total > 0 else int(rng.integers(len(values)))
        centres.append(values[index].copy())

    centres_array = np.asarray(centres)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(100):
        squared = ((values[:, None, :] - centres_array[None, :, :]) ** 2).sum(axis=2)
        new_labels = squared.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            rows = values[labels == cluster]
            if len(rows):
                centres_array[cluster] = rows.mean(axis=0)
    return labels, centres_array


def _silhouette(values: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= len(values):
        return None
    silhouettes = []
    for i, row in enumerate(values):
        own = labels[i]
        own_rows = values[labels == own]
        if len(own_rows) <= 1:
            silhouettes.append(0.0)
            continue
        a = float(np.linalg.norm(row - own_rows).sum() / (len(own_rows) - 1))
        b = float("inf")
        for cluster in set(labels.tolist()) - {own}:
            other = values[labels == cluster]
            b = min(b, float(np.linalg.norm(row - other).mean()))
        silhouettes.append((b - a) / max(b, a, 1e-12))
    return float(np.mean(silhouettes))


def usage_profile(df_population: pd.DataFrame, current_id: str, current_df: pd.DataFrame,
                  seed: int = 4242) -> dict:
    """Fit Lloyd KMeans and robust distance anomaly scoring on the provided population.

    This is descriptive unsupervised analysis only. It is not filter-life
    training and must not be presented as real-world accuracy.
    """
    population = population_features(df_population)
    if len(population) < 12:
        return {
            "available": False,
            "reason": "At least 12 synthetic households are required.",
            "method": "Lloyd KMeans + robust standardized-distance score",
            "data_origin": "SYNTHETIC",
        }

    current = household_features(current_df)
    if not current.empty:
        current["household_id"] = str(current_id)

    population = population.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    current = current.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    if len(population) < 12 or current.empty:
        return {
            "available": False,
            "reason": "Too little complete telemetry for unsupervised profiling.",
            "method": "Lloyd KMeans + robust standardized-distance score",
            "data_origin": "SYNTHETIC",
        }

    population_matrix = population[FEATURES].to_numpy(dtype=float)
    scaled, mean, std = _standardize(population_matrix)
    n_clusters = min(3, max(2, len(population) // 8))
    labels, centres = _lloyd_kmeans(scaled, n_clusters, seed)

    profile = population.copy()
    profile["cluster"] = labels
    cluster_usage = profile.groupby("cluster")["daily_usage_mean"].mean().sort_values()
    cluster_names = ["Low-use profile", "Typical-use profile", "High-use profile"][:n_clusters]
    names = {cluster: name for cluster, name in zip(cluster_usage.index, cluster_names)}

    current_values = current[FEATURES].to_numpy(dtype=float)
    current_scaled = (np.where(np.isfinite(current_values), current_values, mean) - mean) / std
    squared = ((current_scaled[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    current_cluster = int(squared.argmin(axis=1)[0])
    distance = float(np.sqrt(squared[0, current_cluster]))

    all_distances = np.sqrt(((scaled[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).min(axis=1))
    threshold = float(np.quantile(all_distances, 0.92))
    anomaly = distance >= threshold
    anomaly_score = round(float(distance / max(threshold, 1e-12)), 3)
    cluster_sizes = profile["cluster"].value_counts().to_dict()

    return {
        "available": True,
        "method": "Standardized Lloyd KMeans + robust standardized-distance score",
        "data_origin": "SYNTHETIC",
        "n_households": int(len(profile)),
        "n_clusters": int(n_clusters),
        "cluster": current_cluster,
        "profile": names[current_cluster],
        "cluster_size": int(cluster_sizes.get(current_cluster, 0)),
        "distance_from_cluster_centre": round(distance, 3),
        "anomaly": bool(anomaly),
        "anomaly_score": anomaly_score,
        "silhouette_score": (None if _silhouette(scaled, labels) is None
                             else round(float(_silhouette(scaled, labels)), 3)),
        "interpretation": (
            "This pattern is rare relative to the synthetic population."
            if anomaly
            else "This pattern is consistent with the synthetic population."
        ),
        "honesty": (
            "Exploratory model only: not a filter-life label, fault diagnosis, "
            "or real-world accuracy claim."
        ),
    }
