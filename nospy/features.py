import numpy as np
import pandas as pd





# ============================================================
# FeaturesCalculator class for time series feature extraction
# ============================================================

import warnings
warnings.filterwarnings("ignore")


from tsfeatures import (
    acf_features,
    pacf_features,
    stl_features,
    entropy,
    hurst,
    lumpiness,
    stability,
    nonlinearity,
    unitroot_kpss,
    unitroot_pp,
    heterogeneity,
    holt_parameters,
    hw_parameters,
    sparsity,
    flat_spots,
    crossing_points,
)

import pycatch22


# ============================================================
# Feature groups
# ============================================================


class FeaturesCalculator:
    FEATURE_GROUPS = {
        "metadata": [
            "n_obs",
            "freq",
        ],
        "trend_seasonality": [
            "trend",
            "seasonal_strength",
            "spike",
            "linearity",
            "curvature",
            "peak",
            "trough",
        ],
        "autocorrelation": [
            "x_acf1",
            "x_acf10",
            "diff1_acf1",
            "diff1_acf10",
            "diff2_acf1",
            "diff2_acf10",
            "x_pacf5",
        ],
        "forecastability": [
            "entropy",
            "hurst",
            "nonlinearity",
        ],
        "stability": [
            "lumpiness",
            "stability",
            "heterogeneity",
        ],
        "stationarity": [
            "unitroot_kpss",
            "unitroot_pp",
        ],
        "sparsity": [
            "sparsity",
            "zero_proportion",
            "flat_spots",
            "crossing_points",
        ],
        "model_shape": [
            "holt_alpha",
            "holt_beta",
            "hw_alpha",
            "hw_beta",
            "hw_gamma",
        ],
        "catch22": "all",
    }

    def __init__(self, df, use_views=True, min_length=20):
        self.df = df
        self.use_views = use_views
        self.min_length = min_length

    def compute_features(self):
        return build_feature_dataframe(self.df, use_views=self.use_views, min_length=self.min_length)



# ============================================================
# Frequency inference
# ============================================================

def infer_seasonal_frequency(ds: pd.Series) -> int:
    """
    Infer seasonal frequency from datetime index.

    Returns:
        Daily    -> 7
        Weekly   -> 52
        Monthly  -> 12
        Quarterly-> 4
        Hourly   -> 24
        Else     -> 1
    """

    ds = pd.to_datetime(ds).sort_values()

    try:
        inferred = pd.infer_freq(ds)
    except Exception:
        inferred = None

    if inferred is None:
        return 1

    inferred = inferred.upper()

    if inferred.startswith("H"):
        return 24

    if inferred.startswith("D"):
        return 7

    if inferred.startswith("W"):
        return 52

    if inferred.startswith("M"):
        return 12

    if inferred.startswith("Q"):
        return 4

    return 1


# ============================================================
# Utility functions
# ============================================================

def clean_series(y: pd.Series) -> np.ndarray:
    """
    Convert series to clean numeric numpy array.
    """

    y = (
        pd.to_numeric(y, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return y.to_numpy(dtype=float)


def safe_update(features: dict, func, y: np.ndarray, freq: int, prefix: str | None = None) -> dict:
    """
    Safely call a tsfeatures function and update feature dictionary.
    """

    try:
        out = func(y, freq=freq)

        if not isinstance(out, dict):
            return features

        if prefix is not None:
            out = {f"{prefix}_{k}": v for k, v in out.items()}

        features.update(out)

    except Exception:
        pass

    return features


def safe_catch22(y: np.ndarray, prefix: str = "catch22") -> dict:
    """
    Compute catch22 features safely.
    """

    try:
        out = pycatch22.catch22_all(y)

        return {
            f"{prefix}_{name}": value
            for name, value in zip(out["names"], out["values"])
        }

    except Exception:
        return {}


# ============================================================
# Single-view feature extraction
# ============================================================

def compute_single_view_features(
    y: np.ndarray,
    freq: int,
    prefix: str | None = None,
    min_length: int = 20,
    include_catch22: bool = True,
) -> dict:
    """
    Compute all feature groups for a single numeric time series.

    Args:
        y: numeric numpy array
        freq: seasonal frequency
        prefix: optional prefix for feature names
        min_length: minimum required observations
        include_catch22: whether to compute catch22

    Returns:
        dict of features
    """

    features = {}

    if len(y) < min_length:
        return features

    def add_name(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    # -----------------------------
    # Trend / Seasonality
    # -----------------------------

    temp = {}
    temp = safe_update(temp, stl_features, y, freq)
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Autocorrelation
    # -----------------------------

    temp = {}
    temp = safe_update(temp, acf_features, y, freq)
    temp = safe_update(temp, pacf_features, y, freq)
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Forecastability
    # -----------------------------

    temp = {}
    temp = safe_update(temp, entropy, y, freq)
    temp = safe_update(temp, hurst, y, freq)
    temp = safe_update(temp, nonlinearity, y, freq)
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Stability
    # -----------------------------

    temp = {}
    temp = safe_update(temp, lumpiness, y, freq)
    temp = safe_update(temp, stability, y, freq)
    temp = safe_update(temp, heterogeneity, y, freq)
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Stationarity
    # -----------------------------

    temp = {}
    temp = safe_update(temp, unitroot_kpss, y, freq)
    temp = safe_update(temp, unitroot_pp, y, freq)
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Sparsity
    # -----------------------------

    temp = {}
    temp = safe_update(temp, sparsity, y, freq)
    temp = safe_update(temp, flat_spots, y, freq)
    temp = safe_update(temp, crossing_points, y, freq)

    temp["zero_proportion"] = float(np.mean(y == 0))

    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Model shape
    # -----------------------------

    try:
        holt = holt_parameters(y, freq=freq)
        for k, v in holt.items():
            features[add_name(f"holt_{k}")] = v
    except Exception:
        pass

    try:
        hw = hw_parameters(y, freq=freq)
        for k, v in hw.items():
            features[add_name(f"hw_{k}")] = v
    except Exception:
        pass

    # -----------------------------
    # catch22
    # -----------------------------

    if include_catch22:
        catch_prefix = add_name("catch22")
        features.update(safe_catch22(y, prefix=catch_prefix))

    return features


# ============================================================
# Multi-view feature extraction
# ============================================================

def compute_features_for_group(
    group: pd.DataFrame,
    use_views: bool = True,
    min_length: int = 20,
) -> pd.Series:
    """
    Compute features for one unique_id.

    If use_views=True, computes:
        - price features on y
        - return features on log returns
        - volatility features on abs(log returns)

    If use_views=False, computes only raw y features.
    """

    group = group.sort_values("ds")

    y_raw = clean_series(group["y"])
    freq = infer_seasonal_frequency(group["ds"])

    features = {
        "n_obs": len(y_raw),
        "freq": freq,
    }

    if len(y_raw) < min_length:
        return pd.Series(features)

    # -----------------------------
    # View 1: raw level / price / original y
    # -----------------------------

    if use_views:
        level_features = compute_single_view_features(
            y=y_raw,
            freq=freq,
            prefix="level",
            min_length=min_length,
            include_catch22=True,
        )

        features.update(level_features)

        # -----------------------------
        # View 2: log returns
        # -----------------------------

        if np.all(y_raw > 0):
            log_y = np.log(y_raw)
            returns = np.diff(log_y)
        else:
            returns = np.diff(y_raw)

        returns = returns[np.isfinite(returns)]

        return_features = compute_single_view_features(
            y=returns,
            freq=freq,
            prefix="return",
            min_length=min_length,
            include_catch22=True,
        )

        features.update(return_features)

        # -----------------------------
        # View 3: volatility proxy
        # -----------------------------

        volatility = np.abs(returns)
        volatility = volatility[np.isfinite(volatility)]

        volatility_features = compute_single_view_features(
            y=volatility,
            freq=freq,
            prefix="volatility",
            min_length=min_length,
            include_catch22=True,
        )

        features.update(volatility_features)

    else:
        raw_features = compute_single_view_features(
            y=y_raw,
            freq=freq,
            prefix=None,
            min_length=min_length,
            include_catch22=True,
        )

        features.update(raw_features)

    return pd.Series(features)


# ============================================================
# Main function
# ============================================================


def build_feature_dataframe(
    df: pd.DataFrame,
    use_views: bool = True,
    min_length: int = 20,
) -> pd.DataFrame:
    """
    Build feature dataframe from long-format time series data.

    Required input columns:
        unique_id, ds, y
    """

    required_cols = {"unique_id", "ds", "y"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values(["unique_id", "ds"])

    features_df = (
        data
        .groupby("unique_id", group_keys=False)
        .apply(
            lambda g: compute_features_for_group(
                g,
                use_views=use_views,
                min_length=min_length,
            )
        )
        .reset_index()
    )

    # Replace infinite values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = (
        features_df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
    )

    return features_df


# ============================================================
# Optional: compact LLM profile
# ============================================================

def build_llm_profile(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact interpretable feature set for the LLM.

    This avoids passing hundreds of raw features to the LLM.
    """

    preferred_cols = [
        "unique_id",
        "n_obs",
        "freq",

        # level
        "level_trend",
        "level_seasonal_strength",
        "level_spike",
        "level_linearity",
        "level_curvature",
        "level_x_acf1",
        "level_x_acf10",
        "level_x_pacf5",
        "level_entropy",
        "level_hurst",
        "level_nonlinearity",
        "level_lumpiness",
        "level_stability",
        "level_heterogeneity",
        "level_unitroot_kpss",
        "level_unitroot_pp",

        # returns
        "return_x_acf1",
        "return_x_acf10",
        "return_x_pacf5",
        "return_entropy",
        "return_hurst",
        "return_nonlinearity",
        "return_lumpiness",
        "return_stability",
        "return_heterogeneity",
        "return_unitroot_kpss",
        "return_unitroot_pp",

        # volatility
        "volatility_x_acf1",
        "volatility_x_acf10",
        "volatility_x_pacf5",
        "volatility_entropy",
        "volatility_hurst",
        "volatility_nonlinearity",
        "volatility_lumpiness",
        "volatility_stability",
        "volatility_heterogeneity",
    ]

    existing_cols = [
        col for col in preferred_cols
        if col in features_df.columns
    ]

    return features_df[existing_cols].copy()



# ============================================================
# Usage example (uncomment and adapt for standalone testing)
# ============================================================
# if __name__ == "__main__":
#     # df must have: unique_id | ds | y
#     features_df = build_feature_dataframe(
#         df=ts,
#         use_views=True,
#         min_length=20,
#     )
#     llm_profile_df = build_llm_profile(features_df)

