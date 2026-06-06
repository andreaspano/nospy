"""
Feature extraction and summarization for time series.
"""

import numpy as np
import pandas as pd

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
# FeaturesCalculator — future extension point for feature extraction
# ============================================================


class FeaturesCalculator:
    """Encapsulates feature extraction and summarization for a collection of time series."""

    def __init__(self, df: pd.DataFrame, use_views: bool = True, min_length: int = 20) -> None:
        self.df = df
        self.use_views = use_views
        self.min_length = min_length
        self._features_df: pd.DataFrame | None = None

    def compute_features(self) -> pd.DataFrame:
        """Compute feature DataFrame for all series in this calculator."""
        self._features_df = build_feature_dataframe(
            self.df, use_views=self.use_views, min_length=self.min_length
        )
        return self._features_df

    def summarize(self) -> dict:
        """Return a compact, distributional summary of the computed features."""
        if self._features_df is None:
            self.compute_features()
        return summarize_features(self._features_df)

    def build_prompt(
        self,
        model_name: str,
        h: int,
        config=None,
        existing_json: dict | None = None,
        llm_config=None,
    ) -> str:
        """Build a GPT prompt for generating a model.json for *model_name*."""
        from nospy.prompt import build_model_prompt

        summary = self.summarize()
        return build_model_prompt(
            summary=summary,
            model_name=model_name,
            h=h,
            config=config,
            existing_json=existing_json,
        )


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

    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    return y.to_numpy(dtype=float)


def safe_update(
    features: dict, func, y: np.ndarray, freq: int, prefix: str | None = None
) -> dict:
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


def _apply_group(
    features: dict,
    funcs: list,
    y: np.ndarray,
    freq: int,
    prefix: str | None,
) -> None:
    """Apply a list of tsfeatures functions and merge results into features."""
    temp: dict = {}
    for fn in funcs:
        temp = safe_update(temp, fn, y, freq)
    add = (lambda k: f"{prefix}_{k}") if prefix else (lambda k: k)
    features.update({add(k): v for k, v in temp.items()})


def compute_single_view_features(
    y: np.ndarray,
    freq: int,
    prefix: str | None = None,
    min_length: int = 20,
    include_catch22: bool = True,
    include_model_shape: bool = True,
) -> dict:
    """
    Compute all feature groups for a single numeric time series.

    Args:
        y: numeric numpy array
        freq: seasonal frequency
        prefix: optional prefix for feature names
        min_length: minimum required observations
        include_catch22: whether to compute catch22
        include_model_shape: whether to compute holt/hw exponential-smoothing
            parameters. Expensive (~180ms/call). Only meaningful on level
            (price) series; disable for returns/volatility views.

    Returns:
        dict of features
    """

    features = {}

    if len(y) < min_length:
        return features

    def add_name(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    _apply_group(features, [stl_features], y, freq, prefix)
    _apply_group(features, [acf_features, pacf_features], y, freq, prefix)
    _apply_group(features, [entropy, hurst, nonlinearity], y, freq, prefix)
    _apply_group(features, [lumpiness, stability, heterogeneity], y, freq, prefix)
    _apply_group(features, [unitroot_kpss, unitroot_pp], y, freq, prefix)

    # Sparsity — includes a manual zero_proportion scalar
    temp: dict = {}
    temp = safe_update(temp, sparsity, y, freq)
    temp = safe_update(temp, flat_spots, y, freq)
    temp = safe_update(temp, crossing_points, y, freq)
    temp["zero_proportion"] = float(np.mean(y == 0))
    features.update({add_name(k): v for k, v in temp.items()})

    # -----------------------------
    # Model shape  (level view only — ~180ms/call, degenerate on returns/vol)
    # -----------------------------

    if include_model_shape:
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
            include_model_shape=True,
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
            include_model_shape=False,
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
            include_model_shape=False,
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
        data.groupby("unique_id", group_keys=False)
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
    features_df[numeric_cols] = features_df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    return features_df


# ============================================================
# Feature summarization
# ============================================================


def summarize_features(features_df: pd.DataFrame) -> dict:
    """
    Produce a distributional, coverage-aware summary of a features DataFrame.

    Separates the synthetic ``TOTAL`` series from bottom-level series and
    reports quantiles, coverage percentages, and threshold-based percentages
    for every key metric group.  The resulting dict is JSON-serialisable and
    is intended to be passed to an LLM to guide ``model.json`` configuration.

    Args:
        features_df: output of :func:`build_feature_dataframe`.

    Returns:
        Nested dict with sections ``dataset``, ``level``, ``returns``, and
        ``volatility``.
    """

    df = features_df.copy()

    if "unique_id" in df.columns:
        total_mask = df["unique_id"] == "TOTAL"
        df_bottom = df[~total_mask]
        df_total = df[total_mask]
    else:
        df_bottom = df
        df_total = pd.DataFrame()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _qdesc(series: pd.Series) -> dict | None:
        vals = series.dropna()
        n_total = len(series)
        if len(vals) == 0:
            return None
        return {
            "count": int(len(vals)),
            "coverage_pct": round(len(vals) / n_total * 100, 1),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "p25": round(float(vals.quantile(0.25)), 4),
            "p50": round(float(vals.median()), 4),
            "p75": round(float(vals.quantile(0.75)), 4),
        }

    def _pct_above(series: pd.Series, threshold: float) -> float | None:
        vals = series.dropna()
        if len(vals) == 0:
            return None
        return round(float((vals > threshold).mean()) * 100, 1)

    def _col(prefix: str, name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    def _get(df_: pd.DataFrame, prefix: str, name: str) -> pd.Series:
        col = _col(prefix, name)
        return df_[col] if col in df_.columns else pd.Series(dtype=float)

    def _stat(df_: pd.DataFrame, prefix: str, name: str) -> dict | None:
        return _qdesc(_get(df_, prefix, name))

    # ------------------------------------------------------------------
    # Per-view summary
    # ------------------------------------------------------------------

    def _view_summary(df_: pd.DataFrame, prefix: str) -> dict:
        v: dict = {}

        # Trend / STL
        s = _stat(df_, prefix, "trend")
        if s:
            v["trend_strength"] = s
            v["pct_strong_trend"] = _pct_above(_get(df_, prefix, "trend"), 0.5)

        s = _stat(df_, prefix, "seasonal_strength")
        if s:
            v["seasonal_strength"] = s
            v["pct_seasonal"] = _pct_above(
                _get(df_, prefix, "seasonal_strength"), 0.4
            )

        for key in ["linearity", "curvature", "e_acf1", "spike"]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        # ACF / PACF
        for key in ["x_acf1", "x_acf10", "diff1_acf1", "diff2_acf1", "seas_acf1"]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        s = _stat(df_, prefix, "x_pacf5")
        if s:
            v["x_pacf5"] = s

        # Noise / long memory
        for key in ["entropy", "hurst"]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        # Stationarity
        for key in ["unitroot_kpss", "unitroot_pp"]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        # Heterogeneity / ARCH
        for key in ["arch_acf", "garch_acf", "arch_r2", "garch_r2"]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        # Sparsity / structure
        for key in [
            "zero_proportion",
            "sparsity",
            "stability",
            "lumpiness",
            "flat_spots",
            "crossing_points",
            "nonlinearity",
        ]:
            s = _stat(df_, prefix, key)
            if s:
                v[key] = s

        return v

    # ------------------------------------------------------------------
    # Dataset-level statistics
    # ------------------------------------------------------------------

    obs_series = df_bottom["n_obs"] if "n_obs" in df_bottom.columns else pd.Series(dtype=float)
    pct_short: float | None = None
    if len(obs_series.dropna()) > 0:
        pct_short = round(float((obs_series < 20).mean()) * 100, 1)

    freq_dist: dict | None = None
    if "freq" in df_bottom.columns:
        freq_dist = {
            str(k): int(v)
            for k, v in df_bottom["freq"].value_counts().to_dict().items()
        }

    dataset_section: dict = {
        "n_series_total": len(df),
        "n_bottom_series": len(df_bottom),
        "has_total_series": len(df_total) > 0,
        "obs_count": _qdesc(obs_series),
        "pct_short_series": pct_short,
        "frequency_distribution": freq_dist,
    }

    summary: dict = {
        "dataset": dataset_section,
        "level": _view_summary(df_bottom, "level"),
        "returns": _view_summary(df_bottom, "return"),
        "volatility": _view_summary(df_bottom, "volatility"),
    }

    # Drop empty view sections
    for key in ("level", "returns", "volatility"):
        if not summary[key]:
            del summary[key]

    return summary

