"""
Hierarchical forecast reconciliation for a 2-level hierarchy.

Hierarchy:  TOTAL (top level)  =  sum of individual series (bottom level).

For reconciliation to produce a non-trivial adjustment, TOTAL must be
independently forecasted by the model (not computed as a post-hoc sum).
When the base forecasts are incoherent (TOTAL_forecast ≠ sum of bottom
forecasts), reconciliation finds the nearest coherent solution.

Supported methods
-----------------
BottomUp : Keep bottom-level base forecasts unchanged; recompute TOTAL
           as their sum.  Ignores the top-level base forecast entirely.
OLS      : MinTrace Ordinary Least Squares projection (Wickramasuriya 2019).
wls_var  : MinTrace Weighted Least Squares using per-series, per-model
           in-sample residual variance as diagonal weights.

References
----------
Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    Optimal forecast reconciliation using unbiased MinT shrinkage.
    Journal of the American Statistical Association.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SUPPORTED_METHODS: frozenset[str] = frozenset({"BottomUp", "OLS", "wls_var"})
_TOTAL_ID: str = "TOTAL"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summing_matrix(n: int) -> np.ndarray:
    """
    Build the 2-level summing matrix S of shape (n+1, n).

    Row 0 → TOTAL  (= sum of all bottom series).
    Rows 1..n → individual bottom-level series (identity block).
    """
    S = np.zeros((n + 1, n))
    S[0, :] = 1.0
    S[1:, :] = np.eye(n)
    return S


def _projection_matrix(
    S: np.ndarray,
    method: str,
    W_diag: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the reconciliation matrix M = S @ G so that
    ŷ_reconciled = M @ ŷ_base.

    Parameters
    ----------
    S       : summing matrix of shape (m, n), where m = n + 1.
    method  : "BottomUp", "OLS", or "wls_var".
    W_diag  : length-m diagonal of the weight matrix W (wls_var only).

    Returns
    -------
    M : ndarray of shape (m, m).
    """
    n = S.shape[1]
    m = S.shape[0]

    if method == "BottomUp":
        # G selects only the bottom-level forecasts (columns 1..m in ŷ_base)
        G = np.zeros((n, m))
        G[:, 1:] = np.eye(n)
        return S @ G

    # OLS and wls_var: G = (S' W⁻¹ S)⁻¹ S' W⁻¹
    W_inv = np.diag(1.0 / np.maximum(W_diag, 1e-10)) if W_diag is not None else np.eye(m)
    St_Winv = S.T @ W_inv
    G = np.linalg.solve(St_Winv @ S, St_Winv)
    return S @ G


def _residual_variances(
    df_cv: pd.DataFrame,
    model: str,
    ordered_ids: list[str],
) -> np.ndarray:
    """
    Compute per-series in-sample residual variance for one model column.

    Parameters
    ----------
    df_cv       : cross-validation results.
    model       : name of the model forecast column.
    ordered_ids : series IDs in the same row order as S.

    Returns
    -------
    1-D array of variances of length len(ordered_ids).
    """
    variances: list[float] = []
    for uid in ordered_ids:
        sub = df_cv[df_cv["unique_id"] == uid]
        var = float((sub["y"] - sub[model]).var(ddof=1))
        variances.append(max(var, 1e-10))
    return np.array(variances)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile(
    df_cv: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Apply hierarchical forecast reconciliation to cross-validation results.

    TOTAL must be present as an independently forecasted series in df_cv
    (i.e. included in the training data before calling NeuralForecast).

    Parameters
    ----------
    df_cv   : DataFrame with columns
              [unique_id, ds, cutoff, y, <model_1>, ..., <model_k>].
    method  : Reconciliation method — "BottomUp", "OLS", or "wls_var".

    Returns
    -------
    Copy of df_cv with model forecast columns replaced by reconciled values.
    Column order is preserved.

    Raises
    ------
    ValueError : if method is unknown, no model columns are found, TOTAL is
                 missing, or any model column contains NaN values.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown reconciliation method '{method}'. "
            f"Choose from {sorted(_SUPPORTED_METHODS)}."
        )

    id_cols: set[str] = {"unique_id", "ds", "cutoff", "y"}
    model_cols: list[str] = [c for c in df_cv.columns if c not in id_cols]
    if not model_cols:
        raise ValueError("No model columns found in df_cv.")

    unique_ids: list[str] = sorted(df_cv["unique_id"].unique())
    if _TOTAL_ID not in unique_ids:
        raise ValueError(
            f"'{_TOTAL_ID}' not found in unique_id. "
            "Include TOTAL in the training data so it is forecasted independently."
        )

    bottom_ids: list[str] = [uid for uid in unique_ids if uid != _TOTAL_ID]
    ordered_ids: list[str] = [_TOTAL_ID] + bottom_ids  # must match row order of S

    n = len(bottom_ids)
    S = _summing_matrix(n)

    result = df_cv.copy()

    for model in model_cols:
        W_diag: np.ndarray | None = None
        if method == "wls_var":
            W_diag = _residual_variances(df_cv, model, ordered_ids)

        M = _projection_matrix(S, method, W_diag)  # (m, m)

        # Pivot: rows = (ds, cutoff), cols = unique_id (ordered to match S rows)
        pivot: pd.DataFrame = df_cv.pivot_table(
            index=["ds", "cutoff"],
            columns="unique_id",
            values=model,
            aggfunc="first",
        ).reindex(columns=ordered_ids)

        base = pivot.to_numpy(dtype=float)  # (T, m)

        if np.isnan(base).any():
            raise ValueError(
                f"NaN values in model column '{model}' prevent reconciliation. "
                "Ensure all series have complete forecasts for every (ds, cutoff)."
            )

        reconciled = base @ M.T  # (T, m)

        rec_long = (
            pd.DataFrame(reconciled, index=pivot.index, columns=pivot.columns)
            .reset_index()
            .melt(id_vars=["ds", "cutoff"], var_name="unique_id", value_name=model)
        )

        result = result.drop(columns=[model]).merge(
            rec_long, on=["ds", "cutoff", "unique_id"], how="left"
        )

    return result[df_cv.columns]
