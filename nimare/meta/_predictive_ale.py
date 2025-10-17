"""Utilities for ALE cutoff prediction models."""

import os
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

from nimare.utils import get_resource_path

_MODEL_FILENAMES = {
    "vfwe": "vFWE_model.txt",
    "cfwe": "cFWE_model.txt",
    "tfce": "tfce_model.txt",
}


def _validate_inputs(nsub: np.ndarray, nfoci: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Validate subject and foci counts."""
    nsub = np.asarray(nsub, dtype=float)
    nfoci = np.asarray(nfoci, dtype=float)

    if nsub.ndim != 1 or nfoci.ndim != 1:
        raise ValueError("Subject and foci counts must be one-dimensional arrays.")
    if nsub.size != nfoci.size:
        raise ValueError("Subject and foci arrays must have the same length.")
    if nsub.size == 0:
        raise ValueError("At least one experiment is required for cutoff prediction.")
    if np.any(nsub <= 0):
        raise ValueError("Subject counts must be positive for all experiments.")
    if np.any(nfoci < 0):
        raise ValueError("Foci counts must be non-negative for all experiments.")

    nexp = int(nsub.size)
    if nexp > 150:
        raise ValueError(
            "Dataset contains more than 150 experiments. "
            "The predictive cutoff models were not trained on datasets of this size."
        )
    if np.max(nsub) > 300:
        raise ValueError(
            "Dataset includes experiments with more than 300 subjects. "
            "Cutoff prediction accuracy cannot be guaranteed for this sample size."
        )
    if np.max(nfoci) > 150:
        raise ValueError(
            "Dataset includes experiments with more than 150 foci. "
            "Cutoff prediction accuracy cannot be guaranteed for this number of foci."
        )
    return nexp, nsub, nfoci


def _extract_features(nexp: int, nsub: np.ndarray, nfoci: np.ndarray) -> np.ndarray:
    """Recreate the feature extraction routine from PyALE."""
    nsub_total = np.sum(nsub)
    nsub_mean = np.mean(nsub)
    nsub_median = np.median(nsub)
    nsub_std = np.std(nsub)
    nsub_max = np.max(nsub)
    nsub_min = np.min(nsub)
    if np.allclose(nsub, nsub[0]):
        nsub_skew = 0.0
        nsub_kurtosis = 0.0
    else:
        nsub_skew = float(skew(nsub))
        nsub_kurtosis = float(kurtosis(nsub))

    nfoci_total = np.sum(nfoci)
    nfoci_mean = np.mean(nfoci)
    nfoci_median = np.median(nfoci)
    nfoci_std = np.std(nfoci)
    nfoci_max = np.max(nfoci)
    nfoci_min = np.min(nfoci)
    if np.allclose(nfoci, nfoci[0]):
        nfoci_skew = 0.0
        nfoci_kurtosis = 0.0
    else:
        nfoci_skew = float(skew(nfoci))
        nfoci_kurtosis = float(kurtosis(nfoci))

    ratio = nfoci / nsub
    ratio_mean = np.mean(ratio)
    ratio_std = np.std(ratio)
    ratio_max = np.max(ratio)
    ratio_min = np.min(ratio)

    nstudies_foci_ratio = nfoci_total / nexp

    hi_mask = nsub > 20
    mi_mask = (nsub <= 20) & (nsub > 15)
    li_mask = (nsub <= 15) & (nsub > 10)
    vi_mask = nsub <= 10

    hi_foci = float(np.sum(nfoci[hi_mask]))
    mi_foci = float(np.sum(nfoci[mi_mask]))
    li_foci = float(np.sum(nfoci[li_mask]))
    vi_foci = float(np.sum(nfoci[vi_mask]))

    features = np.c_[
        nexp,
        nsub_total,
        nsub_mean,
        nsub_median,
        nsub_std,
        nsub_max,
        nsub_min,
        nsub_skew,
        nsub_kurtosis,
        nfoci_total,
        nfoci_mean,
        nfoci_median,
        nfoci_std,
        nfoci_max,
        nfoci_min,
        nfoci_skew,
        nfoci_kurtosis,
        ratio_mean,
        ratio_std,
        ratio_max,
        ratio_min,
        nstudies_foci_ratio,
        hi_foci,
        mi_foci,
        li_foci,
        vi_foci,
    ]
    return features.astype(float, copy=False)


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    """Load an XGBoost regressor for the requested cutoff."""
    if model_name not in _MODEL_FILENAMES:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {tuple(_MODEL_FILENAMES)}.")

    model_path = os.path.join(get_resource_path(), "ale_cutoff_models", _MODEL_FILENAMES[model_name])
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Cutoff model '{model_name}' not found at '{model_path}'. "
            "Ensure the resource files are installed with NiMARE."
        )

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def predict_ale_cutoffs(
    nsub: Iterable[float],
    nfoci: Iterable[float],
) -> Dict[str, float]:
    """Predict ALE cutoff values using pre-trained XGBoost models.

    Parameters
    ----------
    nsub : iterable of float
        Subject counts per experiment.
    nfoci : iterable of float
        Foci counts per experiment.

    Returns
    -------
    dict
        Dictionary with keys ``vfwe``, ``cfwe``, and ``tfce`` containing the predicted thresholds.
    """
    nexp, nsub_arr, nfoci_arr = _validate_inputs(nsub, nfoci)
    features = _extract_features(nexp, nsub_arr, nfoci_arr)

    vfwe_cutoff = float(_load_model("vfwe").predict(features)[0])
    cfwe_cutoff = float(np.round(_load_model("cfwe").predict(features)[0], 0))
    tfce_cutoff = float(_load_model("tfce").predict(features)[0])

    return {
        "vfwe": vfwe_cutoff,
        "cfwe": cfwe_cutoff,
        "tfce": tfce_cutoff,
    }


__all__ = ["predict_ale_cutoffs"]
