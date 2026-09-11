"""Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""

import numpy as np
import pandas as pd
import pytest

from nimare.decode import discrete
from nimare.transforms import nlogp_to_z


def test_neurosynth_decode(testdata_laird):
    """Smoke test for discrete.neurosynth_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.neurosynth_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_brainmap_decode(testdata_laird):
    """Smoke test for discrete.brainmap_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.brainmap_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder."""
    ids = testdata_laird.ids[:5]
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.NeurosynthDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


def test_NeurosynthDecoder_featuregroup(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder with feature group selection."""
    ids = testdata_laird.ids[:5]
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF")
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder_featuregroup_failure(testdata_laird):
    """Smoke test for NeurosynthDecoder with feature group selection and no detected features."""
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF", features=["01", "05"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_BrainMapDecoder(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder."""
    ids = testdata_laird.ids[:5]
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.BrainMapDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


def test_BrainMapDecoder_failure(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder where there are no features left."""
    decoder = discrete.BrainMapDecoder(features=["doggy"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_ROIAssociationDecoder(testdata_laird, roi_img):
    """Smoke test for discrete.ROIAssociationDecoder."""
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.ROIAssociationDecoder(masker=roi_img, features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform()
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 1)


def test_brainmap_decode_forward_z_is_one_tailed_and_unsigned(testdata_laird):
    """Forward inference is ``binom.logsf``, an upper tail, so its z has no lower side.

    Regression test: the one-sided p-value was converted with a two-tailed rule and then
    multiplied by a sign taken from the mean label count, which could report a
    significant *depletion* that the test never tested for.
    """
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.brainmap_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )

    finite = decoded_df["zForward"].dropna()
    assert (finite >= 0).all()

    expected = nlogp_to_z(np.log(decoded_df["pForward"].values), "one")
    np.testing.assert_allclose(decoded_df["zForward"].values, expected, rtol=1e-6)


def _saturated_label_inputs():
    """One label whose every focus falls inside the selection, and one carried by nobody."""
    ids = [f"s{i:02d}" for i in range(40)]
    coordinates = pd.DataFrame({"id": ids, "x": 0.0, "y": 0.0, "z": 0.0, "space": "MNI"})
    annotations = pd.DataFrame(
        {
            "id": ids,
            "common": [1] * 30 + [0] * 10,
            "absent": [0] * 39 + [1],  # carried only by an unselected study
            "saturated": [1] * 8 + [0] * 32,  # every carrier is selected
        }
    )
    return coordinates, annotations, ids[:20]


def test_brainmap_decode_forward_p_is_the_inclusive_upper_tail():
    """The one-sided p-value for observing k is P(X >= k), i.e. ``logsf(k - 1)``.

    Regression test: ``logsf(k)`` is P(X > k), which excludes the observation. A label whose
    every focus falls inside the selection then gets P(X > n) == 0 and an infinite z.
    """
    coordinates, annotations, selected = _saturated_label_inputs()
    decoded_df = discrete.brainmap_decode(
        coordinates,
        annotations,
        ids=selected,
        features=["common", "absent", "saturated"],
        correction=None,
    )

    assert np.isfinite(decoded_df["zForward"]).all()
    # All 8 carriers of 'saturated' are selected, and p_selected is 0.5, so P(X >= 8) = 0.5 ** 8.
    assert decoded_df.loc["saturated", "pForward"] == pytest.approx(0.5**8)
    # A label no selected study carries is never evidence of enrichment.
    assert decoded_df.loc["absent", "pForward"] == pytest.approx(1.0)
    assert decoded_df.loc["absent", "zForward"] == pytest.approx(0.0)


def test_brainmap_decode_zero_count_is_never_enrichment():
    """``logsf(-1)`` is log(1), so an unobserved label needs no special case."""
    from scipy.stats import binom

    assert binom.logsf(k=-1, n=1, p=0.001) == 0.0
    assert binom.logsf(k=-1, n=50, p=0.3) == 0.0
