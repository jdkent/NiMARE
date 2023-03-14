"""

.. _cbma_workflow:

====================================================
Run a coordinate-based meta-analysis (CBMA) workflow
====================================================

NiMARE provides a plethora of tools for performing meta-analyses on neuroimaging data.
Sometimes it's difficult to know where to start, especially if you're new to meta-analysis.
This tutorial will walk you through using a CBMA workflow function which puts together
the fundamental steps of a CBMA meta-analysis.
"""

import os

from nimare.dataset import Dataset
from nimare.utils import get_resource_path
from nimare.workflows import cbma_workflow


###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

###############################################################################
# Run CBMA Workflow
# -----------------------------------------------------------------------------
# The CBMA workflow function runs the following steps:
# 1. Runs a meta-analysis using the specified method (default: ALE)
# 2. Applies a corrector to the meta-analysis results (default: FWECorrector)
# 3. Generates cluster tables and runs diagnostics on the corrected results (default: Jackknife)
#
# All in one function call!

result = cbma_workflow(dset)


###############################################################################
# Plot Results
# -----------------------------------------------------------------------------
# The CBMA workflow function returns a dictionary of results.


