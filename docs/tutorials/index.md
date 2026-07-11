# Tutorials

These notebooks are maintained as deterministic, self-contained tutorials. They
use only public Python APIs, small generated arrays, and no network access or
credentials.

- [First reconstruction](notebooks/quickstart.ipynb) compiles a model, simulates
  an object, runs alternating projection, and plots the result.
- [Synthetic objects](notebooks/synthetic_objects_quickstart.ipynb) previews the
  in-memory object constructors and reconstructs one test target.
- [Reconstruction diagnostics](notebooks/diagnostics_quickstart.ipynb) records
  convergence and Fourier coverage and renders overview plots.

Notebook execution is deliberately disabled during `mkdocs serve` and
`docs-build`, keeping documentation builds fast and deterministic. The source
notebooks have no stored large outputs. Their code cells are executed as tests
by `python/tests/test_diagnostic_notebooks.py` in the Python test suite.

Two more focused examples, `python/examples/diagnostics_debug.ipynb` and
`python/examples/diagnostics_plots.ipynb`, remain in the repository but are not
published in the primary tutorial path. They are maintained test fixtures for
advanced per-frame inspection and direct plot customization.
