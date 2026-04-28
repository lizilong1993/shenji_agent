"""pytest configuration: import real torch/numpy before any test can install fakes."""
import sys

# Import real numpy and torch at collection time so fake installers in other
# test modules (test_mainline_contract.py) hit the sys.modules guard and skip.
_HAS_REAL_NUMPY = False
_HAS_REAL_TORCH = False

try:
    import numpy as np  # noqa: F401
    _HAS_REAL_NUMPY = True
except ImportError:
    pass

try:
    import torch  # noqa: F401
    _HAS_REAL_TORCH = True
except ImportError:
    pass
