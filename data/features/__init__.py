"""Feature engineering per §3.5.

The single rule, restated from §3.5:

    "The anti-leakage rule is absolute: no future value, including any
     feature derived from a future value, may appear in a training input."

Every transform in this package is causal — it uses only values at times
≤ t to compute the feature for time t. The leakage validator in
`data.leakage_check` empirically verifies this for any feature pipeline.
"""

from .engineer import engineer_features
from .upstream import add_upstream_request_rate

__all__ = ["engineer_features", "add_upstream_request_rate"]
