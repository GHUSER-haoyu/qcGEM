"""qcGEM-Student standalone molecular representation & calibrated QC package."""
from .api import QCGEMStudent, kabsch_align
from .head import (
    EDGE_NAMES,
    NODE_NAMES,
    STARRED_ELEMENT_NPA_CHARGE,
    STARRED_UNRELIABLE_CHANNELS,
)

__version__ = "0.1.0"
__all__ = [
    "QCGEMStudent",
    "kabsch_align",
    "NODE_NAMES",
    "EDGE_NAMES",
    "STARRED_UNRELIABLE_CHANNELS",
    "STARRED_ELEMENT_NPA_CHARGE",
    "__version__",
]
