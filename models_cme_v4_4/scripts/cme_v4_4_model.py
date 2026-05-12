"""CME-v4.4 model wrapper.

v4.4 intentionally reuses the proven CME-v4.2 neural architecture unchanged.
Its experimental variable is *input signal*: leakage-safe rolling
possession/process-style features are appended to the tabular pregame context.
This isolates whether better pregame style data helps without confounding the
comparison with new heads or auxiliary targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
sys.path.insert(0, str(V42_SCRIPTS))

from cme_v4_2_model import CmeV42, CmeV42Config, total_loss  # noqa: E402


class CmeV44(CmeV42):
    """Alias class for run metadata and checkpoint readability."""


CmeV44Config = CmeV42Config

__all__ = ["CmeV44", "CmeV44Config", "total_loss"]
