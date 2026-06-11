"""Compatibility shim for the PI0 policy.

This module intentionally stays tiny.

The stable import path `rho.policies.pi0.modeling_pi0` is used in other parts of the
codebase (e.g. registration in `rho.policies`). The implementation lives in the
split modules:
- `configuration_pi0.py`
- `processing_pi0.py`
- `pi0_models.py`
- `pi0_policy.py`
"""

from .configuration_pi0 import PI0Config
from .pi0_policy import PI0Policy

__all__ = [
    "PI0Config",
    "PI0Policy",
]
