# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Compatibility shim for the owner gate's former home.

The rules moved to :mod:`names` when repository validation joined them,
because the structural half is shared and only the grammars differ.
``require_owner`` and ``require_owner_from_path`` are public, and this
module was an importable path for them, so a split that removed it
would be a breaking change rather than a refactor.

New code should import from :mod:`names`.
"""

from __future__ import annotations

from .names import require_owner as require_owner
from .names import require_owner_from_path as require_owner_from_path

__all__ = ["require_owner", "require_owner_from_path"]
