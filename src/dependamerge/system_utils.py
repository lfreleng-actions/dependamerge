# SPDX-FileCopyrightText: 2024 Linux Foundation
# SPDX-License-Identifier: Apache-2.0

"""System utilities for dependamerge and related tools.

This module provides system-level utilities that can be shared across
dependamerge, markdown-table-fixer, and pull-request-fixer.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _detect_macos_perf_cores() -> int | None:
    """Return macOS performance-core count via sysctl, or None if unavailable."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        perf_cores = int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        logger.debug(f"Could not detect macOS performance cores: {e}, using fallback")
        return None
    if perf_cores > 0:
        logger.debug(f"Detected {perf_cores} performance cores on macOS")
        return perf_cores
    return None


def _read_linux_core_identity(cpu_dir: str) -> tuple[int, int] | None:
    """Return the (package_id, core_id) topology identity for a cpu dir.

    ``core_id`` is only unique within a physical package/socket, so the
    physical-core identity is the ``(physical_package_id, core_id)`` pair;
    keying on the pair avoids undercounting on multi-socket hosts where
    each socket reuses the same ``core_id`` range. A missing or unreadable
    ``physical_package_id`` falls back to package 0 (the common
    single-socket case).

    Reads are best-effort per CPU: an unreadable or malformed ``core_id``
    file yields None so a single bad entry does not abort detection for the
    remaining CPUs. Opening directly (rather than checking existence first)
    also avoids a TOCTOU window.
    """
    if not (cpu_dir.startswith("cpu") and cpu_dir[3:].isdigit()):
        return None
    topology = f"/sys/devices/system/cpu/{cpu_dir}/topology"
    try:
        with open(f"{topology}/core_id") as f:
            core_id = int(f.read().strip())
    except (OSError, ValueError):
        return None
    try:
        with open(f"{topology}/physical_package_id") as f:
            package_id = int(f.read().strip())
    except (OSError, ValueError):
        package_id = 0
    return (package_id, core_id)


def _detect_linux_physical_cores() -> int | None:
    """Return Linux physical-core count (floor of 2) from sysfs, or None.

    The detected count is clamped to a minimum of 2 so single-core or
    partially-readable systems still report a usable parallelism level;
    callers never receive 1. ``None`` means detection failed entirely
    (no readable sysfs topology).
    """
    try:
        # Physical cores approximate performance cores (excludes SMT siblings).
        core_identities: set[tuple[int, int]] = set()
        for cpu_dir in os.listdir("/sys/devices/system/cpu/"):
            identity = _read_linux_core_identity(cpu_dir)
            if identity is not None:
                core_identities.add(identity)
    except (OSError, ValueError) as e:
        logger.debug(f"Could not detect Linux physical cores: {e}, using fallback")
        return None
    if core_identities:
        phys_cores = len(core_identities)
        logger.debug(f"Detected {phys_cores} physical cores on Linux")
        return max(2, phys_cores)
    return None


def get_performance_core_count() -> int:
    """Get the number of performance cores available on the system.

    This function attempts to detect the actual number of performance cores
    (P-cores) rather than the total logical CPU count which includes:
    - Efficiency cores (E-cores) on hybrid architectures
    - Hyperthreading/SMT virtual cores

    Detection methods by platform:
    - macOS: Uses sysctl to query hw.perflevel0.physicalcpu
    - Linux: Parses /sys/devices/system/cpu/ topology for physical cores
    - Windows: Future enhancement could use WMI queries
    - Fallback: Uses half of total CPU count (assumes hyperthreading)

    Returns:
        int: Number of performance cores, minimum of 2

    Examples:
        >>> cores = get_performance_core_count()
        >>> # On a M1 Max with 10 cores: returns 8 (performance cores)
        >>> # On a 16-thread Intel CPU: returns 8 (physical cores)
    """
    cpu_count = os.cpu_count() or 4

    if sys.platform == "darwin":
        perf_cores = _detect_macos_perf_cores()
        if perf_cores is not None:
            # Clamp to the documented floor of 2 (the sysctl path can
            # report 1); the Linux and fallback paths already clamp.
            return max(2, perf_cores)

    if sys.platform.startswith("linux"):
        phys_cores = _detect_linux_physical_cores()
        if phys_cores is not None:
            return phys_cores

    # Windows: Future enhancement
    # Could use: wmic cpu get NumberOfCores
    # Or: Get-CimInstance Win32_Processor | Select-Object NumberOfCores

    # Fallback: Use half of total CPU count
    # This assumes hyperthreading (2 threads per core) which is common
    # on modern CPUs. For CPUs without hyperthreading, this will
    # underestimate, but it's a safe conservative default.
    fallback_cores = max(2, cpu_count // 2)
    logger.debug(f"Using fallback: {fallback_cores} cores (total CPUs: {cpu_count})")
    return fallback_cores


def get_default_workers() -> int:
    """Get default worker count based on CPU performance cores.

    This is the recommended function to use for determining the default
    number of parallel workers for I/O-bound tasks like GitHub API calls.

    For I/O-bound workloads (which is what these tools primarily do),
    the performance core count is a good default as it:
    - Maximizes parallelism without oversubscribing the CPU
    - Avoids excessive context switching
    - Works well with async I/O operations

    Returns:
        int: Recommended default number of workers

    Examples:
        >>> workers = get_default_workers()
        >>> # Use in CLI: default=get_default_workers()
    """
    return get_performance_core_count()
