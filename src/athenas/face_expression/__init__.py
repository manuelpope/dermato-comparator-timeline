"""Local facial-expression analysis toolkit.

This sub-package powers the aesthetics (Botox before/after) comparison of
the Athenas API. It is intentionally self-contained — the only reuse from
the rest of ``athenas`` is the lightweight ``athenas.io.read_rgb`` /
``athenas.quality.*`` helpers and matplotlib's headless ``PdfPages``
pattern already used in ``src/athenas/report.py``.

Public modules (importable directly as ``athenas.face_expression.<mod>``)
-------------------------------------------------------------------------

- :mod:`face_detection`   — MediaPipe Face Landmarker wrapper + pose estimation.
- :mod:`alignment`         — similarity transform + capture-protocol validation.
- :mod:`wrinkle_detection` — black-hat morphological wrinkle-line detection.
- :mod:`overlays`          — visual overlays (landmarks, lines, change map).

Local-only
----------
No cloud call — the photos never leave the machine.
"""

from __future__ import annotations

# Modules are imported on demand — each one is independently testable and
# the rest of the package does not need to be importable for a single module
# to work. This keeps the import surface small and avoids partial-init errors.
__all__ = [
    "alignment",
    "face_detection",
    "overlays",
    "wrinkle_detection",
]
