"""Helpers for building safe HTTP response headers.

The PDF endpoints name their attachment after user-supplied values (the
patient id, the upload file names). Those end up inside the quoted
``filename=`` parameter of the ``Content-Disposition`` header, so any
character that could close the quoted string — a ``"``, a ``;``, a
newline — has to be replaced before it is interpolated.
"""

from __future__ import annotations

import re

#: Characters allowed in a ``Content-Disposition`` filename fragment.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename_part(value: str) -> str:
    """Reduce a user-supplied string to a safe filename fragment.

    The result is safe to interpolate into the quoted ``filename=``
    parameter of a ``Content-Disposition`` header. Empty or fully
    stripped values fall back to ``"x"`` so the header always carries a
    name.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", value).strip("._-")
    return cleaned[:64] or "x"
