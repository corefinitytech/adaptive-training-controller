"""Single source of truth for the installed package version.

Mirrors `pyproject.toml`'s `[project].version` — kept as a plain string (not read back
from package metadata at import time) so the version is available even in an editable
install before the package is built.
"""

__version__ = "0.1.0.dev0"
