"""The whitelist unpickler for upstream's pickles (``design.md``, Safety).

Unpickling calls whatever global a pickle names, so a pickle from a third
party can run code. ``load_pickle`` resolves exactly the 11 globals that a
static scan of all 2,068 upstream pickles found (2026-11-24): numpy arrays
and pandas DataFrames. It refuses any other global by name before anything
can call it.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

_GLOBALS = (
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
)
# Each whitelisted (module, name), mapped to the object that pickle's own find_class would return for it.
ALLOWED: dict[tuple[str, str], object] = {
    (module, name): getattr(importlib.import_module(module), name) for module, name in _GLOBALS
}


class _WhitelistUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        try:
            return ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"refused the global {module}.{name}: not one of the {len(ALLOWED)} globals of design.md (Safety)"
            ) from None


def load_pickle(path: Path) -> object:
    """The object pickled in ``path``; ``pickle.UnpicklingError`` naming any global outside ``ALLOWED``."""
    with Path(path).open("rb") as file:
        return _WhitelistUnpickler(file).load()
