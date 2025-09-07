import sys
import types
from pathlib import Path

# Ensure the repository root is on sys.path so `nlhe` can be imported when the
# project is not installed as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a stub for the optional compiled evaluator so ``nlhe.core`` can be
# imported in environments without the native extension.
stub = types.SimpleNamespace(best5_rank_from_7_py=lambda cards: (0, ()))
sys.modules.setdefault("nlhe_engine", stub)

# Import the Python engine module once while the stub is in place so it becomes
# available to tests even after the stub is removed.
import nlhe.core.engine  # noqa: F401

# Remove the stub so that attempts to import the optional Rust backend fail,
# causing parity tests to be skipped when the backend is unavailable.
sys.modules.pop("nlhe_engine", None)
