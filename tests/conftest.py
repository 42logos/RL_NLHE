import sys
from pathlib import Path
import types

# Ensure the repository root is on sys.path so `nlhe` can be imported when the
# project is not installed as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a minimal stub for the optional evaluation module so that the
# engine can be imported without the compiled extension.
import nlhe.core  # noqa: F401 - ensure package exists
fake_eval = types.ModuleType("nlhe.core.eval")

def best5_rank_from_7(cards7):
    return (0, ())

fake_eval.best5_rank_from_7 = best5_rank_from_7
sys.modules.setdefault("nlhe.core.eval", fake_eval)
