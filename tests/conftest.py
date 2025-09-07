import sys
import types
from pathlib import Path
import types

# Ensure the repository root is on sys.path so `nlhe` can be imported when the
# project is not installed as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


