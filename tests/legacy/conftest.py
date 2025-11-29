import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skip(reason="Legacy path – removed from active pipelines")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
