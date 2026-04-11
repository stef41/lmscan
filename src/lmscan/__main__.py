"""Allow running lmscan as ``python -m lmscan``."""
from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
