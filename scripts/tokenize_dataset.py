#!/usr/bin/env python
from __future__ import annotations

import importlib
import os
import sys


if __name__ == "__main__":
    importlib.import_module("2nzq.tokenize").main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Some datasets/pyarrow builds can crash during interpreter finalization after
    # streaming. The data is already saved by this point, so bypass finalizers.
    os._exit(0)
