"""Robustness check on the non-monotonic width result (ladder_probes.py).

First pass at k=4 gave: width 8 dead (0.036), 16 dead (0.075), 24 ALIVE
(0.993), 32 dead (0.030). Non-monotonicity in width is surprising enough
to verify before recording: sweep widths {8, 12, 16, 20, 24, 28, 32} at
k=4 with 2 fresh seeds each.

Run:  python ladder_width_check.py  (results to ladder_width_check.json)
"""

from __future__ import annotations

import json
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_probes as lp  # noqa: E402


def main():
    results = {}
    for wb in (8, 12, 16, 20, 24, 28, 32):
        for seed in (43, 44):
            rng = np.random.default_rng(1000 + seed)
            X, y = lp.ladder_data(rng, 4, width_bits=wb)
            tag = f'width{wb}_k4_s{seed}'
            print(f"== {tag} ==", flush=True)
            lp.run_cell(tag, X, y, results, seed=seed)
            del X, y
            with open('ladder_width_check.json', 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
