"""Read-out position probe: is the controlling variable the number of
CARRY BITS BELOW THE OBSERVABLE, not width or k per se?

The width ladder (ladder_width_check.py) is, read correctly, a sweep of
"carry bits below the read-out": the target is always the TOP byte of the
sum, so at operand width W the observed byte sits atop a (W-8)-bit carry
chain. Width 8 -> 0 carry bits below; width 32 -> 24. The death at width
32 is death when ~24 bits of carry separate inputs from observable.

This probe tests that reading directly. Fix k=4 and full width=32 (so the
INPUT distribution and the features are identical across all cells), and
vary only WHICH byte of the sum we predict:

  readout bit b in {0, 8, 16, 24}; target = (sum mod 2^32 >> b) & 0xFF.

  b=0   low byte:  ~0 carry bits below the observable -> should be LEARNABLE
  b=24  top byte:  24 carry bits below (== base_k4)   -> should be DEAD
  b=8,16 intermediate carry depth.

Prediction (carry-depth-below-observable / SQ-degree story): a monotone
staircase, learnable at b=0 falling to the floor by b=24. A generic
"optimisation basin shrinks with width/k" story predicts no such
structure — all width-32 read-outs behave alike. Same operands, same
features, same net: the only variable is carry depth below the read-out.

All FP32 (fast). Architecture/training identical to the ladder
(MLP w1024/b3, 30 epochs, batch 8192, adam 1e-3), full 2^20 train so the
b=24 cells coincide with the documented base_k4 baseline.

Run:  ../../bs2/.venv/bin/python readout_position_probe.py
Results stream to readout_position_probe_results.json.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_probes as lp           # noqa: E402  (brings WIDTH/BLOCKS/EPOCHS, run_cell, word_features)
from gohr_sweep import word_features  # noqa: E402

RESULTS_PATH = 'readout_position_probe_results.json'
lp.RESULTS_PATH = RESULTS_PATH  # redirect lp.run_cell's internal save() here, not over the ladder file
K, WIDTH_BITS = 4, 32
READOUTS = (0, 8, 16, 24)
SEEDS = (43, 44, 45, 46)


def readout_data(rng, readout_bit):
    """k=4 full-width operands; target = the byte at bit-offset readout_bit
    of their sum mod 2^32. Features are word_features over the SAME 32-bit
    operands for every readout_bit (only the label changes)."""
    mod = 1 << WIDTH_BITS
    ops = rng.integers(0, mod, size=(lp.N_TRAIN + lp.N_TEST, K), dtype=np.uint64)
    s = ops.sum(1) % mod
    y = ((s >> np.uint64(readout_bit)) & 0xFF).astype(np.uint8)
    X = word_features(ops.astype(np.uint32))
    return X, y


def main():
    results = {}
    print(f"readout-position probe: k={K} width={WIDTH_BITS}, "
          f"readouts={READOUTS}, {len(SEEDS)} seeds each", flush=True)
    for b in READOUTS:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)  # operands depend on seed only
            X, y = readout_data(rng, b)
            tag = f'b{b:02d}_s{seed}'
            lp.run_cell(tag, X, y, results, seed=seed)
            with open(RESULTS_PATH, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
            del X, y

    # summary: mean/max advantage and #alive per readout position
    print("\n==== SUMMARY (k=4, width=32): advantage vs read-out byte ====", flush=True)
    print(f"  carry bits below observable -> learnability", flush=True)
    summ = {}
    for b in READOUTS:
        mx = [results[f'b{b:02d}_s{s}']['max_adv'] for s in SEEDS]
        alive = sum(m > 0.5 for m in mx)
        summ[f'b{b:02d}'] = dict(carry_bits_below=b, max_adv=mx,
                                 alive=alive, best=max(mx))
        print(f"  b={b:>2} ({b:>2} carry bits below): {alive}/{len(SEEDS)} alive, "
              f"best={max(mx):.3f}  [{', '.join(f'{m:.3f}' for m in mx)}]", flush=True)
    results['_summary'] = summ
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
