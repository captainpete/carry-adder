"""Feature-isolation probe: is the low-byte death (readout_position_probe)
a distractor/representation artefact, or something intrinsic?

readout_position_probe found that at k=4/width=32 EVERY read-out byte is
dead (0/16), including the LOW byte — even though the low byte of a sum of
four 32-bit operands is exactly an 8-bit modular addition of their low
bytes, the same function the width-8 ladder cell learned (1/2 seeds).

The only thing that differs between "width-8, learnable" and "low byte of
width-32, dead" is the FEATURE REPRESENTATION: at width 32 the 8 relevant
low bits are buried among 24 distractor high bits, and the analog
value/Fourier features are dominated by the (irrelevant) high bits.

This probe fixes the task (predict the LOW byte b=0 of the k=4/width-32
sum) and varies ONLY the feature set:

  full   : word_features (32 bits + value + topbyte + 16 Fourier harmonics
           per operand) — the readout_position_probe condition. Expect DEAD.
  bits32 : the 32 raw bits per operand, no analog features. Tests whether
           the misleading analog features alone are the problem.
  low8   : only the 8 low bits per operand — the relevant inputs, nothing
           else. This is the width-8 task by another name. Expect ALIVE.

Reading: low8 alive + full dead  => the low-byte death is feature
isolation (SGD cannot extract the easy sub-circuit from distractors), not
an intrinsic limit. bits32's position locates whether the killer is the
analog features (bits32 revives) or the 24 distractor bits (bits32 stays
dead).

All FP32. Architecture/training identical to the ladder.
Run:  ../../bs2/.venv/bin/python feature_isolation_probe.py
Results stream to feature_isolation_probe_results.json.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_probes as lp           # noqa: E402
from gohr_sweep import word_features  # noqa: E402

RESULTS_PATH = 'feature_isolation_probe_results.json'
lp.RESULTS_PATH = RESULTS_PATH  # redirect lp.run_cell's internal save() here, not over the ladder file
K, WIDTH_BITS, READOUT = 4, 32, 0    # predict the LOW byte
SEEDS = (43, 44, 45, 46)
CONDS = ('full', 'bits32', 'low8')


def make(rng, cond):
    mod = 1 << WIDTH_BITS
    ops = rng.integers(0, mod, size=(lp.N_TRAIN + lp.N_TEST, K), dtype=np.uint64)
    y = ((ops.sum(1) % mod) >> np.uint64(READOUT) & 0xFF).astype(np.uint8)
    N = ops.shape[0]
    if cond == 'full':
        X = word_features(ops.astype(np.uint32))
    elif cond == 'bits32':
        X = ((ops[:, :, None] >> np.arange(31, -1, -1, dtype=np.uint64)) & 1
             ).astype(np.float32).reshape(N, -1)            # 32 bits/op, MSB-first
    elif cond == 'low8':
        X = ((ops[:, :, None] >> np.arange(7, -1, -1, dtype=np.uint64)) & 1
             ).astype(np.float32).reshape(N, -1)            # low 8 bits/op
    else:
        raise ValueError(cond)
    return X, y


def main():
    results = {}
    print(f"feature-isolation probe: k={K} width={WIDTH_BITS} readout=byte@{READOUT} (low byte), "
          f"conds={CONDS}, {len(SEEDS)} seeds each", flush=True)
    for cond in CONDS:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)  # same operands as readout probe
            X, y = make(rng, cond)
            tag = f'{cond}_s{seed}'
            print(f"== {tag} (in_dim={X.shape[1]}) ==", flush=True)
            lp.run_cell(tag, X, y, results, seed=seed)
            with open(RESULTS_PATH, 'w') as fh:
                json.dump(results, fh, indent=2, default=float)
            del X, y

    print("\n==== SUMMARY (predict low byte of k=4/width=32 sum) ====", flush=True)
    summ = {}
    for cond in CONDS:
        mx = [results[f'{cond}_s{s}']['max_adv'] for s in SEEDS]
        alive = sum(m > 0.5 for m in mx)
        summ[cond] = dict(max_adv=mx, alive=alive, best=max(mx))
        print(f"  {cond:>7}: {alive}/{len(SEEDS)} alive, best={max(mx):.3f}  "
              f"[{', '.join(f'{m:.3f}' for m in mx)}]", flush=True)
    results['_summary'] = summ
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
