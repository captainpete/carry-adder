"""Float64 probe at k=4 / width=32 — the discriminating test of the
analog-shortcut hypothesis (REVIEW / RESULTS.md open thread).

The k=4 ladder dies exactly at full 32-bit width: k=2,3 pass at width 32,
k=4 passes below width 32, but at (k>=4 AND width=32) no seed (0/8) ever
leaves the ~0.03 floor. Two non-exclusive hypotheses were recorded:

  (a) optimisation basin shrinks with k and width and vanishes at width 32;
  (b) FLOAT32 ARTEFACT: the value/Fourier features resolve an operand to
      only ~2^-24 (the 24-bit f32 mantissa). The analog shortcut (estimate
      the real-valued sum, read off the top byte) loses the carry-relevant
      low bits exactly when width-8 exceeds 24 — i.e. at width 32 — forcing
      the model onto pure bit-level carry circuitry it cannot learn.

This script runs the DISCRIMINATING probe: rebuild the SAME k=4/width=32
task in full float64 (features AND network arithmetic) and compare,
seed-for-seed, against a float32 control built from the IDENTICAL integer
operands and IDENTICAL parameter-init seed. The only thing that differs
between the paired arms is numeric precision.

Prediction of hypothesis (b): float64 flips a meaningful fraction of seeds
from the ~0.03 floor to ~1.0 (the analog shortcut regains ~2^-53
resolution, enough to resolve all 32 bits). Prediction of (a) / null: both
arms stay at the floor; precision is irrelevant, the wall is real.

Architecture/training identical to the ladder (review_fixes / ladder_probes):
MLP width 1024, 3 residual blocks, 30 epochs, batch 8192, adam 1e-3 with a
0.1x step at 80%.

Run:  ../../bs2/.venv/bin/python float64_probe.py
Results stream to float64_probe_results.json.
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
jax.config.update('jax_enable_x64', True)  # MUST precede any array work

import numpy as np
import jax.numpy as jnp
import optax

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gohr_sweep import selection_adv          # noqa: E402  (numpy, dtype-agnostic)
from review_fixes import init_mlp, fwd_mlp, expected_byte  # noqa: E402

RESULTS_PATH = 'float64_probe_results.json'
# FP64 on this 3090 is ~51x slower than FP32 (641 vs 12.5 ms/step, measured),
# so the full-fidelity 2^20/8-seed run would take ~5.5 h. We hold full FP64
# (required: any FP32 cast after the features reintroduces the 2^-24 wall the
# hypothesis is about) but shrink training data 2^20->2^18 and seeds 8->6.
# Arch, epochs, and the paired FP32 control are unchanged; the FP64-vs-FP32
# comparison is internally paired (same operands/seed/N), which is what the
# precision claim rests on.
N_TRAIN, N_TEST = 1 << 18, 1 << 17
WIDTH, BLOCKS, EPOCHS, BATCH = 1024, 3, 30, 8192
K, WIDTH_BITS, N_HARM = 4, 32, 8
SEEDS = (43, 44, 45, 46, 47, 48)   # 6 paired seeds (baseline at full size was 0/8)


def save(results):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)


def make_ops(seed):
    """Identical integer operands for both arms at a given seed."""
    rng = np.random.default_rng(1000 + seed)
    mod = 1 << WIDTH_BITS
    ops = rng.integers(0, mod, size=(N_TRAIN + N_TEST, K), dtype=np.uint64)
    y = (((ops.sum(1) % mod) >> (WIDTH_BITS - 8)) & 0xFF).astype(np.uint8)
    return ops, y


def features(ops, dtype):
    """word_features, generic over float dtype. For dtype=float64 the
    value/harmonic features carry full 2^-53 operand resolution; for
    float32 they are truncated to ~2^-24 (the hypothesised artefact)."""
    N = ops.shape[0]
    bits = ((ops[:, :, None] >> np.arange(WIDTH_BITS - 1, -1, -1, dtype=np.uint64))
            & 1).astype(dtype)                                   # (N,k,32) MSB-first, exact
    v = ops.astype(np.float64) / float(1 << WIDTH_BITS)          # (N,k) in [0,1), f64
    val = v.astype(dtype)[:, :, None]
    top = ((ops >> np.uint64(WIDTH_BITS - 8)).astype(np.float64) / 256.0).astype(dtype)[:, :, None]
    harm = []
    for m in range(N_HARM):
        ang = 2.0 * np.pi * np.mod(v * (1 << m), 1.0)            # computed in f64
        harm.append(np.sin(ang).astype(dtype)[:, :, None])
        harm.append(np.cos(ang).astype(dtype)[:, :, None])
    return np.concatenate([bits, val, top] + harm, axis=2).reshape(N, -1)


def cast_tree(p, dtype):
    return jax.tree_util.tree_map(lambda a: a.astype(dtype), p)


def train(Xtr, ytr, Xte, yte, *, seed, dtype):
    spe = Xtr.shape[0] // BATCH
    total = spe * EPOCHS
    sched = optax.piecewise_constant_schedule(1e-3, {int(0.8 * total): 0.1})
    opt = optax.adam(sched)
    params = cast_tree(init_mlp(jax.random.PRNGKey(seed), Xtr.shape[1], WIDTH, BLOCKS), dtype)
    opt_state = opt.init(params)
    ytr_i = ytr.astype(np.int32)

    @jax.jit
    def step(params, opt_state, xb, yb):
        def loss_fn(p):
            return optax.softmax_cross_entropy_with_integer_labels(fwd_mlp(p, xb), yb).mean()
        loss, g = jax.value_and_grad(loss_fn)(params)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, loss

    @jax.jit
    def pred_chunk(params, xc):
        return expected_byte(fwd_mlp(params, xc))

    def preds(params):
        cs = 1 << 15
        return np.concatenate(
            [np.asarray(pred_chunk(params, jnp.asarray(Xte[i:i + cs])))
             for i in range(0, Xte.shape[0], cs)])

    rng = np.random.default_rng(seed)
    traj = {}
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(Xtr.shape[0])
        for s_ in range(spe):
            sl = perm[s_ * BATCH:(s_ + 1) * BATCH]
            params, opt_state, _ = step(params, opt_state,
                                        jnp.asarray(Xtr[sl]), jnp.asarray(ytr_i[sl]))
        traj[ep] = selection_adv(preds(params), yte)
    advs = list(traj.values())
    return dict(max_adv=max(advs), final_adv=advs[-1], traj=traj)


def run_arm(ops, y, seed, dtype, label, results):
    X = features(ops, dtype)
    t0 = time.time()
    rec = train(X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:], seed=seed, dtype=dtype)
    tag = f'{label}_s{seed}'
    print(f"[{tag}] max_adv={rec['max_adv']:+.4f} final={rec['final_adv']:+.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results[tag] = rec
    save(results)
    del X


def main():
    results = {}
    print(f"float64 probe: k={K} width={WIDTH_BITS}, x64={jax.config.read('jax_enable_x64')}, "
          f"{len(SEEDS)} seeds/arm", flush=True)
    for seed in SEEDS:
        ops, y = make_ops(seed)
        # paired: same operands, same init seed; only dtype differs
        run_arm(ops, y, seed, jnp.float64, 'f64', results)
        run_arm(ops, y, seed, jnp.float32, 'f32', results)
        del ops, y

    def summarise(label):
        live = [s for s in SEEDS if results[f'{label}_s{s}']['max_adv'] > 0.5]
        mx = [results[f'{label}_s{s}']['max_adv'] for s in SEEDS]
        return live, mx

    f64_live, f64_mx = summarise('f64')
    f32_live, f32_mx = summarise('f32')
    print("\n==== SUMMARY (k=4, width=32) ====", flush=True)
    print(f"  float32 control:  {len(f32_live)}/{len(SEEDS)} seeds alive "
          f"(max_adv: {', '.join(f'{m:.3f}' for m in f32_mx)})", flush=True)
    print(f"  float64 treat.:   {len(f64_live)}/{len(SEEDS)} seeds alive "
          f"(max_adv: {', '.join(f'{m:.3f}' for m in f64_mx)})", flush=True)
    verdict = ("ANALOG-SHORTCUT SUPPORTED: float64 revives the task"
               if len(f64_live) > len(f32_live)
               else "ANALOG-SHORTCUT DISFAVOURED: float64 does not move the wall")
    print(f"  verdict: {verdict}", flush=True)
    results['_summary'] = dict(f64_live=len(f64_live), f32_live=len(f32_live),
                               f64_max=f64_mx, f32_max=f32_mx, verdict=verdict)
    save(results)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
