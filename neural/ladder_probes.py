"""Probes at the k-operand ladder's death point (REVIEW.md item 3).

The ladder (review_fixes.py cell 2) measured: top-byte of k-operand sum
mod 2^32 is learnable at k=2 (0.9996) and k=3 (0.9983) and dead at k=4
(max 0.041). Why exactly there? Three probes:

  curriculum  — train at k=3 to convergence, then fine-tune the SAME
                parameters on k=4. If scaffolding moves the frontier,
                carry composition is learnable with the right curriculum
                and the paper's interpretation needs a caveat.
  fourier16   — k=4 with a wider Fourier bank (16 harmonics per word
                instead of 8, m=0..15, reaching bit-level resolution of
                the top half-word).
  width       — k=4 at operand widths 8/16/24/32 bits (top byte of the
                W-bit sum). Death point as a function of carry-chain
                length at fixed operand count: if k=4 is learnable at
                narrow widths, the failure is carry-chain depth x count,
                not the count itself.

Baselines k=3 and k=4 (standard features) are re-run in the same process
for comparability. Architecture/training identical to the original ladder:
MLP width 1024, 3 blocks, 30 epochs, batch 8192, adam 1e-3.

Run:  python ladder_probes.py        (GPU, on-demand allocation)
Results stream to ladder_probes_results.json.
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax
import jax.numpy as jnp
import optax

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gohr_sweep import selection_adv, word_features  # noqa: E402
from review_fixes import init_mlp, fwd_mlp, expected_byte  # noqa: E402

RESULTS_PATH = 'ladder_probes_results.json'
N_TRAIN, N_TEST = 1 << 20, 1 << 18
WIDTH, BLOCKS, EPOCHS, BATCH = 1024, 3, 30, 8192


def save(results):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)


# ----------------------------------------------------------------- data
def ladder_data(rng, k, width_bits=32, n_harm=8):
    """k uniform width_bits-bit operands; target = top byte of their sum
    mod 2^width_bits. Features mirror gohr_sweep.word_features but are
    width- and harmonic-count-generic."""
    mod = 1 << width_bits
    ops = rng.integers(0, mod, size=(N_TRAIN + N_TEST, k),
                       dtype=np.uint64)
    y = (((ops.sum(1) % mod) >> (width_bits - 8)) & 0xFF).astype(np.uint8)

    if width_bits == 32 and n_harm == 8:
        X = word_features(ops.astype(np.uint32))
    else:
        N = ops.shape[0]
        v = ops.astype(np.float64) / mod                    # (N,k) in [0,1)
        bits = ((ops[:, :, None] >> np.arange(width_bits - 1, -1, -1,
                                              dtype=np.uint64))
                & 1).astype(np.float32)                     # MSB-first
        val = v.astype(np.float32)[:, :, None]
        top = ((ops >> np.uint64(width_bits - 8)).astype(np.float32)
               / 256.0)[:, :, None]
        harm = []
        for m in range(n_harm):
            ang = 2.0 * np.pi * np.mod(v * (1 << m), 1.0)
            harm.append(np.sin(ang).astype(np.float32)[:, :, None])
            harm.append(np.cos(ang).astype(np.float32)[:, :, None])
        X = np.concatenate([bits, val, top] + harm, axis=2).reshape(N, -1)
    return X, y


# -------------------------------------------------------------- trainer
def train(Xtr, ytr, Xte, yte, *, seed, lr=1e-3, epochs=EPOCHS,
          init_params=None):
    """The review_fixes ladder trainer, with optional warm-start params
    (for the curriculum probe). Returns rec + final params."""
    n_tr = Xtr.shape[0]
    spe = n_tr // batch_size()
    total = spe * epochs
    sched = optax.piecewise_constant_schedule(lr, {int(0.8 * total): 0.1})
    opt = optax.adam(sched)
    params = (init_params if init_params is not None else
              init_mlp(jax.random.PRNGKey(seed), Xtr.shape[1], WIDTH, BLOCKS))
    opt_state = opt.init(params)
    ytr_i = ytr.astype(np.int32)

    @jax.jit
    def step(params, opt_state, xb, yb):
        def loss_fn(p):
            return optax.softmax_cross_entropy_with_integer_labels(
                fwd_mlp(p, xb), yb).mean()
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
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_tr)
        for s_ in range(spe):
            sl = perm[s_ * batch_size():(s_ + 1) * batch_size()]
            params, opt_state, loss = step(
                params, opt_state, jnp.asarray(Xtr[sl]),
                jnp.asarray(ytr_i[sl]))
        traj[ep] = selection_adv(preds(params), yte)
    advs = list(traj.values())
    return dict(max_adv=max(advs), final_adv=advs[-1], traj=traj), params


def batch_size():
    return BATCH


def run_cell(tag, X, y, results, *, seed=42, lr=1e-3, epochs=EPOCHS,
             init_params=None):
    t0 = time.time()
    rec, params = train(X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:],
                        seed=seed, lr=lr, epochs=epochs,
                        init_params=init_params)
    print(f"[{tag}] max_adv={rec['max_adv']:+.4f} "
          f"final={rec['final_adv']:+.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    results[tag] = rec
    save(results)
    return params


# ------------------------------------------------------------------ main
def main():
    results = {}
    rng = np.random.default_rng(5)

    # Baselines, same process for comparability
    print("== baselines: k=3, k=4 standard features ==", flush=True)
    X3, y3 = ladder_data(rng, 3)
    p3 = run_cell('base_k3', X3, y3, results)
    X4, y4 = ladder_data(rng, 4)
    run_cell('base_k4', X4, y4, results)

    # Probe 1: curriculum k=3 -> k=4. Feature dims differ (3 vs 4 words),
    # so warm-start carries the trunk: re-init the input layer only.
    print("== probe 1: curriculum k3 -> k4 ==", flush=True)
    p4 = init_mlp(jax.random.PRNGKey(7), X4.shape[1], WIDTH, BLOCKS)
    warm = {'in': p4['in'], 'out': p3['out'], 'blocks': p3['blocks']}
    p_cur4 = run_cell('curriculum_k3_to_k4', X4, y4, results,
                      init_params=warm, lr=3e-4)
    del X3, y3, p3, p4, warm

    # Probe 2: wider Fourier bank at k=4 (16 harmonics, m=0..15)
    print("== probe 2: k=4 with 16 Fourier harmonics ==", flush=True)
    Xf, yf = ladder_data(rng, 4, n_harm=16)
    run_cell('fourier16_k4', Xf, yf, results)
    del Xf, yf, X4, y4

    # Probe 3: operand width ladder at k=4 (8/16/24-bit; 32 = base_k4)
    for wb in (8, 16, 24):
        print(f"== probe 3: k=4 at width {wb} bits ==", flush=True)
        Xw, yw = ladder_data(rng, 4, width_bits=wb)
        run_cell(f'width{wb}_k4', Xw, yw, results)
        del Xw, yw

    # Bonus rungs: if a probe moved the frontier, where does it die now?
    if results['curriculum_k3_to_k4']['max_adv'] > 0.5:
        print("== frontier moved by curriculum: k4 -> k5 warm-start ==",
              flush=True)
        X5, y5 = ladder_data(rng, 5)
        p5 = init_mlp(jax.random.PRNGKey(8), X5.shape[1], WIDTH, BLOCKS)
        warm5 = {'in': p5['in'], 'out': p_cur4['out'],
                 'blocks': p_cur4['blocks']}
        run_cell('curriculum_k4_to_k5', X5, y5, results,
                 init_params=warm5, lr=3e-4)
        del X5, y5, p5, warm5
    if results['fourier16_k4']['max_adv'] > 0.5:
        print("== frontier moved by fourier16: probing k=5 ==", flush=True)
        X5, y5 = ladder_data(rng, 5, n_harm=16)
        run_cell('fourier16_k5', X5, y5, results)

    print("done.", flush=True)


if __name__ == '__main__':
    main()
