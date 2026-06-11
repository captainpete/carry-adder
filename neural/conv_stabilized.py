"""Stabilised rerun of the conv arm (REVIEW.md item 1).

The Gohr-faithful 1-D conv over bit positions in review_fixes.py collapsed
at epoch 14 of its j=1 run and its j=0 power control reached only 0.058
against a gate of 0.5, so its j=1 null was uninterpretable and the
conv-inductive-bias objection is formally OPEN in RESULTS.md.

This rerun changes exactly the three things the review prescribes:
  - lower peak LR with linear warmup (warmup_cosine_decay, peak 5e-4;
    automatic second attempt at 2e-4 if the control still fails)
  - normalisation: pre-norm LayerNorm over channels in every conv block
  - residual conv blocks with zero-initialised final conv (identity start)

Protocol (decided before running):
  1. j=0 power control first, gate 0.5 (same gate as the original arm).
  2. If the control passes at either LR: run j=1 (3 shuffle controls) and
     j=2 (1 shuffle control) at that LR; those cells then COUNT.
  3. If both attempts fail the control: the conv arm is recorded as twice
     attempted / still unpowered, upgrading "uninterpretable" in
     RESULTS.md to "twice attempted".

Run:  python conv_stabilized.py            (GPU, on-demand allocation)
Results stream to conv_stabilized_results.json.
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
from gohr_sweep import (  # noqa: E402
    make_dataset, selection_adv, FEATS_PER_WORD)
from review_fixes import dense, expected_byte, verdict  # noqa: E402

RESULTS_PATH = 'conv_stabilized_results.json'
CONV_CH, CONV_BLOCKS, CONV_K = 64, 5, 3
R_READ = 30
POWER_GATE = 0.5
PEAK_LRS = [5e-4, 2e-4]


def save(results):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)


# ---------------------------------------------------------------- model
def init_conv(key, n_words, aux_dim):
    ks = jax.random.split(key, 4 + 2 * CONV_BLOCKS)

    def conv_w(k, kw, ci, co, zero=False):
        w = (jnp.zeros((kw, ci, co)) if zero else
             jax.random.normal(k, (kw, ci, co)) * jnp.sqrt(2.0 / (kw * ci)))
        return {'w': w, 'b': jnp.zeros(co)}

    p = {'in': conv_w(ks[0], 1, n_words, CONV_CH), 'blocks': [],
         'd1': dense(ks[1], 32 * CONV_CH + aux_dim, 512),
         'out': dense(ks[2], 512, 256)}
    for b in range(CONV_BLOCKS):
        p['blocks'].append({
            'c1': conv_w(ks[3 + 2 * b], CONV_K, CONV_CH, CONV_CH),
            'c2': conv_w(ks[4 + 2 * b], CONV_K, CONV_CH, CONV_CH, zero=True),
            'ln_g': jnp.ones(CONV_CH), 'ln_b': jnp.zeros(CONV_CH)})
    return p


def _conv(x, w):
    return jax.lax.conv_general_dilated(
        x, w['w'], window_strides=(1,), padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')) + w['b']


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    sd = jnp.sqrt(x.var(-1, keepdims=True) + 1e-6)
    return (x - mu) / sd * g + b


def fwd_conv(p, xb_bits, xb_aux):
    # xb_bits: (N, 32, n_words) bits along positions, words as channels.
    # Pre-norm residual: h + c2(gelu(c1(LN(h)))), c2 zero-init.
    h = jax.nn.gelu(_conv(xb_bits, p['in']))
    for blk in p['blocks']:
        z = _ln(h, blk['ln_g'], blk['ln_b'])
        z = jax.nn.gelu(_conv(z, blk['c1']))
        z = _conv(z, blk['c2'])
        h = h + z
    flat = h.reshape(h.shape[0], -1)
    hd = jax.nn.gelu(jnp.concatenate([flat, xb_aux], 1)
                     @ p['d1']['w'] + p['d1']['b'])
    return hd @ p['out']['w'] + p['out']['b']


# --------------------------------------------------------------- trainer
def train_conv(bits_tr, aux_tr, ytr, bits_te, aux_te, yte, *, epochs, batch,
               seed, peak_lr):
    n_tr = bits_tr.shape[0]
    spe = n_tr // batch
    total = spe * epochs
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=max(1, total // 10), decay_steps=total)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    params = init_conv(jax.random.PRNGKey(seed),
                       bits_tr.shape[2], aux_tr.shape[1])
    opt_state = opt.init(params)
    ytr_i = ytr.astype(np.int32)

    @jax.jit
    def step(params, opt_state, xb, ab, yb):
        def loss_fn(p):
            return optax.softmax_cross_entropy_with_integer_labels(
                fwd_conv(p, xb, ab), yb).mean()
        loss, g = jax.value_and_grad(loss_fn)(params)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, loss

    @jax.jit
    def pred_chunk(params, xc, ac):
        return expected_byte(fwd_conv(params, xc, ac))

    def preds(params):
        cs = 1 << 14
        return np.concatenate(
            [np.asarray(pred_chunk(params, jnp.asarray(bits_te[i:i + cs]),
                                   jnp.asarray(aux_te[i:i + cs])))
             for i in range(0, bits_te.shape[0], cs)])

    rng = np.random.default_rng(seed)
    traj, losses, collapse = {}, {}, None
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_tr)
        for s_ in range(spe):
            sl = perm[s_ * batch:(s_ + 1) * batch]
            params, opt_state, loss = step(
                params, opt_state, jnp.asarray(bits_tr[sl]),
                jnp.asarray(aux_tr[sl]), jnp.asarray(ytr_i[sl]))
        lossf = float(loss)
        pred = preds(params)
        traj[ep] = selection_adv(pred, yte)
        losses[ep] = lossf
        if not np.isfinite(lossf) or float(np.std(pred)) < 1e-5:
            collapse = f'collapse at ep{ep}'
            break
    advs = list(traj.values())
    rec = dict(max_adv=max(advs), final_adv=advs[-1], traj=traj, loss=losses,
               collapse=collapse)
    del params, opt_state
    return rec


def conv_arrays(X, n_words):
    Xw = X.reshape(-1, n_words, FEATS_PER_WORD)
    bits = np.ascontiguousarray(Xw[:, :, :32].transpose(0, 2, 1))
    aux = np.ascontiguousarray(Xw[:, :, 32:].reshape(Xw.shape[0], -1))
    return bits, aux


# ------------------------------------------------------------------ main
def main():
    results = {}

    # ---- 1. j=0 power control, attempts at descending peak LR ----
    print("== conv_stabilized: j=0 power control ==", flush=True)
    n0, n0t = 1 << 18, 1 << 17
    sf0, wf0, ys0, _ = make_dataset(0, R_READ, 0, n0 + n0t)
    X0 = np.concatenate([sf0, wf0[:, :FEATS_PER_WORD]], axis=1)
    nw0 = X0.shape[1] // FEATS_PER_WORD
    bits0, aux0 = conv_arrays(X0, nw0)

    passed_lr = None
    for lr in PEAK_LRS:
        t0 = time.time()
        rc = train_conv(bits0[:n0], aux0[:n0], ys0[0][:n0], bits0[n0:],
                        aux0[n0:], ys0[0][n0:], epochs=25, batch=4096,
                        seed=42, peak_lr=lr)
        results[f'control_j0_lr{lr:g}'] = dict(
            max_adv=rc['max_adv'], final_adv=rc['final_adv'],
            collapse=rc['collapse'], traj=rc['traj'], loss=rc['loss'])
        ok = rc['max_adv'] >= POWER_GATE and not rc['collapse']
        print(f"[control_j0 lr={lr:g}] max_adv={rc['max_adv']:+.4f} "
              f"final={rc['final_adv']:+.4f} collapse={rc['collapse']} "
              f"gate {POWER_GATE} -> {'PASS' if ok else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save(results)
        if ok:
            passed_lr = lr
            break

    results['control_verdict'] = (
        f'PASS at peak_lr {passed_lr:g}' if passed_lr else
        'FAIL at both LRs: conv arm twice attempted, still unpowered')
    save(results)
    del sf0, wf0, ys0, X0, bits0, aux0
    if not passed_lr:
        print("control failed at both LRs -> stopping per protocol; "
              "conv objection upgrades to 'twice attempted'.", flush=True)
        return

    # ---- 2. j=1 (3 shuffles) and j=2 (1 shuffle), now interpretable ----
    n_train, n_test = 1 << 20, 1 << 19
    for j, n_sh in ((1, 3), (2, 1)):
        print(f"== conv_stabilized: j={j} ({n_sh} shuffles) ==", flush=True)
        sf, wf, ys, _ = make_dataset(0, R_READ, j, n_train + n_test)
        X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
        yall = ys[j]
        nw = X.shape[1] // FEATS_PER_WORD
        bits, aux = conv_arrays(X, nw)
        t0 = time.time()
        real = train_conv(bits[:n_train], aux[:n_train], yall[:n_train],
                          bits[n_train:], aux[n_train:], yall[n_train:],
                          epochs=40, batch=4096, seed=42, peak_lr=passed_lr)
        shufs = []
        for i in range(n_sh):
            rngs = np.random.default_rng(100 + i)
            shufs.append(train_conv(
                bits[:n_train], aux[:n_train],
                rngs.permutation(yall[:n_train]),
                bits[n_train:], aux[n_train:], yall[n_train:],
                epochs=40, batch=4096, seed=200 + i, peak_lr=passed_lr))
        sm = [s['max_adv'] for s in shufs]
        results[f'conv_j{j}'] = dict(
            real_max=real['max_adv'], real_final=real['final_adv'],
            shuffle_maxes=sm, collapse=real['collapse'],
            shuffle_collapses=[s['collapse'] for s in shufs],
            real_traj=real['traj'], real_loss=real['loss'],
            reach=verdict(real['max_adv'], sm))
        flag = 'COLLAPSED' if real['collapse'] else (
            'REACH?' if results[f'conv_j{j}']['reach'] else 'null')
        print(f"[conv_j{j}] real_max={real['max_adv']:+.4f} "
              f"real_final={real['final_adv']:+.4f} "
              f"shuf_maxes={[f'{x:+.3f}' for x in sm]} -> {flag} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save(results)
        del sf, wf, ys, X, bits, aux

    print("done.", flush=True)


if __name__ == '__main__':
    main()
