"""0170 review-response runs (ML referee, round 5). Five cells, in order of
load-bearing-ness, each with multi-seed shuffle controls and collapse
detection. Results written incrementally to review_fixes_results.json.

Cells:
  1. post31    — Amendment-4 design replicated at read point 31 (j in 0..2).
                 If the pass-then-cliff pattern appears at a SHIFTED read
                 point, the wall tracks the carry boundary, not round 30.
  2. ladder    — toy k-operand modular sums (k=2..7), same pipeline: pins
                 where MLP+SGD dies on pure adder composition.
  3. conv      — Gohr-faithful 1D residual conv over BIT POSITIONS (words
                 as channels) + onecycle LR, at j=1: the inductive bias
                 whose absence most plausibly biased toward null.
  4. interleaved — j=1 with a randomly interleaved train/test split
                 (closes the sequential-nonce coset asymmetry).
  5. big_rerun — the 206M cell rerun at lr 3e-4 with gradient clipping and
                 divergence/collapse detection (the prior run collapsed at
                 epoch 35; its verdict was uninformative).

Run:  python review_fixes.py
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
    make_dataset, selection_adv, word_features, FEATS_PER_WORD)

RESULTS_PATH = 'review_fixes_results.json'


def save(results):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)


# ---------------------------------------------------------------- models
def dense(k, i, o):
    return {'w': jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i),
            'b': jnp.zeros(o)}


def init_mlp(key, in_dim, width, blocks):
    ks = jax.random.split(key, 2 + 2 * blocks)
    p = {'in': dense(ks[0], in_dim, width),
         'out': dense(ks[1], width, 256), 'blocks': []}
    for b in range(blocks):
        p['blocks'].append({'d1': dense(ks[2 + 2 * b], width, width),
                            'd2': dense(ks[3 + 2 * b], width, width)})
    return p


def fwd_mlp(p, x):
    h = jax.nn.gelu(x @ p['in']['w'] + p['in']['b'])
    for blk in p['blocks']:
        z = jax.nn.gelu(h @ blk['d1']['w'] + blk['d1']['b'])
        z = z @ blk['d2']['w'] + blk['d2']['b']
        h = jax.nn.gelu(h + z)
    return h @ p['out']['w'] + p['out']['b']


CONV_CH, CONV_BLOCKS, CONV_K = 64, 5, 3


def init_conv(key, n_words, aux_dim):
    ks = jax.random.split(key, 4 + 2 * CONV_BLOCKS)
    def conv_w(k, kw, ci, co):
        return {'w': jax.random.normal(k, (kw, ci, co)) * jnp.sqrt(2.0 / (kw * ci)),
                'b': jnp.zeros(co)}
    p = {'in': conv_w(ks[0], 1, n_words, CONV_CH), 'blocks': [],
         'd1': dense(ks[1], 32 * CONV_CH + aux_dim, 512),
         'out': dense(ks[2], 512, 256)}
    for b in range(CONV_BLOCKS):
        p['blocks'].append({'c1': conv_w(ks[3 + 2 * b], CONV_K, CONV_CH, CONV_CH),
                            'c2': conv_w(ks[4 + 2 * b], CONV_K, CONV_CH, CONV_CH)})
    return p


def _conv(x, w):
    return jax.lax.conv_general_dilated(
        x, w['w'], window_strides=(1,), padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')) + w['b']


def fwd_conv(p, xb_bits, xb_aux):
    # xb_bits: (N, 32, n_words) bits along positions, words as channels
    h = jax.nn.gelu(_conv(xb_bits, p['in']))
    for blk in p['blocks']:
        z = jax.nn.gelu(_conv(h, blk['c1']))
        z = _conv(z, blk['c2'])
        h = jax.nn.gelu(h + z)
    flat = h.reshape(h.shape[0], -1)
    hd = jax.nn.gelu(jnp.concatenate([flat, xb_aux], 1) @ p['d1']['w'] + p['d1']['b'])
    return hd @ p['out']['w'] + p['out']['b']


def expected_byte(logits):
    return jax.nn.softmax(logits, axis=-1) @ jnp.arange(256.0)


# --------------------------------------------------------------- trainer
def train_mlp(Xtr, ytr, Xte, yte, *, width, blocks, epochs, batch, seed,
              lr=1e-3, clip=None):
    """Returns dict with max/final adv, per-epoch trajectory, loss log,
    collapse flag."""
    n_tr = Xtr.shape[0]
    spe = n_tr // batch
    total = spe * epochs
    sched = optax.piecewise_constant_schedule(lr, {int(0.8 * total): 0.1})
    tx = [optax.clip_by_global_norm(clip)] if clip else []
    opt = optax.chain(*tx, optax.adam(sched))
    params = init_mlp(jax.random.PRNGKey(seed), Xtr.shape[1], width, blocks)
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
    traj, losses = {}, {}
    collapse = None
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_tr)
        for s_ in range(spe):
            sl = perm[s_ * batch:(s_ + 1) * batch]
            params, opt_state, loss = step(
                params, opt_state, jnp.asarray(Xtr[sl]), jnp.asarray(ytr_i[sl]))
        lossf = float(loss)
        pred = preds(params)
        traj[ep] = selection_adv(pred, yte)
        losses[ep] = lossf
        if not np.isfinite(lossf):
            collapse = f'nonfinite loss at ep{ep}'
            break
        if float(np.std(pred)) < 1e-5:
            collapse = f'constant predictions at ep{ep}'
            break
    advs = list(traj.values())
    rec = dict(max_adv=max(advs), final_adv=advs[-1], traj=traj, loss=losses,
               collapse=collapse)
    del params, opt_state
    return rec


def train_conv(bits_tr, aux_tr, ytr, bits_te, aux_te, yte, *, epochs, batch,
               seed, peak_lr=2e-3):
    n_tr = bits_tr.shape[0]
    spe = n_tr // batch
    total = spe * epochs
    sched = optax.cosine_onecycle_schedule(total, peak_lr)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    params = init_conv(jax.random.PRNGKey(seed), bits_tr.shape[2], aux_tr.shape[1])
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


def verdict(real_max, shuf_maxes):
    """REACH iff the real max clears every shuffle draw's max by 0.02."""
    return bool(real_max > max(shuf_maxes) + 0.02)


def run_mlp_cell(tag, Xtr, ytr, Xte, yte, *, width=1024, blocks=3, epochs=50,
                 batch=8192, n_shuffles=3, lr=1e-3, clip=None, results=None):
    t0 = time.time()
    real = train_mlp(Xtr, ytr, Xte, yte, width=width, blocks=blocks,
                     epochs=epochs, batch=batch, seed=42, lr=lr, clip=clip)
    shufs = []
    for i in range(n_shuffles):
        rng = np.random.default_rng(100 + i)
        shufs.append(train_mlp(Xtr, rng.permutation(ytr), Xte, yte,
                               width=width, blocks=blocks, epochs=epochs,
                               batch=batch, seed=200 + i, lr=lr, clip=clip))
    sm = [s['max_adv'] for s in shufs]
    rec = dict(real_max=real['max_adv'], real_final=real['final_adv'],
               shuffle_maxes=sm, collapse=real['collapse'],
               shuffle_collapses=[s['collapse'] for s in shufs],
               real_traj=real['traj'], real_loss=real['loss'],
               reach=verdict(real['max_adv'], sm))
    flag = 'COLLAPSED' if real['collapse'] else (
        'REACH?' if rec['reach'] else 'null')
    print(f"[{tag}] real_max={real['max_adv']:+.4f} "
          f"real_final={real['final_adv']:+.4f} "
          f"shuf_maxes={[f'{x:+.3f}' for x in sm]} -> {flag} "
          f"({time.time()-t0:.0f}s)", flush=True)
    results[tag] = rec
    save(results)
    return rec


# ------------------------------------------------------------------ main
def main():
    results = {}
    r30 = 30

    # ---- 1. post31: Amendment-4 design at read point 31 ----
    print("== 1. post31 control (r_read=31, stem 0, j in 0..2) ==", flush=True)
    n_train, n_test = 1 << 20, 1 << 19
    sf, wf, ys, _ = make_dataset(0, 31, 2, n_train + n_test)
    for j in [0, 1, 2]:
        X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
        n_sh = 3 if j == 1 else 1
        run_mlp_cell(f"post31_j{j}", X[:n_train], ys[j][:n_train],
                     X[n_train:], ys[j][n_train:], n_shuffles=n_sh,
                     results=results)
    del sf, wf, ys

    # ---- 2. ladder: toy k-operand modular sums ----
    print("== 2. toy ladder: topbyte of k-operand sum mod 2^32 ==", flush=True)
    n_l, n_lt = 1 << 20, 1 << 18
    rng = np.random.default_rng(5)
    for k in range(2, 8):
        ops = rng.integers(0, 2**32, size=(n_l + n_lt, k), dtype=np.uint64
                           ).astype(np.uint32)
        y = ((ops.astype(np.uint64).sum(1) % (1 << 32)) >> 24).astype(np.uint8)
        X = word_features(ops)
        rec = train_mlp(X[:n_l], y[:n_l], X[n_l:], y[n_l:], width=1024,
                        blocks=3, epochs=30, batch=8192, seed=42)
        print(f"[ladder_k{k}] max_adv={rec['max_adv']:+.4f} "
              f"final={rec['final_adv']:+.4f}", flush=True)
        results[f"ladder_k{k}"] = dict(max_adv=rec['max_adv'],
                                       final_adv=rec['final_adv'],
                                       traj=rec['traj'])
        save(results)

    # ---- data for cells 3-5 (read point 30, j=1) ----
    print("== datagen for conv/interleaved/big (r_read=30, j=1) ==", flush=True)
    j = 1
    sf, wf, ys, _ = make_dataset(0, r30, 1, n_train + n_test)
    X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
    yall = ys[j]

    # ---- 3. conv (Gohr-faithful) at j=1 ----
    print("== 3. conv over bit positions, j=1 ==", flush=True)
    n_words = X.shape[1] // FEATS_PER_WORD
    Xw = X.reshape(-1, n_words, FEATS_PER_WORD)
    bits = np.ascontiguousarray(
        Xw[:, :, :32].transpose(0, 2, 1))          # (N, 32, n_words)
    aux = np.ascontiguousarray(
        Xw[:, :, 32:].reshape(Xw.shape[0], -1))    # (N, n_words*18)
    t0 = time.time()
    realc = train_conv(bits[:n_train], aux[:n_train], yall[:n_train],
                       bits[n_train:], aux[n_train:], yall[n_train:],
                       epochs=40, batch=4096, seed=42)
    shufc = []
    for i in range(2):
        rngs = np.random.default_rng(100 + i)
        shufc.append(train_conv(bits[:n_train], aux[:n_train],
                                rngs.permutation(yall[:n_train]),
                                bits[n_train:], aux[n_train:], yall[n_train:],
                                epochs=40, batch=4096, seed=200 + i))
    smc = [s['max_adv'] for s in shufc]
    results['conv_j1'] = dict(real_max=realc['max_adv'],
                              real_final=realc['final_adv'],
                              shuffle_maxes=smc, collapse=realc['collapse'],
                              real_traj=realc['traj'],
                              reach=verdict(realc['max_adv'], smc))
    print(f"[conv_j1] real_max={realc['max_adv']:+.4f} "
          f"real_final={realc['final_adv']:+.4f} "
          f"shuf_maxes={[f'{x:+.3f}' for x in smc]} -> "
          f"{'REACH?' if results['conv_j1']['reach'] else 'null'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    save(results)
    # conv positive control at j=0 (sanity that the conv arch can learn):
    sf0, wf0, ys0, _ = make_dataset(0, r30, 0, (1 << 18) + (1 << 17))
    X0 = np.concatenate([sf0, wf0[:, :FEATS_PER_WORD]], axis=1)
    n0 = 1 << 18
    Xw0 = X0.reshape(-1, X0.shape[1] // FEATS_PER_WORD, FEATS_PER_WORD)
    bits0 = np.ascontiguousarray(Xw0[:, :, :32].transpose(0, 2, 1))
    aux0 = np.ascontiguousarray(Xw0[:, :, 32:].reshape(Xw0.shape[0], -1))
    rc0 = train_conv(bits0[:n0], aux0[:n0], ys0[0][:n0], bits0[n0:],
                     aux0[n0:], ys0[0][n0:], epochs=15, batch=4096, seed=42)
    results['conv_j0_control'] = dict(max_adv=rc0['max_adv'],
                                      final_adv=rc0['final_adv'])
    print(f"[conv_j0_control] max_adv={rc0['max_adv']:+.4f} "
          f"(power gate 0.5)", flush=True)
    save(results)
    del sf0, wf0, ys0, X0, Xw0, bits0, aux0, bits, aux, Xw

    # ---- 4. interleaved split at j=1 ----
    print("== 4. interleaved-split j=1 ==", flush=True)
    rngp = np.random.default_rng(11)
    perm = rngp.permutation(n_train + n_test)
    tr, te = perm[:n_train], perm[n_train:]
    run_mlp_cell("interleaved_j1", X[tr], yall[tr], X[te], yall[te],
                 n_shuffles=3, results=results)

    # ---- 5. 206M rerun with clipping + lower lr ----
    print("== 5. big rerun: w4096 b6, lr 3e-4, clip 1.0, j=1 ==", flush=True)
    run_mlp_cell("big_rerun_j1", X[:n_train], yall[:n_train],
                 X[n_train:], yall[n_train:], width=4096, blocks=6,
                 epochs=50, n_shuffles=2, lr=3e-4, clip=1.0, results=results)

    print("done.", flush=True)


if __name__ == '__main__':
    main()
