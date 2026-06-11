"""Dilated-conv arm: the receptive-field fix (REVIEW.md item 1, attempt 3).

Both stabilised attempts (conv_stabilized.py: warmup + LayerNorm +
zero-init residual blocks, peak LR 5e-4 then 2e-4) train stably but still
fail the j=0 power control (0.063 / 0.027 vs gate 0.5). Diagnosis: with
kernel 3 and 5 blocks the conv trunk's receptive field is ~21 bit
positions, but the j=0 task (top byte of a 32-bit modular sum, given the
operand words as bit-channels) needs carry information propagated across
24+ positions — the trunk structurally cannot see the carry chain, and
the dense head alone is too small to do the carry work (the original
conv arm and both stabilised reruns all landed at the same ~0.03-0.06
floor an MLP-head-only model would).

This variant changes ONE thing: block dilations [1, 2, 4, 8, 16], giving
a receptive field of ~125 positions. If the conv now passes the control,
the j>=1 cells finally count and the conv-inductive-bias objection gets a
real answer; if it fails with a receptive field 4x the word width, the
"never got a fair run" objection is closed by exhaustion — three
attempts, the last with no structural excuse.

Run:  python conv_dilated.py     (results to conv_dilated_results.json)
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
from conv_stabilized import conv_arrays, _ln  # noqa: E402

RESULTS_PATH = 'conv_dilated_results.json'
CONV_CH, CONV_K = 64, 3
DILATIONS = [1, 2, 4, 8, 16]
R_READ = 30
POWER_GATE = 0.5
PEAK_LR = 5e-4


def save(results):
    with open(RESULTS_PATH, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)


def init_conv(key, n_words, aux_dim):
    ks = jax.random.split(key, 4 + 2 * len(DILATIONS))

    def conv_w(k, kw, ci, co, zero=False):
        w = (jnp.zeros((kw, ci, co)) if zero else
             jax.random.normal(k, (kw, ci, co)) * jnp.sqrt(2.0 / (kw * ci)))
        return {'w': w, 'b': jnp.zeros(co)}

    p = {'in': conv_w(ks[0], 1, n_words, CONV_CH), 'blocks': [],
         'd1': dense(ks[1], 32 * CONV_CH + aux_dim, 512),
         'out': dense(ks[2], 512, 256)}
    for b in range(len(DILATIONS)):
        p['blocks'].append({
            'c1': conv_w(ks[3 + 2 * b], CONV_K, CONV_CH, CONV_CH),
            'c2': conv_w(ks[4 + 2 * b], CONV_K, CONV_CH, CONV_CH, zero=True),
            'ln_g': jnp.ones(CONV_CH), 'ln_b': jnp.zeros(CONV_CH)})
    return p


def _conv(x, w, dilation=1):
    return jax.lax.conv_general_dilated(
        x, w['w'], window_strides=(1,), padding='SAME',
        rhs_dilation=(dilation,),
        dimension_numbers=('NWC', 'WIO', 'NWC')) + w['b']


def fwd_conv(p, xb_bits, xb_aux):
    h = jax.nn.gelu(_conv(xb_bits, p['in']))
    for blk, d in zip(p['blocks'], DILATIONS):
        z = _ln(h, blk['ln_g'], blk['ln_b'])
        z = jax.nn.gelu(_conv(z, blk['c1'], dilation=d))
        z = _conv(z, blk['c2'], dilation=d)
        h = h + z
    flat = h.reshape(h.shape[0], -1)
    hd = jax.nn.gelu(jnp.concatenate([flat, xb_aux], 1)
                     @ p['d1']['w'] + p['d1']['b'])
    return hd @ p['out']['w'] + p['out']['b']


def train_conv(bits_tr, aux_tr, ytr, bits_te, aux_te, yte, *, epochs, batch,
               seed, peak_lr=PEAK_LR):
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


def main():
    results = {}

    print("== conv_dilated: j=0 power control ==", flush=True)
    n0, n0t = 1 << 18, 1 << 17
    sf0, wf0, ys0, _ = make_dataset(0, R_READ, 0, n0 + n0t)
    X0 = np.concatenate([sf0, wf0[:, :FEATS_PER_WORD]], axis=1)
    bits0, aux0 = conv_arrays(X0, X0.shape[1] // FEATS_PER_WORD)
    t0 = time.time()
    rc = train_conv(bits0[:n0], aux0[:n0], ys0[0][:n0], bits0[n0:],
                    aux0[n0:], ys0[0][n0:], epochs=25, batch=4096, seed=42)
    ok = rc['max_adv'] >= POWER_GATE and not rc['collapse']
    results['control_j0'] = dict(max_adv=rc['max_adv'],
                                 final_adv=rc['final_adv'],
                                 collapse=rc['collapse'], traj=rc['traj'],
                                 loss=rc['loss'],
                                 verdict='PASS' if ok else 'FAIL')
    print(f"[control_j0 dilated] max_adv={rc['max_adv']:+.4f} "
          f"final={rc['final_adv']:+.4f} collapse={rc['collapse']} "
          f"gate {POWER_GATE} -> {'PASS' if ok else 'FAIL'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    save(results)
    del sf0, wf0, ys0, X0, bits0, aux0
    if not ok:
        print("dilated control failed -> conv arm three times attempted; "
              "objection closes by exhaustion.", flush=True)
        return

    n_train, n_test = 1 << 20, 1 << 19
    for j, n_sh in ((1, 3), (2, 1)):
        print(f"== conv_dilated: j={j} ({n_sh} shuffles) ==", flush=True)
        sf, wf, ys, _ = make_dataset(0, R_READ, j, n_train + n_test)
        X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
        yall = ys[j]
        bits, aux = conv_arrays(X, X.shape[1] // FEATS_PER_WORD)
        t0 = time.time()
        real = train_conv(bits[:n_train], aux[:n_train], yall[:n_train],
                          bits[n_train:], aux[n_train:], yall[n_train:],
                          epochs=40, batch=4096, seed=42)
        shufs = []
        for i in range(n_sh):
            rngs = np.random.default_rng(100 + i)
            shufs.append(train_conv(
                bits[:n_train], aux[:n_train],
                rngs.permutation(yall[:n_train]),
                bits[n_train:], aux[n_train:], yall[n_train:],
                epochs=40, batch=4096, seed=200 + i))
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
