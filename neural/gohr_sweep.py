"""0170: Gohr-style neural distinguisher vs the carry-depth cliff.

See PROTOCOL.md (pre-registered, incl. Amendment 1) for design and gates.
Self-contained: numpy data generation (SHA-256 rounds per FIPS 180-4,
mirroring repro/sha256_ref.py), pure-JAX residual MLP, optax Adam.

Target: regress the top byte of state[0] at depth j; metric is the paper's
selection advantage (select bottom 1/256 by prediction; advantage = 1 -
mean_topbyte(selected)/mean_topbyte(all)). Hand-score baseline uses the
same metric, reproducing the anchor sweep's 0.886-at-j=0 row.

Run:  python gohr_sweep.py          # full grid
      python gohr_sweep.py --smoke  # 1 stem, j in {0,1}, short
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

# Coexist with other GPU tenants: allocate on demand, never preallocate.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax
import jax.numpy as jnp
import optax

# ---------------------------------------------------------------- SHA-256
_K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2], dtype=np.uint32)
_IV = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], dtype=np.uint32)


def ror(x, n):
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))).astype(np.uint32)


def expand_schedule(W16, upto):
    """(N,16) uint32 -> (N,upto+1) message schedule."""
    N = W16.shape[0]
    W = np.zeros((N, upto + 1), dtype=np.uint32)
    W[:, :16] = W16
    for t in range(16, upto + 1):
        w15 = W[:, t - 15]; w2 = W[:, t - 2]
        s0 = ror(w15, 7) ^ ror(w15, 18) ^ (w15 >> np.uint32(3))
        s1 = ror(w2, 17) ^ ror(w2, 19) ^ (w2 >> np.uint32(10))
        W[:, t] = (W[:, t - 16] + s0 + W[:, t - 7] + s1).astype(np.uint32)
    return W


def apply_round(state, k, w):
    """One SHA-256 round on (N,8) state; returns (new_state, t1, t2)."""
    a, b, c, d = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    e, f, g, h = state[:, 4], state[:, 5], state[:, 6], state[:, 7]
    s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
    s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
    ch = (e & f) ^ (~e & g)
    maj = (a & b) ^ (a & c) ^ (b & c)
    t1 = (h + s1 + ch + np.uint32(k) + w).astype(np.uint32)
    t2 = (s0 + maj).astype(np.uint32)
    new = np.stack([(t1 + t2).astype(np.uint32), a, b, c,
                    (d + t1).astype(np.uint32), e, f, g], axis=1)
    return new, t1, t2


def stem_words(seed):
    h = hashlib.sha256(f"stem-{seed}".encode()).digest()
    base = np.frombuffer(h, dtype=">u4").astype(np.uint32)
    w = np.zeros(16, dtype=np.uint32)
    w[:8] = base
    w[8:] = base ^ np.uint32(0x9e3779b9)
    return w


FEATS_PER_WORD = 50  # 32 bits + value + topbyte + 8 harmonics x (sin,cos)


def word_features(words):
    """(N,k) uint32 -> (N, k*50) float32: per word, 32 bits (MSB-first),
    value/2^32, topbyte/256, and sin/cos(2*pi*2^m*value) for m=0..7
    (Fourier features at byte harmonics; Amendment 3)."""
    N, k = words.shape
    be = words.astype('>u4').view(np.uint8).reshape(N, 4 * k)
    bits = np.unpackbits(be, axis=1).reshape(N, k, 32).astype(np.float32)
    v = words.astype(np.float64) / 2**32                     # (N,k) in [0,1)
    val = v.astype(np.float32)[:, :, None]
    top = ((words >> np.uint32(24)).astype(np.float32) / 256.0)[:, :, None]
    harm = []
    for m in range(8):
        ang = 2.0 * np.pi * np.mod(v * (1 << m), 1.0)
        harm.append(np.sin(ang).astype(np.float32)[:, :, None])
        harm.append(np.cos(ang).astype(np.float32)[:, :, None])
    return np.concatenate([bits, val, top] + harm, axis=2).reshape(
        N, k * FEATS_PER_WORD)


# ----------------------------------------------------------- data per stem
def make_dataset(stem_seed, r_read, J, n_total):
    """Returns: read-point state features, schedule-word features (J+1
    words), top-byte targets per depth j, and the hand score."""
    w16 = np.broadcast_to(stem_words(stem_seed)[None, :], (n_total, 16)).copy()
    w16[:, 3] = np.arange(n_total, dtype=np.uint32)        # the nonce word
    W = expand_schedule(w16, r_read + J)

    state = np.broadcast_to(_IV[None, :], (n_total, 8)).copy()
    for r in range(r_read):
        state, _, _ = apply_round(state, _K[r], W[:, r])

    # Carry-free (depth-0) derived words at the read point: the GF(2)-linear
    # and bit-local round operations, free to the carry/local attacker.
    a, b_, c_ = state[:, 0], state[:, 1], state[:, 2]
    e, f_, g_ = state[:, 4], state[:, 5], state[:, 6]
    derived = np.stack([
        ror(a, 2) ^ ror(a, 13) ^ ror(a, 22),       # Sigma0(a)
        ror(e, 6) ^ ror(e, 11) ^ ror(e, 25),       # Sigma1(e)
        (e & f_) ^ (~e & g_),                       # Ch(e,f,g)
        (a & b_) ^ (a & c_) ^ (b_ & c_),            # Maj(a,b,c)
    ], axis=1).astype(np.uint32)

    ys, score, t1t2 = [], None, None
    s = state
    for j in range(J + 1):
        s, t1, t2 = apply_round(s, _K[r_read + j], W[:, r_read + j])
        if j == 0:
            score = ((t1 >> np.uint32(24)) + (t2 >> np.uint32(24))).astype(np.int32)
            t1t2 = np.stack([t1, t2], axis=1)
        ys.append((s[:, 0] >> np.uint32(24)).astype(np.uint8))  # top byte

    # Amendment 4: the attacker computes round r_read in full; T1,T2 of the
    # read round are feature words (the anchor sweep's own read point).
    state_feat = word_features(np.concatenate([state, derived, t1t2], axis=1))
    sched_feat = word_features(W[:, r_read:r_read + J + 1])
    return state_feat, sched_feat, ys, score


def selection_adv(pred_small_first, y, frac=1.0 / 256.0):
    """The paper's metric: 1 - mean_topbyte(selected)/mean_topbyte(all),
    selecting the frac fraction predicted smallest."""
    k = max(1, int(len(y) * frac))
    sel = np.argpartition(pred_small_first, k)[:k]
    mean_all = float(y.mean())
    return 1.0 - float(y[sel].mean()) / mean_all if mean_all > 0 else float('nan')


# ------------------------------------------------------------------- model
WIDTH, BLOCKS = 1024, 3


def init_params(key, in_dim):
    ks = jax.random.split(key, 2 + 2 * BLOCKS)
    def dense(k, i, o):
        return {'w': jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i),
                'b': jnp.zeros(o)}
    params = {'in': dense(ks[0], in_dim, WIDTH),
              'out': dense(ks[1], WIDTH, 256), 'blocks': []}
    for b in range(BLOCKS):
        params['blocks'].append({'d1': dense(ks[2 + 2 * b], WIDTH, WIDTH),
                                 'd2': dense(ks[3 + 2 * b], WIDTH, WIDTH)})
    return params


def forward(params, x):
    """Returns 256-way logits over the top byte."""
    h = jax.nn.gelu(x @ params['in']['w'] + params['in']['b'])
    for blk in params['blocks']:
        z = jax.nn.gelu(h @ blk['d1']['w'] + blk['d1']['b'])
        z = z @ blk['d2']['w'] + blk['d2']['b']
        h = jax.nn.gelu(h + z)
    return h @ params['out']['w'] + params['out']['b']


def expected_byte(logits):
    p = jax.nn.softmax(logits, axis=-1)
    return p @ jnp.arange(256.0)


def run_cell(Xtr, ytr, Xte, yte, *, epochs, batch, seed, lr=1e-3):
    """Train one regression cell; return (max_sel_adv, final_sel_adv)."""
    n_tr = Xtr.shape[0]
    steps_per_epoch = n_tr // batch
    total = steps_per_epoch * epochs
    sched = optax.piecewise_constant_schedule(lr, {int(0.8 * total): 0.1})
    opt = optax.adam(sched)

    key = jax.random.PRNGKey(seed)
    params = init_params(key, Xtr.shape[1])
    opt_state = opt.init(params)

    # Data stays on HOST (shared GPU); only per-batch slices are transferred.
    ytr_i = ytr.astype(np.int32)

    @jax.jit
    def step(params, opt_state, xb, yb):
        def loss_fn(p):
            return optax.softmax_cross_entropy_with_integer_labels(
                forward(p, xb), yb).mean()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def predict_chunk(params, xc):
        return expected_byte(forward(params, xc))

    def predict_test(params):
        cs = 1 << 15
        return np.concatenate(
            [np.asarray(predict_chunk(params, jnp.asarray(Xte[i:i + cs])))
             for i in range(0, Xte.shape[0], cs)])

    rng = np.random.default_rng(seed)
    max_adv, final_adv = -1.0, 0.0
    for ep in range(epochs):
        perm = rng.permutation(n_tr)
        for s_ in range(steps_per_epoch):
            sl = perm[s_ * batch:(s_ + 1) * batch]
            params, opt_state, loss = step(
                params, opt_state, jnp.asarray(Xtr[sl]), jnp.asarray(ytr_i[sl]))
        adv = selection_adv(predict_test(params), yte)
        max_adv = max(max_adv, adv)
        final_adv = adv
    del params, opt_state
    return max_adv, final_adv


# -------------------------------------------------------------------- main
def main():
    smoke = '--smoke' in sys.argv[1:]
    r_read, J = 30, 4
    stems = [0] if smoke else [0, 1, 2]
    js = [0, 1] if smoke else list(range(J + 1))
    n_train, n_test = (1 << 17, 1 << 16) if smoke else (1 << 20, 1 << 19)
    epochs, batch = (8, 4096) if smoke else (50, 8192)
    null_gate, power_gate = 0.065, 0.5

    print(f"0170 gohr_sweep (Amendment 4): r_read={r_read} stems={stems} "
          f"js={js} n_train=2^{n_train.bit_length()-1} "
          f"n_test=2^{n_test.bit_length()-1} epochs={epochs} "
          f"{'[SMOKE]' if smoke else ''}", flush=True)

    results = {}
    for stem in stems:
        t0 = time.time()
        sf, wf, ys, score = make_dataset(stem, r_read, J, n_train + n_test)
        print(f"[stem {stem}] datagen {time.time()-t0:.1f}s", flush=True)
        for j in js:
            X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
            Xtr, Xte = X[:n_train], X[n_train:]
            ytr, yte = ys[j][:n_train], ys[j][n_train:]
            sc_adv = selection_adv(score[n_train:], yte)
            t1 = time.time()
            mx, fin = run_cell(Xtr, ytr, Xte, yte,
                               epochs=epochs, batch=batch,
                               seed=1000 * stem + j)
            verdict = ('POSITIVE-CONTROL ' + ('PASS' if mx >= power_gate else 'FAIL')
                       if j == 0 else
                       ('CANDIDATE SIGNAL' if mx > null_gate else 'null'))
            print(f"[stem {stem} j={j}] score_adv={sc_adv:+.4f} "
                  f"net_max_adv={mx:+.4f} net_final_adv={fin:+.4f} "
                  f"({time.time()-t1:.0f}s)  -> {verdict}", flush=True)
            results[f"stem{stem}_j{j}"] = dict(
                score_adv=sc_adv, net_max_adv=mx, net_final_adv=fin,
                verdict=verdict)

        if stem == 0:
            # shuffle control at j=1: permuted training targets
            j = 1
            X = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
            rng = np.random.default_rng(7)
            ytr_shuf = rng.permutation(ys[j][:n_train])
            mx, fin = run_cell(X[:n_train], ytr_shuf,
                               X[n_train:], ys[j][n_train:],
                               epochs=epochs, batch=batch, seed=777)
            ok = mx <= null_gate
            print(f"[stem 0 SHUFFLE j=1] net_max_adv={mx:+.4f} "
                  f"-> noise ceiling {'OK (<= gate)' if ok else 'EXCEEDS GATE'}",
                  flush=True)
            results['shuffle_control_j1'] = dict(
                net_max_adv=mx, net_final_adv=fin, ok=bool(ok))

    out = 'results_smoke.json' if smoke else 'results.json'
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
