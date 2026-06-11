"""0170 follow-up: does model capacity / data / training time extend the
learned distinguisher's reach to j=1 (one round past the read point)?

The main grid found j>=1 null with a fixed (w1024,b3,2^20,50ep) net. The
null could be a power limit, not a reach limit. Here we scale each axis at
j=1, stem 0, and ask whether the advantage ever rises above its own
CONFIG-MATCHED noise floor.

Methodological crux: max-over-epochs advantage is upward-biased, and the
bias GROWS with model capacity and with the number of epoch evaluations.
A fixed gate would therefore manufacture false signal as we scale. So every
cell is paired with a shuffle-label run of the identical config; the noise
ceiling is that shuffle's max advantage, and "reach detected" means the
real run's max exceeds its shuffle's max by a clear margin. We also report
the final-epoch (unbiased) advantage.

Run:  python scale_j1.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gohr_sweep import make_dataset, selection_adv, FEATS_PER_WORD  # noqa: E402


# ----------------------------------------------------- parameterized model
def init_params(key, in_dim, width, blocks):
    ks = jax.random.split(key, 2 + 2 * blocks)
    def dense(k, i, o):
        return {'w': jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i),
                'b': jnp.zeros(o)}
    p = {'in': dense(ks[0], in_dim, width),
         'out': dense(ks[1], width, 256), 'blocks': []}
    for b in range(blocks):
        p['blocks'].append({'d1': dense(ks[2 + 2 * b], width, width),
                            'd2': dense(ks[3 + 2 * b], width, width)})
    return p


def forward(params, x):
    h = jax.nn.gelu(x @ params['in']['w'] + params['in']['b'])
    for blk in params['blocks']:
        z = jax.nn.gelu(h @ blk['d1']['w'] + blk['d1']['b'])
        z = z @ blk['d2']['w'] + blk['d2']['b']
        h = jax.nn.gelu(h + z)
    return h @ params['out']['w'] + params['out']['b']


def expected_byte(logits):
    return jax.nn.softmax(logits, axis=-1) @ jnp.arange(256.0)


def param_count(p):
    import jax.tree_util as tu
    return int(sum(np.prod(l.shape) for l in tu.tree_leaves(p)))


def train_eval(Xtr, ytr, Xte, yte, *, width, blocks, epochs, batch, seed,
               eval_epochs, lr=1e-3):
    """Train; return {epoch: selection_adv} at eval_epochs, plus param count."""
    n_tr = Xtr.shape[0]
    spe = n_tr // batch
    total = spe * epochs
    sched = optax.piecewise_constant_schedule(lr, {int(0.8 * total): 0.1})
    opt = optax.adam(sched)
    params = init_params(jax.random.PRNGKey(seed), Xtr.shape[1], width, blocks)
    opt_state = opt.init(params)
    ytr_i = ytr.astype(np.int32)

    @jax.jit
    def step(params, opt_state, xb, yb):
        def loss_fn(p):
            return optax.softmax_cross_entropy_with_integer_labels(
                forward(p, xb), yb).mean()
        loss, g = jax.value_and_grad(loss_fn)(params)
        u, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, loss

    @jax.jit
    def pred_chunk(params, xc):
        return expected_byte(forward(params, xc))

    def adv(params):
        cs = 1 << 15
        pred = np.concatenate(
            [np.asarray(pred_chunk(params, jnp.asarray(Xte[i:i + cs])))
             for i in range(0, Xte.shape[0], cs)])
        return selection_adv(pred, yte)

    rng = np.random.default_rng(seed)
    out = {}
    eval_set = set(eval_epochs)
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_tr)
        for s_ in range(spe):
            sl = perm[s_ * batch:(s_ + 1) * batch]
            params, opt_state, loss = step(
                params, opt_state, jnp.asarray(Xtr[sl]), jnp.asarray(ytr_i[sl]))
        if ep in eval_set:
            out[ep] = adv(params)
    pc = param_count(params)
    del params, opt_state
    return out, pc


def cell(tag, Xtr, ytr, Xte, yte, *, width, blocks, epochs, batch, eval_epochs):
    """Real + config-matched shuffle; returns a record dict."""
    t0 = time.time()
    real, pc = train_eval(Xtr, ytr, Xte, yte, width=width, blocks=blocks,
                          epochs=epochs, batch=batch, seed=42,
                          eval_epochs=eval_epochs)
    rng = np.random.default_rng(7)
    shuf, _ = train_eval(Xtr, rng.permutation(ytr), Xte, yte, width=width,
                         blocks=blocks, epochs=epochs, batch=batch, seed=43,
                         eval_epochs=eval_epochs)
    real_max = max(real.values()); shuf_max = max(shuf.values())
    real_fin = real[max(real)]
    gap = real_max - shuf_max
    rec = dict(params=pc, n_train=Xtr.shape[0], width=width, blocks=blocks,
               epochs=epochs, real_max=real_max, real_final=real_fin,
               shuffle_max=shuf_max, gap=gap, real_traj=real, shuf_traj=shuf,
               reach=bool(gap > 0.02))   # >~1.5x test SE over the matched floor
    print(f"[{tag}] params={pc/1e6:.1f}M n=2^{int(np.log2(Xtr.shape[0]))} "
          f"ep={epochs} | real_max={real_max:+.4f} real_final={real_fin:+.4f} "
          f"shuffle_max={shuf_max:+.4f} gap={gap:+.4f} "
          f"-> {'REACH?' if rec['reach'] else 'null'}  ({time.time()-t0:.0f}s)",
          flush=True)
    return rec


def main():
    r_read, J = 30, 1
    j = 1
    n_max, n_test = 1 << 22, 1 << 19
    batch = 8192

    print("0170 scale_j1: capacity / data / time sweep at j=1, stem 0", flush=True)
    print(f"datagen n=2^22+2^19 ...", flush=True)
    t0 = time.time()
    sf, wf, ys, _ = make_dataset(0, r_read, J, n_max + n_test)
    Xall = np.concatenate([sf, wf[:, :FEATS_PER_WORD * (j + 1)]], axis=1)
    yall = ys[j]
    Xte, yte = Xall[n_max:], yall[n_max:]
    print(f"datagen {time.time()-t0:.1f}s, input dim={Xall.shape[1]}", flush=True)

    results = {}

    def slice_train(n):
        return Xall[:n], yall[:n]

    # --- A. capacity (data 2^20, 50 epochs) ---
    print("\n== A. capacity sweep (n=2^20, 50 ep) ==", flush=True)
    Xtr, ytr = slice_train(1 << 20)
    for w, b in [(256, 1), (512, 2), (1024, 3), (2048, 4), (4096, 6)]:
        results[f"cap_w{w}_b{b}"] = cell(
            f"cap w{w} b{b}", Xtr, ytr, Xte, yte, width=w, blocks=b,
            epochs=50, batch=batch, eval_epochs=list(range(1, 51)))

    # --- B. data (base net w1024 b3, 50 epochs) ---
    print("\n== B. data sweep (w1024 b3, 50 ep) ==", flush=True)
    for n in [1 << 16, 1 << 18, 1 << 20, 1 << 22]:
        Xtr, ytr = slice_train(n)
        results[f"data_2e{int(np.log2(n))}"] = cell(
            f"data 2^{int(np.log2(n))}", Xtr, ytr, Xte, yte, width=1024,
            blocks=3, epochs=50, batch=batch, eval_epochs=list(range(1, 51)))

    # --- C. training time (base net, n=2^20, to 400 epochs) ---
    print("\n== C. time sweep (w1024 b3, n=2^20, eval @ 10/25/50/100/200/400) ==",
          flush=True)
    Xtr, ytr = slice_train(1 << 20)
    results["time_base"] = cell(
        "time 400ep", Xtr, ytr, Xte, yte, width=1024, blocks=3, epochs=400,
        batch=batch, eval_epochs=[10, 25, 50, 100, 200, 400])

    with open('scale_j1_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print("\ndone. wrote scale_j1_results.json", flush=True)


if __name__ == '__main__':
    main()
