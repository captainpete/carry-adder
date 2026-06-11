# The Carry Adder Wall

Why SHA-256d (Bitcoin) mining cannot beat brute force unless SHA-256 is
broken. Paper, self-contained reproduction code, and a Gohr-style
neural-distinguisher experiment.

The paper decomposes SHA-256d into a shareable message-schedule layer and
a pseudorandom round layer, proves the random-oracle search bound
(surviving preprocessing and quantum attack), reduces the standard-model
claim to a SHA-256 distinguisher, and introduces **carry depth** — the
count of modular-addition (carry) layers between a controllable input and
a target output — computed exactly at **386 layers** for SHA-256d, with
the strongest measured local advantage dying within a single round.

## Layout

| path | contents |
|---|---|
| `paper/` | `carry_depth.tex` and the built PDF |
| `repro/` | the two headline reproductions: the exact 386-layer accounting and the anchor-sweep advantage cliff; see `repro/README.md` |
| `neural/` | the neural-distinguisher experiment: pre-registered protocol (with documented amendments), training code, and raw per-cell results |

## Quick start (no dependencies)

Both headline results verify on a bare Python interpreter:

```bash
python3 repro/carry_depth.py          # 386-layer accounting (pure Python, instant)
python3 repro/anchor_sweep.py --pure  # advantage cliff + hashlib parity (~2 s)
```

The full-scale anchor sweep uses `numpy` (CPU) or `cupy` (GPU). The
neural experiment (`neural/`) requires a GPU with `jax[cuda12]`, `optax`,
and `numpy`:

```bash
cd neural
python gohr_sweep.py --smoke   # 1 stem, ~30 s
python gohr_sweep.py           # full grid, ~35 min on one RTX 3090
```

## Building the paper

```bash
cd paper && latexmk -pdf carry_depth.tex
```
