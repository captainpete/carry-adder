# Reproduction: carry-depth analysis of SHA-256d mining

Self-contained code for the two quantitative results in the paper. The
GPU sweep uses `numpy` and `cupy` plus a vendored minimal SHA-256
(`sha256_ref.py`, bit-exact against `hashlib`); `sha256_ref.py` uses CuPy
if a GPU is present and falls back to NumPy otherwise, so the parity check
and small runs work on CPU. `carry_depth.py` needs no third-party
packages at all, and `anchor_sweep.py --pure` runs a standard-library-only
smoke test (no NumPy/CuPy) that self-checks against `hashlib` and
reproduces the cliff in under two seconds, so the headline empirical claim
is verifiable anywhere a Python interpreter runs.

## Files

| file | what it shows | GPU |
|---|---|---|
| `sha256_ref.py` | vendored SHA-256 round and schedule; run directly for a `hashlib` parity check | not needed (CPU fallback) |
| `carry_depth.py` | exact static dataflow: **386** modular-add layers and ~37,200 carries between the block-1 inputs and the SHA-256d output | not needed |
| `anchor_sweep.py` | the carry-aware advantage falls below detection within one round: ~0.886 at depth one, consistent with 0 one round deeper (per-layer retention near 0) | GPU for the full run; `--pure` needs none |

## Run

```bash
python sha256_ref.py        # SHA-256("abc") vendored == hashlib : PASS  (needs NumPy)
python carry_depth.py       # SHA-256d input->output adder depth (min-tree): 386
python anchor_sweep.py      # full run (GPU); auto-uses --pure if NumPy is absent
python anchor_sweep.py --pure   # stdlib-only smoke test (~2 s), parity + cliff
```

On a machine without NumPy, skip `sha256_ref.py` — the `--pure` path
contains its own equivalent `hashlib` parity check.

## Expected output

`carry_depth.py` (pure, instant):
```
SHA-256d input->output adder depth (min-tree):  386
SHA-256d total modular-add count:               1200
```

`anchor_sweep.py` defaults to the paper's run (24 prefixes, 2^22
candidates each) and wants a GPU; it takes tens of minutes there. The
`--pure` smoke path (standard library only, default 2^13 candidates over 3
prefixes) finishes in about two seconds, checks parity against `hashlib`
first, and shows the same cliff at a higher noise floor:
```
  parity: SHA-256('abc') pure == hashlib : PASS  [pure Python]
 j (rounds deep)   mean advantage
              0         +0.894
              1         +0.006        (consistent with zero)
              2         +0.015
              3         -0.023
  RESULT: cliff confirmed — strong advantage at depth one,
          consistent with zero one round deeper (rho ~ 0).
```
The off-cliff residual is larger than the paper's `<=0.001` purely because
the noise floor scales as `1/sqrt(N)` and the smoke run uses ~500x fewer
candidates.

## What each demonstrates

`carry_depth.py` computes the carry-add depth coordinate: the number of
modular-addition (carry-injection) layers separating a controllable
block-1 input bit from a target output bit. Modular addition is the only
SHA-256 operation that couples bit positions nonlinearly (rotations, XOR,
and the Sigma/sigma functions are GF(2)-linear; Ch and Maj are
bit-local), and the Davies-Meyer feed-forward is itself a modular add, so
this depth is the structural axis along which a proof-of-work function
moves from solvable (small depth, as at two rounds) to infeasible. The
computation uses the minimum-depth associative tree (conservative) and
treats Ch and Maj as depth-0 (they inject no carry), so 386 is a floor on
the nonlinear separation. The figure is convention-dependent: a
sequential or nonce-only accounting gives a somewhat larger number.

`anchor_sweep.py` measures how far the strongest local exploit (the
carry-aware score) propagates. The score controls about 89% of the
top-byte magnitude of `state[0]` one layer downstream, and that control
is gone within a single round, because the rotations in Sigma0 scatter
the controlled byte across all bit positions and a fresh carry chain
re-randomizes it. Information is preserved (the round is invertible) but
delocalized off any local observable. With reach near one layer and the
output 386 layers away, no carry/local attack has realizable advantage at
the SHA-256d output. The measurement is over the carry-aware attacker,
the strongest local exploit we have; it does not bound a hypothetical
better local attacker.

See the paper (`../paper/`) for the full argument: the search lower bound,
the amortization (ASICBoost) envelope, and the honest scope. This work
bounds the carry/local-structure attack class. The residual, a global
algebraic shortcut that resolves the target without resolving the path
carries, would be a SHA-256 break, and closing that unconditionally would
require explicit circuit lower bounds beyond current techniques.
