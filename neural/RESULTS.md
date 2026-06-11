# 0170 Results: a learned distinguisher hits the same one-round cliff

**Run:** `gohr_sweep.py` (Amendment 4 design), 3 stems x j in {0..4},
2^20 train / 2^19 test nonces per cell, 50 epochs, GELU residual MLP
(~9 nonlinear layers, ~256-way top-byte head). Metric: the paper's
selection advantage (1 - mean_topbyte(selected 1/256 by prediction) /
mean_topbyte(all)), identical to `repro/anchor_sweep.py`. Pre-registered
in `PROTOCOL.md`; gates fixed before results were seen.

## Headline

A neural network **stronger than the paper's hand-built carry-aware
score** reproduces the carry-depth cliff exactly: ~0.998 advantage one
adder layer downstream, indistinguishable from zero one round deeper.

| depth | net max advantage (mean of 3 stems) | hand score | verdict |
|---|---|---|---|
| j=0 (1 adder layer) | **0.9978** (0.9972–0.9990) | 0.8841 | POSITIVE-CONTROL PASS |
| j=1 (1 round, ~3 layers) | 0.0291 | ~0.01 | null |
| j=2 | 0.0260 | ~0.01 | null |
| j=3 | 0.0304 | ~0.01 | null |
| j=4 | 0.0309 | ~0.01 | null |

(j>=1 figures are per-depth means over the 3 stems; per-cell range
0.0211–0.0372. These are MAX-over-epochs statistics and so sit at the
selection-inflated noise floor; the unbiased final-epoch statistic is the
headline null below.)

**Headline null statistic (pooled final-epoch advantage, the unbiased
estimator):** mean **−0.0040** over the 12 j>=1 cells, 95% CI
**[−0.0079, −0.0001]** (j=1 alone: −0.0060, CI [−0.0126, +0.0007]).
Cells within a stem share test nonces, so treating the 12 cells as fully
independent is optimistic; under any correlation treatment the data
exclude a persistent learned advantage above ~0.01 at j>=1. Note the
pre-registered per-cell gate (0.065, 5 sigma of the max statistic) is far
weaker — by itself it only excludes per-cell advantages of roughly
0.04–0.05 — so the pooled final-epoch CI, not the gate, is the
quantitatively meaningful exclusion.

- **Net beats the hand score at j=0 in all three stems** (0.997–0.999 vs
  0.883–0.885). The network learns the carry-in corrections that the
  truncated `(T1>>24)+(T2>>24)` score discards — it is a *strictly
  stronger* member of the carry/local attack class than the one in the
  paper.
- **Every j>=1 cell is null.** 12/12 below the 0.065 (5 sigma,
  max-over-50-epochs) gate. Max over all 12: 0.0372.
- **Shuffle control: 0.0212**, inside the j>=1 band (0.021–0.037). This
  empirically pins the j>=1 readings as the pipeline's max-over-epochs
  noise floor, not residual signal. (The `net_final_adv` column — last
  epoch, no max-selection — averages ~0.00 at j>=1, consistent with zero.)
- No candidate signals -> no escalation runs triggered.

## What this shows

The paper's anchor sweep measured the cliff with *one hand-built score*.
The natural objection (the "AlphaGo / AlphaFold solved a hard problem"
intuition) is that a learned model might find local structure a human
feature misses — exactly what Gohr (CRYPTO 2019) did to round-reduced
Speck, an ARX cipher with the same modular-addition nonlinearity. This
experiment translates Gohr's methodology (train/test split,
advantage-over-chance metric, learned distinguisher) to the mining read
point: a residual net given, as engineered features, everything computed
through the read round (round 30) — state words, carry-free derivations,
the round's modular sums T1/T2, schedule words, with bit, value, and
Fourier encodings — tested at increasing carry depth. (Differences from
Gohr, stated plainly: he used a 1D conv net over bit positions and
ciphertext-difference PAIRS; we use an MLP over single-sample state
features. The conv-architecture variant is run as a review follow-up in
`review_fixes.py`.)

The learned attacker, despite **beating** the hand score where signal
exists (j=0), retains **no** advantage one round deeper. Searching the
local feature space with SGD and ~7.4M parameters does not extend the
reach past a single round. This is direct, in-distribution evidence that
the single-round collapse is a property of the SHA-256 round function's
carry structure, not an artifact of using a weak hand-crafted feature —
which is precisely the "broader sweep over observables, scores, and read
points" that paper Section 6 names as the obvious next test.

## A second finding: where SGD dies on carry composition

The four pre-registered amendments (see `PROTOCOL.md`) document an
unplanned but informative result. The positive control **failed** until
the net was handed T1, T2 of the read round as features:

- Predicting a raw output bit from input bits: net learns nothing
  (parity-like barrier).
- MSE regression on the top byte: loss plateaus at target variance
  (modular target has no linear correlate).
- Fourier features + 256-way classification + multiplicative gating:
  the net **memorizes** 2^20 training examples (train CE 0.09) with
  **zero** test advantage; weight decay does not flip it to
  generalization within budget.

Stated precisely (a round-5 review correction — an earlier version of
this section overclaimed "not even a single carry layer"): the j=0
result shows SGD **can** learn one 2-operand modular addition
(topbyte(T1+T2) from Fourier-featured operands, 0.998), but it **failed
the ~5-adder-layer composition** of computing T1, T2 from their seven
constituent words, at every capacity/feature/optimizer combination
tried. So this attacker's demonstrated arithmetic reach is between 1 and
~5 adder layers; the `ladder` cells in `review_fixes.py` pin the exact
death depth on pure k-operand sums. This matters for interpretation: the
j>=1 nulls confound "SHA-256's carry structure destroys the signal" with
"this optimizer cannot compose modular arithmetic in this format" — the
post-round-31 control in `review_fixes.py` is the discriminating
experiment. Gohr's networks likewise never synthesized cipher
arithmetic; they exploited statistical bias already present in data they
were given.

## Scope / honesty (what this does NOT show)

- This tests a **one-shot learned predictor** — the direct analogue of
  Gohr's distinguisher. It does **not** rule out ML used as a *search
  heuristic* over algebraic representations (Gohr's strongest attacks
  combined the net with classical key-ranking and multi-round search).
  That is the same residual the paper scopes out in Section 6 and does
  not claim to close.
- It is the simplified single-compression read point of
  `anchor_sweep.py` (synthetic stems, no Bitcoin double-hash wrapper),
  faithful to the round-function mechanism but not the full mining map.
- "null" means "below a 5 sigma max-over-epochs gate at 2^19 test
  samples," i.e. consistent with zero at this power; it is not a proof of
  exactly zero. A confirmed signal would have triggered a 4x-data rerun;
  none did.

## Follow-up: capacity / data / training-time scaling at j=1 (`scale_j1.py`)

The main grid's j=1 null could in principle be a power limit rather than a
reach limit. We scaled each axis independently at j=1 (stem 0), with every
cell paired against a config-matched shuffle-label run (max-over-epochs
advantage inflates with capacity and epoch count, so a fixed gate would
manufacture false signal as scale grows; the verdict per cell is
gap = real_max - own_shuffle_max, "reach" iff gap > 0.02):

| axis | range | gaps |
|---|---|---|
| capacity | 0.4M -> 205.7M params | -0.032, +0.012, +0.013, -0.009, ~~-0.002~~ |
| data | 2^16 -> 2^22 samples (64x) | +0.009, +0.003, +0.013, **-0.014** |
| time | 50 -> 400 epochs (8x; grokking probe) | **+0.015** |

> **Round-5 correction:** the 205.7M endpoint is INVALID — its test
> predictions are bit-identical from epoch 35 on (constant-output
> collapse; caught by the round-5 review), and a rerun at lr 3e-4 with
> gradient clipping collapsed again (see follow-up runs below). The
> defensible capacity range is **90x (0.4M -> 35.7M)**.

**11/11 null.** No dose-response on any axis; the two scale endpoints
(205.7M params; 4M samples) sit at or *below* their own shuffle ceilings,
and the 400-epoch trajectory shows no grokking transition (advantage at
epochs 10/25/50/100/200/400: -0.004/+0.005/+0.021/+0.005/+0.013/-0.018 —
the final eval is the most negative). The j=1 null is a **reach limit,
not a power limit**: a 500x capacity range, 64x data range, and 8x
training-time range all fail to move the learned attacker's reach past
one round. Raw numbers in `scale_j1_results.json`.

## Round-5 follow-up runs (`review_fixes.py`, raw in `review_fixes_results.json`)

Five cells responding to the round-5 ML-methodology review, run overnight
with multi-seed shuffle controls and collapse detection:

1. **Post-round-31 control — REPLICATES THE CLIFF.** The Amendment-4
   design at read point 31: j=0 passes at **0.9988**; j=1 max-advantage
   **0.0284, below all three of its own shuffle ceilings**
   (0.031/0.034/0.040); j=2 null (0.0206). The pass-then-cliff pattern is
   not specific to round 30 — the wall tracks the carry boundary. This is
   the discriminating control the review asked for.
2. **k-operand ladder — the attacker's death depth, measured.** Pure
   synthetic top-byte-of-modular-sum, same pipeline: k=2 -> 0.9996,
   k=3 -> 0.9983, then **k=4..7 all dead** (max 0.025-0.041, finals ~0).
   The learned attacker fails exactly at carry composition beyond two
   chained additions — the paper's mechanism in its barest form, on data
   with no SHA-256 structure at all. This also calibrates interpretation
   of the j>=1 nulls: they show this attacker CLASS dies on carry
   composition; they cannot certify SHA-256-specific hardness beyond it.
3. **Conv architecture arm — CLOSED by exhaustion (2026-06-11 update).**
   The original Gohr-faithful conv-over-bit-positions net collapsed at
   epoch 14 and its j=0 power control reached only 0.058 (gate 0.5), so
   its j=1 null was uninterpretable and this objection stood OPEN. It has
   now been re-attempted twice more (`conv_stabilized.py`,
   `conv_dilated.py`): (a) stabilised — warmup + cosine LR, pre-norm
   LayerNorm, zero-init residual conv blocks — at peak LR 5e-4 and 2e-4:
   trains stably (no collapse), control reaches 0.063 / 0.027, FAIL;
   (b) dilated — same stabilisation plus block dilations 1/2/4/8/16
   (receptive field ~125 bit positions, removing the
   receptive-field-shorter-than-the-carry-chain excuse): control 0.036,
   FAIL. Three architectures, stable optimisation, identical ~0.03–0.06
   floor on a task the MLP passes at 0.99+. The conv-over-bit-positions
   inductive bias is affirmatively wrong for carry composition (it is
   matched to XOR-differential locality, the structure Gohr's
   distinguishers exploit, which carry chains do not have). The
   objection is no longer "never got a fair run" — it got three.
4. **Interleaved-split j=1 — null** (0.0297 vs shuffle maxes
   0.021/0.021/0.033). The sequential-nonce coset-split concern is
   closed.
5. **206M rerun — collapsed again** (constant predictions at epoch 34;
   loss pinned at ln 256 throughout — at lr 3e-4 + clipping it never
   learned at all). The w4096/b6 configuration is unstable in this
   format; the cell is excluded as invalid. **The defensible capacity
   range is therefore 90x (0.4M -> 35.7M params, clean endpoints), not
   500x.** The paper's scaling sentence is corrected accordingly.

## Reproduce

```bash
python gohr_sweep.py --smoke   # 1 stem, j in {0,1}, ~30 s
python gohr_sweep.py           # full grid, ~35 min on one 3090
```
Raw per-cell numbers in `results.json`.

## 2026-06-11 — Ladder death-point probes (REVIEW.md item 3)

Scripts: `ladder_probes.py`, `ladder_width_check.py`; raw:
`ladder_probes_results.json`, `ladder_width_check.json`. Same pipeline as
the original ladder (MLP w1024/b3, 30 epochs, 1M train).

The review asked why the ladder dies exactly at k=4 (k=2: 0.9996, k=3:
0.9983, k=4: 0.041) and prescribed three probes. Results:

1. **Curriculum does not move the frontier.** Train k=3 to convergence
   (0.9973), warm-start k=4 from its trunk: max 0.0241 — no better than
   cold (0.0303). Carry composition at k=4/32-bit is not learnable with
   this scaffolding.
2. **Wider Fourier banks do not move it.** 16 harmonics (m=0..15) instead
   of 8: max 0.0238.
3. **The width ladder restructures the question.** Top-byte of k=4
   modular sum at operand widths 8..32 bits, 2-3 seeds per width:

   | width | seeds alive / tried | max_adv range |
   |---|---|---|
   | 8  | 1/2 | 0.03, **1.000** |
   | 12 | 2/2 | 0.99, 1.00 |
   | 16 | 1/2 (+1 at 0.49) | 0.49–0.99 |
   | 20 | 1/2 (+1 at 0.44) | 0.44–0.99 |
   | 24 | 3/3 | 0.95–0.99 |
   | 28 | 1/2 (+1 at 0.92 peak) | 0.05–0.92 |
   | 32 | **0/8** | 0.02–0.047 |

   Below 32 bits, training is **bimodal** — seeds either solve the task
   nearly perfectly or sit at the floor, with no middle ground — and at
   32 bits no seed has ever left the floor (8 independent runs here plus
   the two original cells). The k=4 "death" is therefore not a smooth
   capacity limit but an optimisation cliff whose success probability
   collapses to zero exactly at full width.

**Interpretation and open thread.** The wall is a joint condition
(k ≥ 4 AND width = 32) in this setup: k=2,3 pass at width 32, and k=4
passes below width 32. Two non-exclusive hypotheses, recorded as
hypotheses: (a) the optimisation basin for the carry-composition circuit
shrinks with both k and width, and width 32 at k=4 is where it
effectively vanishes; (b) a float32 artefact with teeth: the value and
harmonic features can only resolve an operand to ~2^-24, so the analog
shortcut (estimate the real-valued sum, read off the top byte) loses the
carry-relevant low bits exactly when width − 8 exceeds the 24-bit
mantissa — i.e., at width 32 — forcing the model onto pure bit-level
carry circuitry, which this attacker class cannot learn (the program's
core finding). A discriminating probe, if it ever matters: rebuild the
k=4/width-32 features in float64. For the paper, the honest sentence is:
the frontier does not move under curriculum or feature scaffolding; the
threshold is real for this attacker class, but it marks where the
gradient-descent attacker's last analog shortcut runs out rather than an
information boundary — one more face of "carries destroy locality."
