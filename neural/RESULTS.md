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

**Interpretation and open thread (RESOLVED 2026-06-15).** The wall is a
joint condition (k ≥ 4 AND width = 32) in this setup: k=2,3 pass at
width 32, and k=4 passes below width 32. Two non-exclusive hypotheses were
recorded: (a) the optimisation basin for the carry-composition circuit
shrinks with both k and width, and width 32 at k=4 is where it
effectively vanishes; (b) a float32 artefact with teeth: the value and
harmonic features can only resolve an operand to ~2^-24, so the analog
shortcut (estimate the real-valued sum, read off the top byte) loses the
carry-relevant low bits exactly when width − 8 exceeds the 24-bit
mantissa — i.e., at width 32 — forcing the model onto pure bit-level
carry circuitry, which this attacker class cannot learn (the program's
core finding).

The discriminating probe named here — rebuild the k=4/width-32 task in
float64 — has now been run (`float64_probe.py`, see the dated section
below). **Hypothesis (b) is disfavoured: full float64 does not move the
wall.** Six seeds paired arm-for-arm (identical operands and identical
init seed, float32 vs full float64 the only difference) all sit at the
~0.03–0.09 floor in both arms, tracking each other to within ~0.01;
0/6 alive in either precision. Restoring 2^-53 operand resolution does
not revive the task, so the carry-relevant low bits falling below the
float32 mantissa is **not** what kills k=4/width-32. The supported
explanation is (a): a precision-independent optimisation/trainability
barrier. The honest paper sentence is therefore stronger than the earlier
hedge: the threshold is real for this attacker class and is **not** a
float32 feature-resolution artefact — there is no analog shortcut for
gradient descent to ride once carry composition crosses k ≥ 4 at full
width, and handing SGD exact 64-bit features does not change that. One
more face of "carries destroy locality."

## 2026-06-15 — Float64 probe: the analog-shortcut hypothesis, tested

Script: `float64_probe.py`; raw: `float64_probe_results.json`. The
discriminating probe for the k=4/width-32 death point (open thread above).

**Design.** A tightly paired comparison: for each seed, the same integer
operands and the same parameter-init seed are run twice, differing only in
numeric precision — full float64 (features *and* network arithmetic, via
`jax_enable_x64`) vs the float32 baseline. Full float64 is required:
mixed precision is not a valid test, because any cast back to float32
after the features reintroduces the exact 2^-24 wall the hypothesis is
about. Architecture and 30-epoch schedule identical to the ladder; to make
float64 tractable on a 3090 (FP64 measured at 641 ms/step vs 12.5 ms/step
FP32, ~51x slower), training data was reduced 2^20 → 2^18 and seeds to 6.
This is fair because the paired float32 control was run at the *same*
reduced size and also stayed at the floor — the floor is the wall, not
data starvation.

**Result — 0/6 alive in both arms, float64 tracks float32 seed-for-seed:**

| seed | f64 max | f32 max | f64 final | f32 final |
|---|---|---|---|---|
| 43 | +0.0566 | +0.0581 | −0.0137 | −0.0139 |
| 44 | +0.0401 | +0.0412 | −0.0273 | −0.0204 |
| 45 | +0.0622 | +0.0716 | +0.0373 | +0.0467 |
| 46 | +0.0535 | +0.0534 | +0.0181 | +0.0129 |
| 47 | +0.0398 | +0.0332 | +0.0210 | +0.0245 |
| 48 | +0.0835 | +0.0862 | +0.0405 | +0.0305 |

Max advantage anywhere is 0.086, an order of magnitude below the ~1.0 a
revival would show, and within the documented noise floor. Because the
arms are paired (precision the only variable), the null is causal:
restoring 53-bit operand resolution changes nothing.

**Conclusion.** Hypothesis (b), the float32-mantissa artefact, is
disfavoured. The k=4/width-32 wall is a precision-independent
optimisation/trainability barrier (hypothesis (a)). What remains to
characterise is the *structure* of that barrier — see the analysis thread
opened next.

## 2026-06-15 — Structure of the barrier: carry depth vs feature isolation

Two follow-up probes to characterise the precision-independent barrier.
The headline: it has **two separable causes**, and only one of them drives
the paper's experiments. Scripts: `readout_position_probe.py`,
`feature_isolation_probe.py`; raw: `*_results.json`.

### Probe 1 (readout position) — a confounded detour, DO NOT cite as carry-depth evidence

Hypothesis under test: the controlling variable is *carry bits below the
observable*. The width ladder, read as a sweep, is exactly this (the
target is always the top byte, so an operand of width W puts a (W−8)-bit
carry chain below it). Direct test: fix k=4 and full width=32 (so the
input distribution and features are identical), vary only WHICH byte of
the sum is predicted — byte@b has b carry bits below it. Predicted
staircase: byte@0 learnable, byte@24 dead.

Result: **no staircase — all four read-outs dead, 0/16** (best 0.046),
including byte@0. This *falsified* the naive prediction, but the probe is
**self-confounded**: the low byte of the sum cannot depend on the high
bits, so reading out a non-top byte at full width turns the upper bits
into genuine distractor inputs. The death is a feature-isolation artefact
introduced by the probe, not carry-depth evidence. Recorded so we don't
re-cite it.

| read-out | carry bits below | per-seed max_adv | alive |
|---|---|---|---|
| byte@0  |  0 | 0.019 0.017 0.046 0.016 | 0/4 |
| byte@8  |  8 | 0.035 0.024 0.044 0.025 | 0/4 |
| byte@16 | 16 | 0.036 0.024 0.027 0.032 | 0/4 |
| byte@24 | 24 | 0.039 0.034 0.030 0.043 | 0/4 |

### Probe 2 (feature isolation) — the low-byte death is distractor BITS, not analog features

Fix the task (predict the low byte of the k=4/width-32 sum) and vary only
the feature set. The low byte is, mathematically, the same 8-bit modular
addition the width-8 ladder cell learned (1/2 seeds), so it is learnable
*in principle* — what changed is the representation.

| feature set | what it carries | alive | best max | best final |
|---|---|---|---|---|
| `full`   | 32 bits + value + topbyte + 16 Fourier harmonics | 0/4 | 0.034 | 0.022 |
| `bits32` | 32 raw bits, no analog features | 0/4 | 0.063 | 0.020 |
| `low8`   | only the 8 relevant low bits | 0/4* | **0.454** | **0.434** |

\*0/4 by the 0.5 gate, but `low8` seed 45 reached 0.454 max / 0.434 final
(stable at the last epoch — real learning, not max-over-epochs noise),
while the other three sat at the floor. `full`/`bits32` produced nothing
above 0.063 across 8 seeds.

Diagnosis: removing the (misleading) analog features does nothing
(`bits32` still dead); removing the 24 irrelevant high bits is what brings
signal back (`low8` revives, bimodally). **The killer is the distractor
bits, not the analog features.** SGD cannot extract the easy sub-circuit
when its inputs are buried among distractor dimensions.

### What this means — and why the paper's claims are unaffected

The barrier has two separable causes:

1. **Carry-composition depth** — the controlling variable in the
   *distractor-free* experiments. Consistent with the high-degree /
   statistical-query reading: the carry into the observable is a
   high-degree Boolean function of the low bits (flat gradient for SGD,
   bimodal/grok-like, sharper as depth grows).
2. **Feature isolation** — a *separate* weakness of this attacker: it
   cannot pull a relevant sub-circuit out of distractor inputs. Real, but
   in principle addressable (attention, sparsity priors, feature
   selection), so not an intrinsic-hardness claim.

**Crucially, the paper's two neural experiments are distractor-free**, so
cause 2 does not drive them. Both use **top-byte** targets, and the top
byte of a k-operand sum depends on *every* operand bit (a carry can
propagate from bit 0 to the top), so no input bit is irrelevant:

- *k-operand ladder* (top byte, width 32, k=2..7): all 32×k bits
  carry-relevant — the k≥4 death is genuine carry-composition hardness.
- *width ladder* (top byte, operands 8→32 bits, features scale with
  width): all bits relevant at each width — clean carry-depth sweep.

So Probe 2 *rules out* the worry that the ladder nulls were secretly
feature-selection artefacts; combined with the float64 result (not
precision), "the learned attacker expires on carry composition as such"
stands. The paper is left as-is; cause 2 is recorded here as attacker
phenomenology, not a paper claim.

## 2026-06-15 — Probability-space analysis: the k-cliff is the learner's edge, not the carry's

Analytic (no new runs). Treat each input bit as iid Bernoulli(1/2) and
look at the induced law on the carries; this pins down what is and isn't
intrinsic about the k>=4 cliff.

**The k-operand carry chain is Holte's "amazing matrix."** Summing k
operands bit-by-bit, the carry is the integer Markov chain
`c_{i+1} = floor((c_i + S_i)/2)`, `S_i ~ Binomial(k, 1/2)`, on states
`{0,...,k-1}`. Computed by hand (reflection-symmetry block-diagonalisation)
for k=2..5 and matching Holte (1997) / Diaconis–Fulman (2009): the
eigenvalues are exactly `{2^-j : j=0..k-1}`.

| k | states | eigenvalues | λ₂ (gap) |
|---|---|---|---|
| 2 | 2 | 1, ½ | ½ |
| 3 | 3 | 1, ½, ¼ | ½ |
| 4 | 4 | 1, ½, ¼, ⅛ | ½ |
| 5 | 5 | 1, ½, ¼, ⅛, 1/16 | ½ |

**The spectral gap is k-invariant: λ₂ = ½ for every k.** Adding operands
only introduces *faster*-decaying modes (⅛, 1/16, …); it never produces a
slower one. So the carry chain's intrinsic correlation-decay rate (2^-d
per bit position) is set by the *base*, not the operand count. Nothing in
the carry's own mixing singles out k=4.

**Consequence (corroborates the float64 and feature-isolation probes).**
The k>=4 learnability cliff therefore has *no counterpart* in the carry
chain's mixing — it is the **learner's** edge (gradient descent over fixed
features), not an intrinsic property of carry composition. This is a
third, independent line (pure probability) reaching the same place as the
float64 null (not precision) and the feature-isolation probe (a separate
distractor weakness): the wall at k=4 is about the attacker, while the
intrinsic, k-invariant `2^-d` decay is the real cryptographic quantity.
The paper records this as a scope clarification ("One contraction, several
observables") rather than as support for intrinsic hardness.

**Two decay scales (resolves the cliff-vs-slope tension).** The `2^-d`
geometric decay lives on the *bit-position* axis within one addition. The
*round* axis is different: one SHA round applies Σ₀ (re-scatters the
controlled byte) plus a fresh carry chain, i.e. a full mixing time, so the
downstream advantage collapses in one round (the measured cliff) rather
than decaying as a per-round `ρ^d` slope. The naive Markov heuristic
conflated one carry step with one round.

**Stem-yield unification.** The same contraction bounds stem selection:
`|P(hit|stem) − P(hit)| ≲ 2^-d` and `Var_stems ≲ 2^-2d`. For SHA-256d,
d ~ 386 layers, so the bias sits far below the `2^-32` resolution of a
best-of-N nonce search — the bs2 0150 stem-exchangeability null is forced,
not contingent. Stem yield, the hand score, and the learned distinguisher
are three observables of one contraction coefficient (the measured
ρ ≤ 0.0011). Folded into the paper's scope section.
