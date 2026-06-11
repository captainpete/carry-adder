# 0170: Gohr-style neural distinguisher vs the carry-depth cliff

**Pre-registered protocol. Written before the run; gates fixed before results
were seen.**

## Question

The paper's anchor sweep shows the strongest *hand-built* local score (the
carry-aware `(T1>>24)+(T2>>24)`) loses all advantage within one round of its
read point. Gohr (CRYPTO 2019) showed neural networks can find local
structure in reduced-round ARX that human cryptanalysts missed. Does a
learned model retain prediction advantage about a SHA-256 state observable
**more rounds downstream** than the hand-built score?

The carry-depth thesis predicts: no. Each round adds ~3 carry layers; a
fixed-depth network trained by SGD should show the same cliff. If the net
holds advantage at j>=1, the cliff claim (and §6's scoping) must be revised.

## Setup

Mirrors `repro/anchor_sweep.py`: fixed message stem (16 words from
`sha256("stem-{seed}")`), `W[3]` = nonce varying; state advanced through
`r_read = 30` rounds of a single SHA-256 compression from the IV.

- **Inputs to the net:** the 256 state bits at the read point (pre-round
  `r_read`) PLUS the bits of the message-schedule words
  `W[r_read .. r_read+j]` consumed by the rounds being predicted
  (32·(j+1) bits). The target is therefore a **deterministic** function of
  the input: failure cannot be blamed on missing information, only on the
  learnability of composed carry layers at fixed network depth.
- **Target:** bit 31 (top bit) of `state[0]` after applying rounds
  `r_read .. r_read+j` (so j=0 means one round applied — the same "one
  adder layer from the score" point where the hand score gets 0.886
  selection advantage). Balanced ~50/50 binary classification.
- **Metric:** advantage = 2·(test accuracy) − 1, on held-out nonces.
- **Split:** nonces 0..2^20−1 train, 2^20..2^20+2^19−1 test (disjoint,
  same stem — the mining-relevant within-template setting).
- **Baseline:** the hand score's best-threshold classifier (threshold fit
  on train, evaluated on test) for the same targets.

## Model (fixed before running)

Pure-JAX MLP with residual blocks (Gohr-flavored; no conv since our input
has no pair structure): Dense(1024) → GELU → 3× [residual block:
Dense(1024)→GELU→Dense(1024), skip, GELU] → Dense(1). AdamW-free plain
Adam, lr 1e-3 dropping to 1e-4 for the last 20%, batch 8192, 50 epochs
(6400 steps). ~2.4M parameters, ≈8 nonlinear layers.

## Grid

stems {0,1,2} × j {0,1,2,3,4}, plus two controls on stem 0:
- **Shuffle control (j=1):** training labels permuted. Calibrates the
  pipeline's empirical max-over-epochs noise ceiling.
- **Positive control (j=0):** must clear the power gate below.

We report **max test advantage over all epoch evaluations** — deliberately
anti-conservative (gives the attacker its best epoch), which strengthens a
null result but inflates noise; the null gate accounts for this.

## Pre-registered gates

- **Power gate (positive control):** j=0 net advantage ≥ 0.20. If it
  fails, the experiment is unpowered (architecture/optimization too weak
  to learn even one round) and NO conclusion is drawn at j≥1; escalate
  depth/width before interpreting.
- **Null gate:** test advantage σ = 1/√(2^19) ≈ 0.0014; max over ~50
  epoch-evals inflates the expected null max to ≈2.6σ. A j≥1 cell is a
  **candidate signal** only if max-advantage > 0.007 (5σ). Anything below
  is consistent with zero.
- **Escalation rule:** any candidate signal at j≥1 → rerun that cell with
  a fresh seed and 4× training data (2^22). Confirmed only if it
  reproduces above the gate. Only confirmed signals count as refuting the
  cliff.
- **Shuffle-control sanity:** its max-advantage must be ≤ the null gate;
  if it isn't, the noise model is wrong and gates are recalibrated before
  any interpretation.

## Amendment 1 (before any j>=1 interpretation)

The smoke run's positive control FAILED (j=0 net advantage 0.008 vs the
hand score's 0.499): predicting **bit 31** from raw bits requires SGD to
learn 32-bit modular arithmetic from scratch — a parity-like optimization
barrier that is not the question under test. Per the pre-registered power
gate, no j>=1 conclusion was drawn. Redesign, documented before re-running:

- **Target:** regress the top byte of `state[0]` at depth j (normalized,
  MSE loss), and score with the PAPER'S OWN metric: select the 1/256 test
  fraction with smallest prediction; advantage = 1 −
  mean_topbyte(selected)/mean_topbyte(all). Directly comparable to the
  anchor sweep's 0.886-at-j=0 row.
- **Inputs:** per word, 32 bits PLUS two value features (word/2^32 and
  topbyte/256). Attacker-generous: removes the "learn powers of two"
  obstacle without leaking anything not cheaply computable.
- **Recalibrated gates** (selection metric, n_test=2^19, frac 1/256 → 2048
  selected, advantage SE ≈ 0.013): power gate j=0 ≥ 0.5; null gate at
  j>=1: max-over-epochs advantage > 0.065 (5σ) = candidate signal.
  Shuffle control must stay ≤ 0.065.
- Escalation rule unchanged.

The smoke-run j=1 nulls under the old design are NOT counted as evidence
(unpowered). Cross-check: the score baseline under the new metric must
reproduce ≈0.886 at j=0, validating the pipeline against the paper.

## Amendment 2 (before any j>=1 interpretation)

Amendment 1's positive control still FAILED (j=0 net 0.013 vs hand score
0.8896 under the new metric — which also validated the pipeline against
the paper's 0.886). Diagnosis: the net was being asked to learn the
GF(2)-linear and bit-local round operations (Sigma0/Sigma1 as
value-of-XOR-of-rotations, Ch, Maj) from raw bits by SGD. Those are
exactly the operations the paper classifies as DEPTH-0 (carry-free). The
faithful operationalization of the paper's carry/local attack class is to
hand the attacker every depth-0-computable quantity for free and test how
far SGD sees past the CARRY layers — which are the claimed wall.

- **Inputs (final):** per word, 32 bits + value/2^32 + topbyte/256, for:
  the 8 read-point state words, the schedule words W[r_read..r_read+j],
  AND the four carry-free derived words at the read point: Sigma0(a),
  Sigma1(e), Ch(e,f,g), Maj(a,b,c). With these, the j=0 target is "top
  byte of a sum of given operands" = exactly one carry layer. For j>=1
  the operands of later rounds are NOT depth-0 computable (they require
  resolving the previous round's carries) — by design: that IS the test.
- Gates, metric, grid, escalation: unchanged from Amendment 1.

This makes the j-axis a clean measurement of "SGD reach past j carry
rounds, given all carry-free structure for free."

## Amendment 3 (before any j>=1 interpretation)

Amendment 2's positive control still failed (j=0 max 0.064; training loss
plateaus at the target's variance, i.e. nothing learned). Root cause
identified: the target is MODULAR — topbyte = floor(256·frac(Σ values)) —
and modular reduction has zero linear correlation with any input
direction, the known hard case for MSE+SGD (the "grokking modular
arithmetic" obstacle). This is, notably, the paper's own thesis rendered
as an optimization pathology; but an experiment whose attacker cannot
represent the obvious computable baseline is unpowered, so we equip it
with the standard remedies from the neural-arithmetic literature:

- **Fourier features:** per word, add sin/cos(2π·2^m·value) for
  m = 0..7 (16 features/word). Angle addition makes modular sums
  expressible as products of given rotations.
- **Head:** 256-way softmax classification over the top byte,
  cross-entropy loss (handles the wrap; regression does not). Selection
  score = expected top byte under the softmax.
- Inputs otherwise per Amendment 2; gates, metric, grid unchanged.

If the positive control STILL fails after this, the experiment is
declared unpowered and reported as such — no j>=1 conclusion either way.

## Amendment 4 (final design; before any j>=1 interpretation)

Amendments 1-3 progressively diagnosed the positive-control failure down
to a known-open problem: SGD does not synthesize exact 32-bit modular
addition at feasible budgets (diagnostics: MSE plateaus at target
variance; gated multiplicative nets MEMORIZE the training set — train CE
0.09, test advantage 0 — and weight decay + 2^20 samples does not flip
them to generalization within our budget). That is the "grokking modular
arithmetic" problem, orthogonal to the cliff question — and its
difficulty is itself an observation in the paper's favor, reported as a
finding. Gohr's nets never synthesized cipher arithmetic; they exploited
statistical signal in data they were given.

**Final design:** add the read round's computed T1 and T2 as feature
words (the attacker computes round r_read in full — this is exactly the
anchor sweep's read point, whose hand score IS a function of T1, T2).
Features: state(8) + derived carry-free(4) + T1,T2(2) + schedule words.

- j=0 ("one adder layer") is now the anchor sweep's own j=0: signal is
  trivially present (the hand score is in the feature span), so the
  power gate tests the PIPELINE (training, data, metric), as it should.
- j>=1 is the genuine reach question: does ANY learned function of
  everything-computed-through-round-r_read predict the target one or
  more rounds deeper? The cliff claim says no.
- Gates, metric, grid, escalation unchanged. Architecture as registered
  (GELU residual MLP, Adam); the gated/weight-decay variants from the
  diagnostics are not used in the final run.

## Corrections (round-5 external review; text above left as registered)

1. The Model section says "~2.4M parameters". The actual w1024/b3 model
   is **7.4M** parameters (verified by direct count). Wrong arithmetic in
   the registration, not a design change; gates did not depend on it.
2. The Model section justifies dropping convolution with "no conv since
   our input has no pair structure". That justification is wrong: Gohr's
   convolution runs over BIT POSITIONS (words as channels), an axis our
   input has identically, and it is the natural inductive bias for
   carry-chain (translation-equivariant) structure. A conv-architecture
   cell at j=1 is run as a review follow-up (`review_fixes.py`).
3. The registered per-cell gate (0.065 on the max-over-epochs statistic)
   only excludes per-cell true advantages of roughly >=0.04-0.05. The
   pooled final-epoch CI reported in RESULTS.md is the quantitatively
   meaningful exclusion; the gate governs only the escalation decision.

## Interpretation (committed in advance)

- All j≥1 cells null + power gate passed → the learned-feature search
  reproduces the cliff; report as direct evidence behind §6's "broader
  sweep" and the ML/AlphaGo objection.
- Confirmed j≥1 signal → the cliff claim as stated is too strong; the
  paper's §5.5/§6 must be revised to the measured reach. (Note: even
  then, a net evaluated per candidate costs far more than the hash it
  predicts, so no mining advantage follows; the *scientific* claim about
  reach is what's at stake.)
