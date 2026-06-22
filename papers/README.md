# The Carry-Adder Wall — paper series

A three-paper series on whether SHA-256d (Bitcoin) mining can beat brute
force. Each paper carries one standalone finding; they cross-cite as a
series. Full rationale and migration plan: `../../PAPER_SERIES_PLAN.md`.

Build a paper:
```
cd papers/0X-.../ && pdflatex paper.tex && pdflatex paper.tex
```
Shared LaTeX preamble (packages + macros) is in `../../shared/preamble.tex`,
`\input` by each paper. The vendored SHA-256 reference is `../../shared/sha256_ref.py`.

## The three papers

| # | Dir | Standalone finding | Status |
|---|-----|--------------------|--------|
| I | `01-amortization` | Every mining win is confined to three bounded sharing regions + a 2^D search floor; the residual reduces to a SHA-256 distinguisher. Stem selection is exchangeable. **Mining can't beat brute force unless SHA-256 is broken.** | **drafted** (builds, 8pp) |
| II | `02-carry-depth` | Carry depth is an exact implementation-invariant coordinate (SHA-256d = 386); the strongest local carry-aware advantage cliffs within one round; Holte's amazing matrix explains why (gap 1/2, base-set); SAT wall at round 11 is the algorithmic companion. | **drafted** (builds, 7pp) |
| III | `03-neural` | A Gohr-style learned distinguisher beats the hand score at one layer, dies one round deeper, expires at k=3->4; precision-independent (float64); the learner's reach, not intrinsic (k-invariant gap); feature isolation a second failure mode. | **drafted** (builds, 5pp) |

## Citation policy: backward-only

Each paper is standalone and may cite **earlier** parts in the sequence,
**never later** ones. Reading order is I → II → III, so:
- **Paper I** is fully self-contained. It poses the residual (does the round
  function admit a local shortcut?) as the open question and does **not**
  forward-reference; the reduction needs nothing from II or III.
- **Paper II** cites I (the localized residual it answers).
- **Paper III** cites I and II.

This lets the series be read and released in sequence, each part referring
back but never across or forward.

## House style

Every paper gets an editing pass against the AI-writing tells catalogued at
`tropes.fyi/directory`. In practice: em-dashes held to a handful (appositives
go in parentheses, asides become colons or separate sentences), no
negative-parallelism reframes ("X, not Y"), no rhetorical tricolons, no
"serves as" / signposted "in conclusion". Substantive technical enumerations
(the three compressions, the three sharing regions) are kept.

## Migration map (monolith → series)

Source: `../paper/carry_depth.tex` (the original monolith, retained until the
series is complete, then retired). Per-section targets are in
`PAPER_SERIES_PLAN.md`; per-paper migration notes are in the `% MIGRATION`
comment block at the top of each `paper.tex`.

Code homes:
- Paper II: `../repro/` (carry_depth.py, anchor_sweep.py), bs2 `0190_sat_sweep`.
- Paper III: `../neural/` (gohr_sweep.py + the probe scripts, RESULTS.md).
