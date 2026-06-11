"""Anchor sweep: the carry-aware advantage cliffs within one SHA-256 round.

Self-contained reproduction. Confirms the fork-2 result: the carry-aware
score at an interior round strongly predicts the top byte of state[0] one
layer downstream (advantage ~0.886), and that advantage collapses to the
noise floor within a single round (~3 adder layers) — i.e. per-layer
retention rho ~ 0.

Setup (faithful to the mechanism, simplified for portability): standard
SHA-256 message schedule + round function, with one message word (W[3])
playing the role of the varied nonce and the others fixed per "stem". The
cliff is a property of the SHA-256 round function, not of the Bitcoin
double-hash wrapper, so this bare-round demonstration reproduces it.

Two backends:
  * default  — numpy (CPU) or cupy (GPU); the full paper-scale run
                (24 stems, 2^22 candidates) wants a GPU.
  * --pure   — pure Python, standard library only, no third-party deps.
                A small smoke run (default 2^13 candidates, 3 stems) that
                reproduces the cliff and self-checks against hashlib. Use
                this to verify the headline empirical claim anywhere a
                Python interpreter runs. Auto-selected if numpy is absent.

Expected output (either backend): advantage(j=0) ~ 0.88; advantage(j>=1)
consistent with 0 (|mean| within a few times the noise floor).
"""

from __future__ import annotations

import hashlib
import sys

try:
    import numpy as np

    from sha256_ref import (
        xp, to_np, K, IV, expand_schedule, round_with_score, apply_round,
    )
    _HAVE_NUMPY = True
except Exception:  # numpy/sha256_ref unavailable: only the --pure path runs
    _HAVE_NUMPY = False


# =====================================================================
# numpy / cupy backend (paper-scale)
# =====================================================================

def stem_words(stem_seed):
    """16 message words for a 'stem': deterministic pseudo-random, with W[3]
    reserved for the varied nonce (set per-candidate below)."""
    h = hashlib.sha256(f"stem-{stem_seed}".encode()).digest()  # 32 bytes = 8 words
    base = np.frombuffer(h, dtype=">u4").astype(np.uint32)
    w = np.zeros(16, dtype=np.uint32)
    w[:8] = base
    w[8:16] = base ^ np.uint32(0x9e3779b9)  # fill remaining words deterministically
    return w


def run_stem(stem_seed, N, r_read, J, frac=1.0 / 256.0):
    w16 = stem_words(stem_seed)
    W16 = xp.broadcast_to(xp.asarray(w16)[None, :], (N, 16)).copy()
    W16[:, 3] = xp.arange(N, dtype=xp.uint32)           # the "nonce" word
    W = expand_schedule(W16)                            # (N,64)

    state = xp.broadcast_to(IV[None, :], (N, 8)).copy()
    for r in range(r_read):
        state = apply_round(state, int(_K_at(r)), W[:, r])

    _, score = round_with_score(state, int(_K_at(r_read)), W[:, r_read])
    score_cpu = to_np(score)
    k_sel = max(1, int(N * frac))
    sel = xp.asarray(np.argpartition(score_cpu, k_sel)[:k_sel])

    out = []
    for j in range(J + 1):
        r = r_read + j
        state = apply_round(state, int(_K_at(r)), W[:, r])
        top = (state[:, 0] >> 24).astype(xp.float64)
        mean_all = float(top.mean())
        mean_sel = float(top[sel].mean())
        adv = 1.0 - mean_sel / mean_all if mean_all > 0 else float("nan")
        out.append(adv)
    return out


def _K_at(r):
    """Round constant K[r] as a Python int, backend-independent."""
    return int(to_np(K[r]))


def main_numpy():
    # Defaults match the paper's run (24 prefixes, 2^22 candidates each).
    # This wants a GPU; on the NumPy/CPU backend, reduce N and stems for a
    # quick smoke test (the cliff is visible at much smaller sizes).
    N = 1 << 22          # 4.2M candidates ("nonces") per stem
    r_read = 30          # interior read round
    J = 8                # downstream rounds
    stems = list(range(24))

    print(f"anchor sweep (repro): N={N}, r_read={r_read}, J={J}, "
          f"stems={len(stems)}")
    print("metric: 1 - mean_topbyte(state[0] | best 1/256 by carry score) "
          "/ mean_topbyte(all), j rounds downstream\n")

    per_j = {j: [] for j in range(J + 1)}
    for s in stems:
        adv = run_stem(s, N, r_read, J)
        for j, a in enumerate(adv):
            per_j[j].append(a)
        print(f"  stem {s}: adv(j=0..4) = " + ", ".join(f"{adv[k]:+.4f}" for k in range(5)))

    print(f"\n{'j (rounds deep)':>15} {'mean advantage':>16}")
    means = []
    for j in range(J + 1):
        m = float(np.mean(per_j[j]))
        means.append(m)
        print(f"{j:>15} {m:>+16.5f}")

    floor = np.std([per_j[j] for j in range(J - 2, J + 1)])
    print(f"\n  depth-1 advantage (j=0): {means[0]:+.4f}")
    print(f"  deeper advantage (mean j>=1): {np.mean(means[1:]):+.5f}  "
          f"(noise scale ~{floor/np.sqrt(len(stems)):.4f})")
    if means[0] > 0.5 and abs(np.mean(means[1:])) < 5 * floor / np.sqrt(len(stems)):
        print("\n  RESULT: cliff confirmed — strong advantage at depth-1, "
              "consistent with zero one round deeper (rho ~ 0).")
    else:
        print("\n  RESULT: unexpected; inspect (more stems / larger N may be needed).")


# =====================================================================
# pure-Python backend (standard library only, no numpy/cupy)
# =====================================================================
#
# This path mirrors sha256_ref.round_with_score / expand_schedule bit for
# bit, in plain Python with 32-bit masking, so the cliff (and a hashlib
# parity check) can be reproduced with nothing but a Python interpreter.

_M = 0xFFFFFFFF

# FIPS 180-4 constants (same values as sha256_ref; inlined so this path has
# zero third-party imports).
_K_PURE = [
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
_IV_PURE = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def _ror(x, n):
    return ((x >> n) | (x << (32 - n))) & _M


def _expand_pure(w16, upto):
    """w16: list of 16 ints -> schedule words W[0..upto] (modular-add recurrence)."""
    W = list(w16) + [0] * (upto + 1 - 16)
    for t in range(16, upto + 1):
        w15 = W[t - 15]; w2 = W[t - 2]
        s0 = _ror(w15, 7) ^ _ror(w15, 18) ^ (w15 >> 3)
        s1 = _ror(w2, 17) ^ _ror(w2, 19) ^ (w2 >> 10)
        W[t] = (W[t - 16] + s0 + W[t - 7] + s1) & _M
    return W


def _round_cols_pure(cols, k, wlist, want_score=False):
    """One SHA-256 round over N candidates held as 8 column lists (a..h).
    Mirrors sha256_ref.round_with_score exactly; score = (T1>>24)+(T2>>24)
    from the PRE-round state. Returns new cols (and score list if asked)."""
    a, b, c, d, e, f, g, h = cols
    N = len(a)
    na = [0] * N; ne = [0] * N
    sc = [0] * N if want_score else None
    for i in range(N):
        ai = a[i]; ei = e[i]
        s0 = _ror(ai, 2) ^ _ror(ai, 13) ^ _ror(ai, 22)
        s1 = _ror(ei, 6) ^ _ror(ei, 11) ^ _ror(ei, 25)
        ch = (ei & f[i]) ^ ((~ei & _M) & g[i])
        maj = (ai & b[i]) ^ (ai & c[i]) ^ (b[i] & c[i])
        t1 = (h[i] + s1 + ch + k + wlist[i]) & _M
        t2 = (s0 + maj) & _M
        if want_score:
            sc[i] = (t1 >> 24) + (t2 >> 24)
        na[i] = (t1 + t2) & _M
        ne[i] = (d[i] + t1) & _M
    new = (na, a, b, c, ne, e, f, g)
    return (new, sc) if want_score else new


def _stem_words_pure(stem_seed):
    h = hashlib.sha256(f"stem-{stem_seed}".encode()).digest()
    base = [int.from_bytes(h[4 * i:4 * i + 4], "big") for i in range(8)]
    return base + [(x ^ 0x9e3779b9) & _M for x in base]   # 16 words


def _compress_block_pure(H, w16):
    """One SHA-256 compression for a single message (N=1), for parity."""
    W = _expand_pure(w16, 63)
    cols = tuple([H[i]] for i in range(8))
    for t in range(64):
        cols = _round_cols_pure(cols, _K_PURE[t], [W[t]])
    return [(cols[i][0] + H[i]) & _M for i in range(8)]


def _parity_check_pure():
    msg = bytearray(b"abc")
    msg.append(0x80)
    msg.extend(b"\x00" * (64 - len(msg) - 8))
    msg.extend((24).to_bytes(8, "big"))
    w16 = [int.from_bytes(bytes(msg)[4 * i:4 * i + 4], "big") for i in range(16)]
    out = _compress_block_pure(_IV_PURE, w16)
    got = b"".join(x.to_bytes(4, "big") for x in out)
    want = hashlib.sha256(b"abc").digest()
    ok = got == want
    print(f"  parity: SHA-256('abc') pure == hashlib : "
          f"{'PASS' if ok else 'FAIL'}  [pure Python]")
    assert ok, "pure SHA-256 does not match hashlib"


def _mean(xs):
    return sum(xs) / len(xs)


def _pstdev(xs):
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def run_stem_pure(stem_seed, N, r_read, J, frac=1.0 / 256.0):
    w16 = _stem_words_pure(stem_seed)
    upto = r_read + J + 1
    Wall = []
    for n in range(N):
        w = list(w16); w[3] = n & _M       # the "nonce" word
        Wall.append(_expand_pure(w, upto))

    cols = tuple([iv] * N for iv in _IV_PURE)
    for r in range(r_read):
        cols = _round_cols_pure(cols, _K_PURE[r], [Wall[i][r] for i in range(N)])

    # score from the pre-r_read state (new state discarded), then read the
    # SAME state forward one round at a time — j=0 applies round r_read.
    _, score = _round_cols_pure(
        cols, _K_PURE[r_read], [Wall[i][r_read] for i in range(N)], want_score=True)
    k_sel = max(1, int(N * frac))
    sel = sorted(range(N), key=lambda i: score[i])[:k_sel]

    out = []
    for j in range(J + 1):
        r = r_read + j
        cols = _round_cols_pure(cols, _K_PURE[r], [Wall[i][r] for i in range(N)])
        a = cols[0]
        top = [a[i] >> 24 for i in range(N)]
        mean_all = _mean(top)
        mean_sel = _mean([top[i] for i in sel])
        adv = 1.0 - mean_sel / mean_all if mean_all > 0 else float("nan")
        out.append(adv)
    return out


def main_pure(N=1 << 13, stems=3, r_read=30, J=3):
    print(f"anchor sweep (pure-Python smoke): N={N}, r_read={r_read}, J={J}, "
          f"stems={stems}")
    print("metric: 1 - mean_topbyte(state[0] | best 1/256 by carry score) "
          "/ mean_topbyte(all), j rounds downstream")
    _parity_check_pure()
    print()

    per_j = {j: [] for j in range(J + 1)}
    for s in range(stems):
        adv = run_stem_pure(s, N, r_read, J)
        for j, a in enumerate(adv):
            per_j[j].append(a)
        print(f"  stem {s}: adv(j=0..{J}) = " + ", ".join(f"{a:+.4f}" for a in adv))

    print(f"\n{'j (rounds deep)':>15} {'mean advantage':>16}")
    means = []
    for j in range(J + 1):
        m = _mean(per_j[j])
        means.append(m)
        print(f"{j:>15} {m:>+16.5f}")

    # Heuristic noise scale only (1/sqrt(N)); the true per-stem SE of the
    # selected-subset mean is roughly an order of magnitude larger because
    # the selection keeps only N/256 candidates, so individual stem values
    # at j>=1 scatter well beyond this figure without indicating signal.
    floor = 1.0 / (N ** 0.5)
    deeper = _mean(means[1:])
    print(f"\n  depth-1 advantage (j=0): {means[0]:+.4f}")
    print(f"  deeper advantage (mean j>=1): {deeper:+.5f}  "
          f"(heuristic noise scale ~{floor:.4f}; per-stem scatter is larger)")
    if means[0] > 0.5 and abs(deeper) < 20 * floor:
        print("\n  RESULT: cliff confirmed — strong advantage at depth-1, "
              "consistent with zero one round deeper (rho ~ 0).")
        print("  (Small smoke run; the off-cliff residual is larger here than "
              "the paper's 2^22 run purely from the higher noise floor.)")
    else:
        print("\n  RESULT: unexpected; inspect (more stems / larger N may help).")


def main():
    force_pure = "--pure" in sys.argv[1:]
    if force_pure or not _HAVE_NUMPY:
        if not _HAVE_NUMPY and not force_pure:
            print("[numpy not found — running the pure-Python smoke path]\n")
        main_pure()
    else:
        main_numpy()


if __name__ == "__main__":
    main()
