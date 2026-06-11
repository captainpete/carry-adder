"""Minimal self-contained SHA-256 for the carry-depth reproduction.

Vendored so the reproduction has no cross-project dependencies. Implements
the SHA-256 message schedule and round function exactly per FIPS 180-4,
plus the carry-aware score used in the anchor sweep, and a hashlib parity
check (run this file directly to verify).

Backend: uses CuPy if a GPU is present, otherwise NumPy. The parity check
and small runs therefore work on CPU; the large anchor sweep wants a GPU.
The feed-forward is wordwise modular addition (s + H), matching SHA-256.
"""

from __future__ import annotations

import hashlib
import numpy as np

try:
    import cupy as xp
    _GPU = True
except Exception:  # no GPU / no cupy: fall back to NumPy
    import numpy as xp
    _GPU = False


def to_np(a):
    """Bring an xp array to NumPy regardless of backend."""
    return xp.asnumpy(a) if _GPU else np.asarray(a)


# FIPS 180-4 round constants
_K = [
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
K = xp.asarray(_K, dtype=xp.uint32)

_IV = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]
IV = xp.asarray(_IV, dtype=xp.uint32)


def ror32(x, n):
    return ((x >> n) | (x << (32 - n))).astype(xp.uint32)


def expand_schedule(W16):
    """W16: (N,16) uint32 -> (N,64) SHA-256 message schedule.
    The schedule combines words by modular addition, so it is a fixed
    (not GF(2)-linear) recurrence."""
    N = W16.shape[0]
    W = xp.zeros((N, 64), dtype=xp.uint32)
    W[:, :16] = W16
    for t in range(16, 64):
        w15 = W[:, t - 15]; w2 = W[:, t - 2]
        s0 = ror32(w15, 7) ^ ror32(w15, 18) ^ (w15 >> 3)
        s1 = ror32(w2, 17) ^ ror32(w2, 19) ^ (w2 >> 10)
        W[:, t] = (W[:, t - 16] + s0 + W[:, t - 7] + s1).astype(xp.uint32)
    return W


def round_with_score(state, k_val, w_val):
    """Apply one SHA-256 round to (N,8) state (columns a..h); also return the
    carry-aware score (T1>>24)+(T2>>24) from the PRE-round state.
    Returns (new_state, score)."""
    a, b, c, d = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    e, f, g, h = state[:, 4], state[:, 5], state[:, 6], state[:, 7]
    s0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22)
    s1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25)
    ch = (e & f) ^ ((~e) & g)
    maj = (a & b) ^ (a & c) ^ (b & c)
    t1 = (h + s1 + ch + xp.uint32(k_val) + w_val).astype(xp.uint32)
    t2 = (s0 + maj).astype(xp.uint32)
    score = ((t1 >> 24) + (t2 >> 24)).astype(xp.uint32)
    a_new = (t1 + t2).astype(xp.uint32)
    e_new = (d + t1).astype(xp.uint32)
    new_state = xp.stack([a_new, a, b, c, e_new, e, f, g], axis=1)
    return new_state, score


def apply_round(state, k_val, w_val):
    return round_with_score(state, k_val, w_val)[0]


def compress_block(H, W16):
    """One SHA-256 compression: H (N,8) initial state, W16 (N,16) message.
    Returns (N,8) block output H + state-after-64-rounds. The feed-forward
    is wordwise modular addition, the same operation as the round adds."""
    W = expand_schedule(W16)
    s = H
    for t in range(64):
        s = apply_round(s, int(_K[t]), W[:, t])
    return (s + H).astype(xp.uint32)


def _parity_check():
    """Verify the vendored SHA-256 matches hashlib on the FIPS test vector.
    Runs on CPU (NumPy backend) when no GPU is present."""
    msg = bytearray(b"abc")
    msg.append(0x80)
    msg.extend(b"\x00" * (64 - len(msg) - 8))
    msg.extend((24).to_bytes(8, "big"))  # bit length = 24
    assert len(msg) == 64
    W16 = xp.asarray(np.frombuffer(bytes(msg), dtype=">u4").astype(np.uint32))[None, :]
    out = to_np(compress_block(IV[None, :], W16))[0]
    got = b"".join(int(x).to_bytes(4, "big") for x in out)
    want = hashlib.sha256(b"abc").digest()
    ok = got == want
    backend = "cupy/GPU" if _GPU else "numpy/CPU"
    print(f"SHA-256('abc') vendored == hashlib : {'PASS' if ok else 'FAIL'}"
          f"   [{backend}]")
    print(f"  vendored: {got.hex()}")
    print(f"  hashlib : {want.hex()}")
    assert ok, "vendored SHA-256 does not match hashlib"


if __name__ == "__main__":
    _parity_check()
