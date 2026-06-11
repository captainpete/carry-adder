"""0160 fork-1: exact carry-add depth accounting for SHA-256d.

Pure static dataflow analysis. NO SHA evaluation, NO data, NO GPU — this
propagates an integer "adder depth" (AD) through the SHA-256d dependency
graph to compute, exactly, how many modular-addition layers separate the
controllable block-1 message inputs (the miner-controlled header words:
nonce, extranonce-driven Merkle root, version, timestamp) from a
mining-target output bit.

Conventions:
  - AD(value) = number of modular-add operations on the dependency chain
    from a leaf (input message word / initial state word) to that value,
    using the MINIMUM-depth associative tree (Huffman-combine) for
    multi-operand sums. Minimum depth is the conservative-for-security
    choice: it credits the attacker with the shallowest legal circuit, so
    the figure is convention-dependent and a sequential or nonce-only
    accounting gives a somewhat larger number.
  - GF(2)-linear ops (rotations, shifts, XOR, Sigma/sigma) and bit-local
    nonlinear ops (Ch, Maj) add 0 adder-depth (they inject no carry).
  - Only modular '+' adds depth (each is one carry-injection layer); the
    Davies-Meyer feed-forward (wordwise add mod 2^32) is one such layer.
  - All 16 block-1 message words are treated as depth-0 source inputs.
    Block 0 of hash1 (the midstate) is fixed for a given header prefix,
    so its initial-state AD = 0; same for the hash2 IV.

Also reports the total modular-add COUNT (and carry-bit count) in the
SHA-256d computation as a secondary structural statistic.
"""

from __future__ import annotations

import heapq


def combine_depth(depths: list[int]) -> int:
    """Minimum adder depth to sum the given terms (Huffman-combine).
    Each pairwise add = one carry layer: depth = max(x, y) + 1."""
    if not depths:
        return 0
    if len(depths) == 1:
        return depths[0]
    h = list(depths)
    heapq.heapify(h)
    while len(h) > 1:
        x = heapq.heappop(h)
        y = heapq.heappop(h)
        heapq.heappush(h, max(x, y) + 1)
    return h[0]


# count of pairwise adds to sum n terms = n - 1
def combine_count(n_terms: int) -> int:
    return max(0, n_terms - 1)


def run_compression(msg_ad: list[int], init_state_ad: list[int]):
    """Propagate adder-depth through one 64-round SHA-256 compression
    plus its feed-forward.

    msg_ad: AD of the 16 input message words W[0..15].
    init_state_ad: AD of the 8 initial state words (a..h).
    Returns (output_ad (8,), total_add_count).
    """
    # --- message schedule expansion W[16..63] ---
    W = list(msg_ad)  # W[0..15]
    add_count = 0
    for t in range(16, 64):
        # W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
        terms = [W[t - 2], W[t - 7], W[t - 15], W[t - 16]]
        W.append(combine_depth(terms))
        add_count += combine_count(4)  # 3 adds per expanded word

    a, b, c, d, e, f, g, h = init_state_ad
    for t in range(64):
        # T1 = h + Sigma1(e) + Ch(e,f,g) + K[t] + W[t]
        #   Sigma1(e) -> AD e ; Ch(e,f,g) -> AD max(e,f,g) ; K[t] const -> 0
        ad_T1 = combine_depth([h, e, max(e, f, g), 0, W[t]])
        add_count += combine_count(5)
        # T2 = Sigma0(a) + Maj(a,b,c)
        ad_T2 = combine_depth([a, max(a, b, c)])
        add_count += combine_count(2)
        # a' = T1 + T2 ; e' = d + T1
        a_new = combine_depth([ad_T1, ad_T2]); add_count += 1
        e_new = combine_depth([d, ad_T1]); add_count += 1
        # rotate
        h = g; g = f; f = e; e = e_new
        d = c; c = b; b = a; a = a_new
    state_after = [a, b, c, d, e, f, g, h]
    # feed-forward: out_i = state_after_i + init_state_i
    out = [combine_depth([state_after[i], init_state_ad[i]]) for i in range(8)]
    add_count += 8
    return out, add_count


def main():
    print("0160 fork-1: exact carry-add depth accounting for SHA-256d")
    print("=" * 64)

    # ---- hash1, block 1 ----
    # 16 block-1 message words, all treated as depth-0 source inputs (the
    # miner controls them via nonce W[3], extranonce-driven Merkle root,
    # version, and timestamp). Depth is measured from input arrival.
    msg1_ad = [0] * 16
    # initial state = block-0 midstate: fixed for a header prefix -> AD 0
    init1_ad = [0] * 8
    out1, count1 = run_compression(msg1_ad, init1_ad)
    print("\n[hash1 block1]  digest word ADs:", out1)
    print(f"[hash1 block1]  max output adder-depth: {max(out1)}")
    print(f"[hash1 block1]  modular-add count: {count1}")

    # ---- hash2 ----
    # input message: W[0..7] = hash1 digest (AD = out1), W[8] = 0x80000000
    # const, W[9..14] = 0, W[15] = length const. Constants -> AD 0.
    msg2_ad = list(out1) + [0] * 8
    init2_ad = [0] * 8  # IV constant
    out2, count2 = run_compression(msg2_ad, init2_ad)
    print("\n[hash2]  digest word ADs:", out2)
    print(f"[hash2]  max output adder-depth: {max(out2)}")
    print(f"[hash2]  modular-add count: {count2}")

    total_depth = max(out2)
    total_count = count1 + count2
    print("\n" + "=" * 64)
    print(f"SHA-256d input->output adder depth (min-tree):  {total_depth}")
    print(f"SHA-256d total modular-add count:               {total_count}")
    print(f"  carry bits (~31 per 32-bit add):              ~{total_count * 31}")

    # ---- per-stage breakdown (single-compression contributions) ----
    print("\nper-stage breakdown:")
    # depth added by one compression's 64 rounds (from AD-0 inputs):
    out_test, _ = run_compression([0] * 16, [0] * 8)
    print(f"  one compression (64 rounds + feed-forward), inputs at 0: "
          f"max output AD = {max(out_test)}")
    # incremental depth from hash1-output feeding hash2:
    print(f"  hash1 contributes depth {max(out1)} into hash2's inputs;")
    print(f"  hash2 lifts that to {max(out2)} "
          f"(+{max(out2) - max(out1)} adder-layers across the 2nd hash).")

    # Note: we deliberately do NOT compound a per-layer retention rho
    # across this depth to manufacture an advantage exponent. The measured
    # rho (anchor_sweep.py) is a detection limit consistent with 0, not a
    # decay rate, so such a product would be false precision. Depth is the
    # structural coordinate; the empirical content is the single-round
    # collapse measured in anchor_sweep.py, which needs no extrapolation.


if __name__ == "__main__":
    main()
