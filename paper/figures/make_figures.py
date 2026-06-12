"""Generate the paper's figures as vector PDFs.

Reads no experiment data files: the depth profile is recomputed via
repro/carry_depth.py's combine_depth (mirroring run_compression with a
per-round trace), and the measured values are the published numbers from
ANCHOR_SWEEP / neural/results.json / neural/review_fixes_results.json,
inlined below with their sources.

Run with any Python that has matplotlib:
    python make_figures.py
Outputs fig_depth_profile.pdf, fig_anchor_cliff.pdf, fig_neural.pdf
into this directory.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "repro"))
from carry_depth import combine_depth  # noqa: E402

# ---- shared style -----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})
ACCENT = "#2b5d8a"
GRAY = "0.45"


# ---- figure 1: depth profile ------------------------------------------
def traced_compression(msg_ad, init_ad):
    """Mirror carry_depth.run_compression, recording max state depth
    after each round and after the feed-forward."""
    W = list(msg_ad)
    for t in range(16, 64):
        W.append(combine_depth([W[t - 2], W[t - 7], W[t - 15], W[t - 16]]))
    a, b, c, d, e, f, g, h = init_ad
    trace = []
    for t in range(64):
        ad_T1 = combine_depth([h, e, max(e, f, g), 0, W[t]])
        ad_T2 = combine_depth([a, max(a, b, c)])
        a_new = combine_depth([ad_T1, ad_T2])
        e_new = combine_depth([d, ad_T1])
        h = g; g = f; f = e; e = e_new
        d = c; c = b; b = a; a = a_new
        trace.append(max(a, b, c, d, e, f, g, h))
    state = [a, b, c, d, e, f, g, h]
    out = [combine_depth([state[i], init_ad[i]]) for i in range(8)]
    trace.append(max(out))  # feed-forward
    return out, trace


def fig_depth_profile():
    out1, trace1 = traced_compression([0] * 16, [0] * 8)
    out2, trace2 = traced_compression(list(out1) + [0] * 8, [0] * 8)
    profile = [0] + trace1 + trace2
    x = np.arange(len(profile))
    assert profile[65] == 194 and profile[130] == 386, profile[-1]

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(x, profile, color=ACCENT, lw=1.4)
    ax.axhline(194, color=GRAY, lw=0.6, ls=(0, (4, 3)))
    ax.axhline(386, color=GRAY, lw=0.6, ls=(0, (4, 3)))
    ax.axvline(65, color=GRAY, lw=0.6, ls=(0, (1, 2)))

    ax.annotate("194", xy=(0, 194), xytext=(-2, 194), ha="right",
                va="center", fontsize=8, color="0.25")
    ax.annotate("386", xy=(0, 386), xytext=(-2, 386), ha="right",
                va="center", fontsize=8, color="0.25")
    ax.annotate("feed-forward,\nsecond hash begins", xy=(65, 200),
                xytext=(72, 255), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
    ax.annotate("$\\approx$3 layers per round", xy=(32, 96),
                xytext=(38, 55), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
    ax.annotate("2-round regime:\nsolvable algebra", xy=(2, 8),
                xytext=(6, 130), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))

    ax.set_xlim(0, 130)
    ax.set_ylim(0, 410)
    ax.set_xticks([0, 16, 32, 48, 65, 81, 97, 113, 130])
    ax.set_xticklabels(["0", "16", "32", "48", "64+ff",
                        "16", "32", "48", "64+ff"])
    ax.set_xlabel("round (first compression, then second hash)")
    ax.set_ylabel("carry depth (adder layers)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_depth_profile.pdf")
    plt.close(fig)


# ---- figure 2: anchor-sweep cliff --------------------------------------
def fig_anchor_cliff():
    # ANCHOR_SWEEP_RESULT.md: 24 stems x 2^22 candidates, read point r=30.
    j = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    adv = np.array([0.8856, 0.00101, -0.00079, -0.00034, 0.00112,
                    -0.0006, -0.0006, -0.0006, -0.0006])
    jj = np.linspace(0, 8, 200)
    heur = 0.886 * 2.0 ** (-3 * jj)

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.axhline(0, color="0.75", lw=0.6, ls=(0, (4, 3)))
    ax.plot(jj, heur, color=GRAY, lw=1.0, ls=(0, (5, 3)),
            label="geometric heuristic (§5.3)")
    ax.plot(j, adv, color=ACCENT, lw=1.3, marker="o", ms=3.5,
            label="measured")
    ax.annotate("$0.8856 \\pm 0.0005$", xy=(0, 0.8856), xytext=(0.35, 0.84),
                fontsize=8, color="0.25")
    ax.annotate("$\\leq 0.0011$, consistent with 0", xy=(1, 0.001),
                xytext=(1.6, 0.10), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=GRAY))
    ax.set_xlabel("rounds past the read point ($\\approx$3 adder layers each)")
    ax.set_ylabel("retained advantage")
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(-0.06, 0.95)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_anchor_cliff.pdf")
    plt.close(fig)


# ---- figure 3: learned attacker ----------------------------------------
def fig_neural():
    # (a) neural/results.json, means over 3 stems, max-over-epochs.
    j = np.arange(5)
    net = np.array([0.9978, 0.0291, 0.0260, 0.0304, 0.0309])
    hand = np.array([0.8841, 0.0127, 0.0050, 0.0122, 0.0083])
    GATE, SHUFFLE = 0.065, 0.0212
    # (b) neural/review_fixes_results.json ladder, max-over-epochs.
    k = np.array([2, 3, 4, 5, 6, 7])
    ladder = np.array([0.9996, 0.9983, 0.0407, 0.0287, 0.0403, 0.0251])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

    ax1.axhline(GATE, color=GRAY, lw=0.7, ls=(0, (5, 3)))
    ax1.text(4.1, GATE * 1.15, "signal gate 0.065", fontsize=7.5,
             color="0.25", ha="right")
    ax1.axhline(SHUFFLE, color=GRAY, lw=0.7, ls=(0, (1, 2)))
    ax1.text(4.1, SHUFFLE * 0.72, "shuffle control 0.021", fontsize=7.5,
             color="0.25", ha="right")
    ax1.plot(j, hand, color=GRAY, lw=1.0, marker="s", ms=3.2,
             label="hand score")
    ax1.plot(j, net, color=ACCENT, lw=1.3, marker="o", ms=3.5,
             label="network, best epoch")
    ax1.set_yscale("log")
    ax1.set_ylim(0.003, 1.6)
    ax1.set_xticks(j)
    ax1.set_xlabel("prediction depth $j$ (rounds)")
    ax1.set_ylabel("max selection advantage")
    ax1.legend(loc="upper right")
    ax1.set_title("(a) distinguisher grid", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    colors = [ACCENT if v > 0.5 else "0.65" for v in ladder]
    ax2.bar(k, ladder, width=0.62, color=colors)
    ax2.axvline(3.5, color=GRAY, lw=0.7, ls=(0, (4, 3)))
    ax2.text(3.62, 0.78, "learning stops", fontsize=7.5, color="0.25")
    for ki, v in zip(k, ladder):
        ax2.text(ki, v + 0.025, f"{v:.3f}" if v < 0.5 else f"{v:.4f}",
                 ha="center", fontsize=6.5, color="0.25")
    ax2.set_xlabel("operands $k$ ($k{-}1$ chained additions)")
    ax2.set_ylabel("max selection advantage")
    ax2.set_ylim(0, 1.12)
    ax2.set_title("(b) $k$-operand ladder", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(HERE / "fig_neural.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_depth_profile()
    fig_anchor_cliff()
    fig_neural()
    print("wrote",
          ", ".join(p.name for p in sorted(HERE.glob("fig_*.pdf"))))
