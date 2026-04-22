#!/usr/bin/env python3
"""Block diagram of the v5 (MAN-Xfmr) architecture.

Two panels:
  LEFT  — top-level architecture: inputs -> PairBlock x2 -> head -> output
  RIGHT — zoomed detail of one PairBlock: embedding -> Q/K -> score
          -> multiplicative gating -> lambda -> pair MLP -> aggregation

Output: docs/plots/v5_arch.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/home/ayf7/trading/docs/plots/v5_arch.png")

# ---------- color palette ----------
C_INPUT   = "#fde2d4"   # warm peach for raw inputs
C_EMBED   = "#dfe7fd"   # soft blue for embedding ops
C_PROJ    = "#cfe1ff"   # team-conditioned projections
C_SCORE   = "#e2f0cb"   # score / pre-activation
C_GATE    = "#ffe4b5"   # gating ops (multiplicative)
C_LAMBDA  = "#ffd8a8"   # the lambda tensor (highlight color)
C_VALUE   = "#d4e9d4"   # value vector path
C_POOL    = "#cfe7d8"   # pooling
C_HEAD    = "#e7d4f3"   # head MLPs
C_OUT     = "#f7c8c8"   # final output
C_NOTE    = "#f0f0f0"   # annotations
C_PAIRBLK = "#ffd8a8"   # the PairBlock placeholder in top-level


def add_box(ax, x, y, w, h, text, *,
            facecolor=C_EMBED, edgecolor="#333", lw=1.2,
            fontsize=10, weight="normal",
            boxstyle="round,pad=0.05,rounding_size=0.10"):
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                         linewidth=lw, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, linespacing=1.4)
    return (x, y, w, h)


def add_arrow(ax, src, dst, *, label=None, color="#222", lw=1.5,
              label_offset=(0.0, 0.0), label_fontsize=8.5,
              connectionstyle="arc3,rad=0.0", label_color="#444",
              label_bg=True):
    arr = FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=14,
                          color=color, lw=lw,
                          connectionstyle=connectionstyle,
                          shrinkA=2, shrinkB=4)
    ax.add_patch(arr)
    if label:
        mx = (src[0] + dst[0]) / 2 + label_offset[0]
        my = (src[1] + dst[1]) / 2 + label_offset[1]
        bg = dict(boxstyle="round,pad=0.20", facecolor="white",
                  edgecolor="none", alpha=0.92) if label_bg else None
        ax.text(mx, my, label, fontsize=label_fontsize, color=label_color,
                ha="center", va="center", style="italic", bbox=bg)


def top_mid(b):    x, y, w, h = b; return (x + w / 2, y + h)
def bot_mid(b):    x, y, w, h = b; return (x + w / 2, y)
def left_mid(b):   x, y, w, h = b; return (x, y + h / 2)
def right_mid(b):  x, y, w, h = b; return (x + w, y + h / 2)


# ============================================================
# LEFT panel — top-level architecture
# ============================================================
def draw_top_level(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5.0, 17.6, "Top-level architecture",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(5.0, 17.1,
            "(per game)",
            ha="center", va="center", fontsize=9.5, style="italic", color="#555")

    # ---------- inputs ----------
    inp = add_box(
        ax, 0.5, 0.5, 9.0, 1.4,
        "Game inputs   (B = batch dim)\n"
        r"home & away rosters: $\mathrm{idx},\,m,\,\pi,\,\mathbf{s}$ for each player slot"
        "\n"
        r"team ids $t^{\mathrm{home}},t^{\mathrm{away}}$  ;  rest days $r^{\mathrm{home}},r^{\mathrm{away}}$  ;"
        r"  tabular features $\mathbf{x}^{\mathrm{tab}}\in\mathbb{R}^{T},\;T=41$",
        facecolor=C_INPUT, fontsize=9.5)

    # ---------- two PairBlocks ----------
    pb1 = add_box(
        ax, 0.4, 3.3, 4.4, 2.6,
        "Pairwise Prevalence Block\n"
        r"(off=$\mathrm{home}$, def=$\mathrm{away}$)" "\n\n"
        r"$\to\;\mathbf{z}^{\mathrm{home}}\in\mathbb{R}^d$",
        facecolor=C_PAIRBLK, fontsize=10.5, weight="bold",
        edgecolor="#a3590b", lw=1.8)
    pb2 = add_box(
        ax, 5.2, 3.3, 4.4, 2.6,
        "Pairwise Prevalence Block\n"
        r"(off=$\mathrm{away}$, def=$\mathrm{home}$)" "\n\n"
        r"$\to\;\mathbf{z}^{\mathrm{away}}\in\mathbb{R}^d$",
        facecolor=C_PAIRBLK, fontsize=10.5, weight="bold",
        edgecolor="#a3590b", lw=1.8)
    ax.text(5.0, 6.10,
            "shared parameters (right panel)",
            ha="center", va="center", fontsize=8.5,
            style="italic", color="#555")

    add_arrow(ax, (2.5, 1.9), bot_mid(pb1),
              label=r"home roster + $t^{\mathrm{home}}$,$t^{\mathrm{away}}$",
              label_offset=(-1.6, 0.1), label_fontsize=8)
    add_arrow(ax, (7.5, 1.9), bot_mid(pb2),
              label=r"away roster + $t^{\mathrm{away}}$,$t^{\mathrm{home}}$",
              label_offset=(1.6, 0.1), label_fontsize=8)

    # ---------- concat ----------
    concat = add_box(
        ax, 1.0, 7.2, 8.0, 1.0,
        r"concat:  $[\mathbf{z}^{\mathrm{home}}\,\Vert\,\mathbf{z}^{\mathrm{away}}\,\Vert\,r^{\mathrm{home}}\,\Vert\,r^{\mathrm{away}}]\;\in\mathbb{R}^{2d+2}$",
        facecolor=C_POOL, fontsize=10)
    add_arrow(ax, top_mid(pb1), (3.5, 7.2),
              label=r"$\mathbf{z}^{\mathrm{home}}$",
              label_offset=(-0.4, 0), label_fontsize=9)
    add_arrow(ax, top_mid(pb2), (6.5, 7.2),
              label=r"$\mathbf{z}^{\mathrm{away}}$",
              label_offset=(0.4, 0), label_fontsize=9)

    # ---------- two-stream heads ----------
    head_lin = add_box(
        ax, 0.4, 9.2, 4.4, 1.4,
        "Lineup head (two-stream)\n"
        r"$\ell^{\mathrm{lineup}} = \mathrm{MLP}_{\mathrm{lineup}}(\cdot)$",
        facecolor=C_HEAD, fontsize=10)
    head_tab = add_box(
        ax, 5.2, 9.2, 4.4, 1.4,
        "Tabular head\n"
        r"$\ell^{\mathrm{tab}} = \mathrm{MLP}_{\mathrm{tab}}(\mathbf{x}^{\mathrm{tab}})$",
        facecolor=C_HEAD, fontsize=10)

    add_arrow(ax, top_mid(concat), bot_mid(head_lin),
              label="lineup vec", label_offset=(-0.6, 0.0))
    # tabular feeds head_tab from inputs (long arrow)
    add_arrow(ax, (8.5, 1.9), bot_mid(head_tab),
              label=r"$\mathbf{x}^{\mathrm{tab}}$",
              connectionstyle="arc3,rad=0.35",
              label_offset=(2.2, -3.0), label_fontsize=9)

    # ---------- sum ----------
    sumbox = add_box(
        ax, 2.5, 11.5, 5.0, 0.9,
        r"$\ell = \ell^{\mathrm{lineup}} + \ell^{\mathrm{tab}}$",
        facecolor="#fff5cc", fontsize=11.5, weight="bold")
    add_arrow(ax, top_mid(head_lin), (3.5, 11.5),
              label=r"$\ell^{\mathrm{lineup}}$", label_offset=(-0.6, 0),
              label_fontsize=9.5)
    add_arrow(ax, top_mid(head_tab), (6.5, 11.5),
              label=r"$\ell^{\mathrm{tab}}$", label_offset=(0.6, 0),
              label_fontsize=9.5)

    # ---------- output ----------
    out_box = add_box(
        ax, 1.8, 13.1, 6.4, 1.0,
        r"$\hat p\;=\;\mathbb{P}[\text{home wins}]\;=\;\sigma(\ell)\in[0,1]$",
        facecolor=C_OUT, fontsize=12, weight="bold",
        edgecolor="#9a2222", lw=1.8)
    add_arrow(ax, top_mid(sumbox), bot_mid(out_box),
              label=r"$\sigma(\cdot)$", label_offset=(0.4, 0),
              label_fontsize=10)

    # ---------- training-time loss footer (small caption above output) ----------
    ax.text(
        5.0, 14.6,
        r"Training loss:  $\mathcal{L} = \mathcal{L}_{\mathrm{BCE}}(\hat p,\,y)"
        r" + \alpha\,\mathcal{L}_{\mathrm{Poisson}}(\lambda,\,e),\;\;\alpha=10$",
        ha="center", va="center", fontsize=9.5, color="#444", style="italic")
    ax.text(
        5.0, 15.05,
        r"$\mathcal{L}_{\mathrm{Poisson}}$ gathers $\lambda_{i_n j_n}$ for each (game, side, $i$, $j$)"
        r" supervision row vs. actual exposure_possessions $e_n$",
        ha="center", va="center", fontsize=8.5, color="#666", style="italic")


# ============================================================
# RIGHT panel — PairBlock detail
# ============================================================
def draw_pairblock_detail(ax):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(7.0, 17.6, "Pairwise Prevalence Block (detail)",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(7.0, 17.1,
            r"executed once per direction; same parameters in both calls",
            ha="center", va="center", fontsize=9.5, style="italic", color="#555")

    # ---------- ROW 1: inputs ----------
    inp_off = add_box(
        ax, 0.3, 0.3, 6.4, 1.2,
        "OFFENSIVE side inputs\n"
        r"$\mathrm{idx}^{\mathrm{off}}\in\{0..V\}^{L_o}$,  "
        r"$m^{\mathrm{off}}\in\{0,1\}^{L_o}$,  "
        r"$\pi^{\mathrm{off}}\in[0,1]^{L_o}$,  "
        r"$\mathbf{s}^{\mathrm{off}}\in\mathbb{R}^{L_o\times S}$,  "
        r"$t^{\mathrm{off}}$",
        facecolor=C_INPUT, fontsize=8.5)
    inp_def = add_box(
        ax, 7.3, 0.3, 6.4, 1.2,
        "DEFENSIVE side inputs\n"
        r"$\mathrm{idx}^{\mathrm{def}}\in\{0..V\}^{L_d}$,  "
        r"$m^{\mathrm{def}}\in\{0,1\}^{L_d}$,  "
        r"$\pi^{\mathrm{def}}\in[0,1]^{L_d}$,  "
        r"$\mathbf{s}^{\mathrm{def}}\in\mathbb{R}^{L_d\times S}$,  "
        r"$t^{\mathrm{def}}$",
        facecolor=C_INPUT, fontsize=8.5)

    # ---------- ROW 2: embedding ----------
    emb_off = add_box(
        ax, 0.5, 2.2, 6.0, 1.3,
        "Player embedding + stats residual\n"
        r"$\mathbf{e}^{\mathrm{off}}_i = E^{\mathrm{player}}[\mathrm{idx}^{\mathrm{off}}_i]"
        r" + P^{\mathrm{stats}}\,\mathbf{s}^{\mathrm{off}}_i\;\in\mathbb{R}^d$",
        facecolor=C_EMBED, fontsize=9.5)
    emb_def = add_box(
        ax, 7.5, 2.2, 6.0, 1.3,
        "Player embedding + stats residual\n"
        r"$\mathbf{e}^{\mathrm{def}}_j = E^{\mathrm{player}}[\mathrm{idx}^{\mathrm{def}}_j]"
        r" + P^{\mathrm{stats}}\,\mathbf{s}^{\mathrm{def}}_j\;\in\mathbb{R}^d$",
        facecolor=C_EMBED, fontsize=9.5)
    add_arrow(ax, top_mid(inp_off), bot_mid(emb_off),
              label=r"$(B,L_o)+(B,L_o,S)$", label_offset=(1.4, 0))
    add_arrow(ax, top_mid(inp_def), bot_mid(emb_def),
              label=r"$(B,L_d)+(B,L_d,S)$", label_offset=(1.4, 0))

    # ---------- ROW 3: Q/K projection (and v_pair branches off here) ----------
    qproj = add_box(
        ax, 0.5, 4.2, 5.0, 1.5,
        "Team-cond. Q projection\n"
        r"$\mathbf{W}^Q=\mathrm{reshape}(W^Q_{\mathrm{team}}[t^{\mathrm{off}}],d\times d)$"
        "\n"
        r"$\mathbf{q}_i = \mathbf{W}^Q\,\mathbf{e}^{\mathrm{off}}_i$",
        facecolor=C_PROJ, fontsize=8.8)
    kproj = add_box(
        ax, 8.5, 4.2, 5.0, 1.5,
        "Team-cond. K projection\n"
        r"$\mathbf{W}^K=\mathrm{reshape}(W^K_{\mathrm{team}}[t^{\mathrm{def}}],d\times d)$"
        "\n"
        r"$\mathbf{k}_j = \mathbf{W}^K\,\mathbf{e}^{\mathrm{def}}_j$",
        facecolor=C_PROJ, fontsize=8.8)
    vpair = add_box(
        ax, 5.7, 4.2, 2.6, 1.5,
        r"$\mathrm{MLP}_{\mathrm{pair}}$" "\n"
        r"$\mathbf{v}_{ij} =$" "\n"
        r"$\mathrm{MLP}([\mathbf{e}^{\mathrm{off}}_i\Vert\mathbf{e}^{\mathrm{def}}_j])$",
        facecolor=C_VALUE, fontsize=8.5)

    add_arrow(ax, top_mid(emb_off), bot_mid(qproj),
              label=r"$\mathbf{e}^{\mathrm{off}}$", label_offset=(0.45, 0))
    add_arrow(ax, top_mid(emb_def), bot_mid(kproj),
              label=r"$\mathbf{e}^{\mathrm{def}}$", label_offset=(0.45, 0))
    # short side arrows from embedding rows into v_pair
    add_arrow(ax, right_mid(emb_off), (5.7, 4.6),
              label=None, lw=1.2, color="#666",
              connectionstyle="arc3,rad=-0.15")
    add_arrow(ax, left_mid(emb_def), (8.3, 4.6),
              label=None, lw=1.2, color="#666",
              connectionstyle="arc3,rad=0.15")

    # ---------- ROW 4: SCORE ----------
    score = add_box(
        ax, 2.0, 6.4, 10.0, 1.7,
        "Pre-activation score\n"
        r"$s_{ij} = \dfrac{\mathbf{q}_i^{\top}\mathbf{k}_j}{\sqrt{d}}"
        r" + b^{\mathrm{off}}[\mathrm{idx}^{\mathrm{off}}_i]"
        r" + b^{\mathrm{def}}[\mathrm{idx}^{\mathrm{def}}_j]$" "\n"
        r"$\tilde s_{ij} = \mathrm{clip}(s_{ij},-12,+12)\;\in\mathbb{R}^{B\times L_o\times L_d}$",
        facecolor=C_SCORE, fontsize=9.5)
    add_arrow(ax, top_mid(qproj), (4.0, 6.4),
              label=r"$\mathbf{q}$", label_offset=(-0.4, 0.05))
    add_arrow(ax, top_mid(kproj), (10.0, 6.4),
              label=r"$\mathbf{k}$", label_offset=(0.4, 0.05))

    # ---------- ROW 5: Π and M outer products (sit on either side of lambda) ----------
    pi_outer = add_box(
        ax, 0.3, 8.6, 3.0, 1.5,
        "Soft availability gate\n"
        r"$\Pi_{ij} = \pi^{\mathrm{off}}_i \cdot \pi^{\mathrm{def}}_j$" "\n"
        r"(outer prod., $[0,1]^{L_o\times L_d}$)",
        facecolor=C_GATE, fontsize=8.5)
    m_outer = add_box(
        ax, 10.7, 8.6, 3.0, 1.5,
        "Hard padding gate\n"
        r"$M_{ij} = m^{\mathrm{off}}_i \cdot m^{\mathrm{def}}_j$" "\n"
        r"(outer prod., $\{0,1\}^{L_o\times L_d}$)",
        facecolor=C_GATE, fontsize=8.5)

    # short curved arrows from inputs into the gates
    add_arrow(ax, (2.0, 1.5), bot_mid(pi_outer),
              label=r"$\pi^{\mathrm{off}}$",
              connectionstyle="arc3,rad=-0.50",
              label_offset=(-1.7, 1.5), label_fontsize=8)
    add_arrow(ax, (8.0, 1.5), bot_mid(pi_outer),
              label=r"$\pi^{\mathrm{def}}$",
              connectionstyle="arc3,rad=0.50",
              label_offset=(-3.5, 3.0), label_fontsize=8)
    add_arrow(ax, (5.5, 1.5), bot_mid(m_outer),
              label=r"$m^{\mathrm{off}}$",
              connectionstyle="arc3,rad=-0.50",
              label_offset=(3.5, 3.0), label_fontsize=8)
    add_arrow(ax, (12.0, 1.5), bot_mid(m_outer),
              label=r"$m^{\mathrm{def}}$",
              connectionstyle="arc3,rad=0.50",
              label_offset=(1.7, 1.5), label_fontsize=8)

    # ---------- ROW 6: lambda ----------
    lam = add_box(
        ax, 4.0, 8.8, 6.0, 1.5,
        r"Pairwise prevalence  $\lambda$" "\n"
        r"$\lambda_{ij} = \exp(\tilde s_{ij}) \;\odot\; \Pi_{ij} \;\odot\; M_{ij}$" "\n"
        r"non-negative, no softmax  ;  $\lambda\in\mathbb{R}_{\geq 0}^{B\times L_o\times L_d}$",
        facecolor=C_LAMBDA, fontsize=9.5, weight="bold",
        edgecolor="#a3590b", lw=2.0)
    add_arrow(ax, top_mid(score), bot_mid(lam),
              label=r"$\exp(\tilde s)$", label_offset=(0.9, 0))
    add_arrow(ax, right_mid(pi_outer), left_mid(lam),
              label=r"$\Pi$", label_offset=(0.0, 0.20))
    add_arrow(ax, left_mid(m_outer), right_mid(lam),
              label=r"$M$", label_offset=(0.0, 0.20))

    # ---------- ROW 7: aggregation ----------
    agg = add_box(
        ax, 3.5, 11.3, 7.0, 1.7,
        r"$\lambda$-weighted pooling" "\n"
        r"$\mathbf{z} = \dfrac{\sum_{i,j}\lambda_{ij}\,\mathbf{v}_{ij}}{\sum_{i,j}\lambda_{ij}+\varepsilon}\;\in\mathbb{R}^d$",
        facecolor=C_POOL, fontsize=10.5, weight="bold",
        edgecolor="#235c2c", lw=1.6)
    add_arrow(ax, top_mid(lam), (6.0, 11.3),
              label=r"$\lambda$", label_offset=(-0.4, 0))
    # v_pair to agg: long arrow up from row 3 (vpair box) along the right
    add_arrow(ax, top_mid(vpair), (8.0, 11.3),
              label=r"$\mathbf{v}\;(B,L_o,L_d,d)$",
              connectionstyle="arc3,rad=0.50",
              label_offset=(3.5, -2.0), label_fontsize=8.5)

    # ---------- output of the block ----------
    out_block = add_box(
        ax, 4.5, 13.6, 5.0, 1.0,
        r"output:  $\mathbf{z}\;\in\mathbb{R}^d$"
        "\n"
        r"(per-direction team vector)",
        facecolor=C_OUT, fontsize=10, weight="bold",
        edgecolor="#9a2222", lw=1.5)
    add_arrow(ax, top_mid(agg), bot_mid(out_block))

    # ---------- footer note ----------
    add_box(
        ax, 0.3, 15.5, 13.4, 1.2,
        r"$d=32$,  $L_o,L_d\approx 13$ (padded roster slots),  "
        r"$S=10$ stats per player,  $V=$ player vocab size,  $K=$ team vocab size,  "
        r"$\varepsilon=10^{-6}$." "\n"
        r"All learnable params: $E^{\mathrm{player}},\,P^{\mathrm{stats}},\,"
        r"W^Q_{\mathrm{team}},\,W^K_{\mathrm{team}},\,b^{\mathrm{off}},\,b^{\mathrm{def}},\,"
        r"\mathrm{MLP}_{\mathrm{pair}}$ (shared across both directions).",
        facecolor="#fafafa", fontsize=9, edgecolor="#888", lw=0.7)


# ============================================================
def main():
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.04)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    draw_top_level(ax_left)
    draw_pairblock_detail(ax_right)

    fig.suptitle("v5 (MAN-Xfmr): pairwise prevalence with multiplicative gating",
                 fontsize=15, weight="bold", y=0.995)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
