"""Plot double-descent sweep results.

Reads summary.json from each run_v2_dd_{xs,s,m,l,xl} run dir and produces:
  1. val_bce vs param-count (capacity axis)
  2. test_bce vs param-count (capacity axis)
  3. test_acc vs param-count (capacity axis)
  4. val_bce vs epoch (one line per size, epoch-wise descent)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RUNS = [
    ("xs",  "d=32, n=1/1"),
    ("s",   "d=64, n=2/2"),
    ("m",   "d=128, n=4/4"),
    ("l",   "d=192, n=6/6"),
    ("xl",  "d=256, n=8/8"),
    ("x2l", "d=384, n=12/12"),
    ("x3l", "d=512, n=16/16"),
]


def load(name: str) -> dict:
    p = ARTIFACTS / f"run_v2_dd_{name}" / "summary.json"
    with p.open() as fh:
        return json.load(fh)


def count_params(name: str) -> int:
    log = Path(f"/tmp/run_v2_dd_{name}.log").read_text()
    for ln in log.splitlines():
        if "[model] params=" in ln:
            tok = ln.split("[model] params=")[1].split()[0].rstrip(",")
            return int(tok.replace(",", ""))
    return -1


def main() -> None:
    data = [(name, label, load(name), count_params(name)) for name, label in RUNS]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_vbce_p, ax_tbce_p, ax_tacc_p, ax_vbce_ep = axes.ravel()

    params = [d[3] for d in data]
    val_bces = [d[2]["best_val_bce"] for d in data]
    test_bces = [d[2]["test_best"]["bce"] for d in data]
    test_accs = [d[2]["test_best"]["acc"] for d in data]
    labels = [f"{d[0]}\n{d[1]}" for d in data]

    for ax, ys, ylabel, title in [
        (ax_vbce_p, val_bces, "best val BCE",  "Val BCE vs params"),
        (ax_tbce_p, test_bces, "test BCE",      "Test BCE vs params"),
        (ax_tacc_p, test_accs, "test acc",      "Test acc vs params"),
    ]:
        ax.plot(params, ys, "o-", lw=2, ms=8)
        for x, y, lbl in zip(params, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("params (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    cmap = plt.get_cmap("viridis")
    for i, (name, label, summ, _) in enumerate(data):
        eps = [h["epoch"] for h in summ["history"]]
        vbces = [h["val_bce"] for h in summ["history"]]
        ax_vbce_ep.plot(eps, vbces, lw=1.2, color=cmap(i / max(1, len(data) - 1)),
                        label=f"{name} ({label})")
    ax_vbce_ep.set_xlabel("epoch")
    ax_vbce_ep.set_ylabel("val BCE")
    ax_vbce_ep.set_title("Val BCE vs epoch (epoch-wise descent)")
    ax_vbce_ep.set_ylim(0.55, 1.5)
    ax_vbce_ep.axhline(0.625, color="red", lw=0.8, ls="--", alpha=0.6, label="sharp market ~0.625")
    ax_vbce_ep.legend(fontsize=8, loc="upper right")
    ax_vbce_ep.grid(True, alpha=0.3)

    fig.suptitle("CME-v2 double-descent sweep · direct win head · BCE only · 200 ep")
    fig.tight_layout()
    out = ARTIFACTS / "dd_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")

    print(f"\n{'name':<5} {'params':>12}  {'val_bce':>8}  {'test_bce':>8}  {'test_acc':>8}  best_ep")
    for (name, _, summ, p) in data:
        print(f"{name:<5} {p:>12,}  {summ['best_val_bce']:>8.4f}  "
              f"{summ['test_best']['bce']:>8.4f}  {summ['test_best']['acc']:>8.3f}  "
              f"{summ['best_epoch']:>3}")


if __name__ == "__main__":
    main()
