"""Figure generation for RESULTS.md: safety-usefulness frontier, transfer matrix.

Reads runs/cell_summary.csv (produced by run_matrix.py analyze) and writes PNGs into
runs/figs/. Kept separate from metric computation so figures can be regenerated at will
without recomputing anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path(__file__).parent.parent / "runs"
FIGS = RUNS / "figs"


def _short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-Instruct-AWQ", "")


def _attack_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Cells whose attack statistics are meaningful: phase_class 'main' (E2 honest+attack
    merged) carries the core signal; 'probe' cells have attack data but no same-cell
    honest baseline. Falls back to the older 'mode' label ('mixed'/'attack') if
    phase_class is absent, so the figure survives schema evolution."""
    if "phase_class" in df.columns:
        return df[(df["phase_class"].isin(["main", "probe", "selective"]))
                  & df["auroc"].notna()]
    return df[(df["mode"].isin(["attack", "mixed"])) & df["auroc"].notna()]


def fig_safety_usefulness(df: pd.DataFrame) -> Path:
    """Attack cells: x = usefulness (honest success), y = safety at 1% audit budget."""
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "safety_usefulness.png"
    atk = _attack_cells(df)
    hon = df[df["mode"] == "honest"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (untr, mon), g in atk.groupby(["untrusted_model", "monitor"]):
        for _, r in g.iterrows():
            color = "tab:red" if r["affordances"] == "FULL" else "tab:blue"
            ax.scatter(r["usefulness"], r["safety_1pct"], color=color, s=90, zorder=3)
            ax.annotate(f"{_short(untr)}\n× {_short(mon)}\n{r['affordances']}",
                        (r["usefulness"], r["safety_1pct"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    # Honest usefulness references (vertical guides).
    for _, r in hon.drop_duplicates("untrusted_model").iterrows():
        ax.axvline(r["usefulness"], color="gray", alpha=0.25, lw=1, ls="--")
    ax.set_xlabel("usefulness (honest main-task success)")
    ax.set_ylabel("safety at 1% audit budget")
    ax.set_title("Safety-usefulness by (untrusted × monitor × affordances)\n"
                 "red = FULL affordances, blue = RESTRICTED; dashed = honest usefulness")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_transfer_matrix(df: pd.DataFrame) -> Path | None:
    """Monitor-family × policy-family AUROC heatmap (attack cells, FULL affordances)."""
    FIGS.mkdir(exist_ok=True)
    atk = _attack_cells(df)
    atk = atk[atk["affordances"] == "FULL"]
    if atk.empty:
        return None
    piv = atk.pivot_table(index="monitor_family", columns="untrusted_family", values="auroc")
    out = FIGS / "transfer_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(piv.columns)), [_short(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [_short(i) for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xlabel("untrusted policy family")
    ax.set_ylabel("monitor family")
    ax.set_title("Monitor AUROC on attacks, FULL affordances\n(cross-lineage transfer)")
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_affordance_effect(df: pd.DataFrame) -> Path | None:
    """Paired FULL vs RESTRICTED comparison per (untrusted, monitor): attack success."""
    FIGS.mkdir(exist_ok=True)
    atk = _attack_cells(df)
    if atk.empty:
        return None
    piv = atk.pivot_table(index=["untrusted_model", "monitor"],
                          columns="affordances", values="attack_success_rate")
    if not {"FULL", "RESTRICTED"}.issubset(piv.columns):
        return None
    piv = piv.dropna()
    if piv.empty:
        return None
    out = FIGS / "affordance_effect.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(piv))
    ax.bar([i - 0.18 for i in x], piv["FULL"], width=0.36, label="FULL", color="tab:red")
    ax.bar([i + 0.18 for i in x], piv["RESTRICTED"], width=0.36, label="RESTRICTED",
           color="tab:blue")
    ax.set_xticks(list(x), [f"{_short(u)}\n× {_short(m)}" for (u, m) in piv.index],
                  fontsize=8)
    ax.set_ylabel("attack success rate")
    ax.set_title("Attack success by affordance configuration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_all() -> list[Path]:
    csv = RUNS / "cell_summary.csv"
    if not csv.exists():
        print("[figs] no cell_summary.csv yet")
        return []
    df = pd.read_csv(csv)
    made = []
    for fn in (fig_safety_usefulness, fig_transfer_matrix, fig_affordance_effect):
        p = fn(df)
        if p:
            made.append(p)
            print(f"[figs] wrote {p.name}")
    return made


if __name__ == "__main__":
    make_all()
