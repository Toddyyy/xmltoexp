from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "vienna_atepp_k331_tempo_compare" / "k331_score_aligned_mean_tempo.csv"
OUT_DIR = ROOT / "MERIX SUBMISSION" / "MIREX_Model" / "vienna_atepp_k331_tempo_compare"

# In the K331 theme score, these two 48-eighth-beat spans are note/onset identical:
# measures 19-26 and 29-36.
FRAGMENTS = {
    "m19-26": (108, 156),
    "m29-36": (168, 216),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    rows = []
    for fragment_name, (start, end) in FRAGMENTS.items():
        local = np.arange(end - start, dtype=np.int32)
        chunk = df.iloc[start:end].copy().reset_index(drop=True)
        for dataset, col in [
            ("Vienna", "vienna_mean_bpm"),
            ("ATEPP", "atepp_score_aligned_mean_bpm"),
        ]:
            for i, value in enumerate(chunk[col].to_numpy(dtype=np.float32)):
                rows.append(
                    {
                        "dataset": dataset,
                        "fragment": fragment_name,
                        "local_beat_idx": int(local[i]),
                        "global_beat_idx": int(start + i),
                        "tempo_bpm": float(value),
                    }
                )
    aligned = pd.DataFrame(rows)
    aligned.to_csv(OUT_DIR / "k331_identical_score_fragment_aligned_tempo.csv", index=False)

    stats = []
    for dataset in ["Vienna", "ATEPP"]:
        wide = aligned[aligned["dataset"] == dataset].pivot(
            index="local_beat_idx",
            columns="fragment",
            values="tempo_bpm",
        )
        a = wide["m19-26"].to_numpy(dtype=np.float32)
        b = wide["m29-36"].to_numpy(dtype=np.float32)
        stats.append(
            {
                "dataset": dataset,
                "fragment_a": "m19-26",
                "fragment_b": "m29-36",
                "beats": int(len(a)),
                "corr": float(np.corrcoef(a, b)[0, 1]),
                "mae_bpm": float(np.mean(np.abs(a - b))),
                "mean_a_bpm": float(np.mean(a)),
                "mean_b_bpm": float(np.mean(b)),
            }
        )
    pd.DataFrame(stats).to_csv(OUT_DIR / "k331_identical_score_fragment_aligned_stats.csv", index=False)

    colors = {"Vienna": "#1f4e8c", "ATEPP": "#c45a00"}
    styles = {"m19-26": "-", "m29-36": "--"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, dataset in zip(axes, ["Vienna", "ATEPP"]):
        sub = aligned[aligned["dataset"] == dataset]
        for fragment_name in FRAGMENTS:
            g = sub[sub["fragment"] == fragment_name].sort_values("local_beat_idx")
            ax.plot(
                g["local_beat_idx"],
                g["tempo_bpm"],
                color=colors[dataset],
                linestyle=styles[fragment_name],
                linewidth=2.2,
                label=f"{dataset} {fragment_name}",
            )
        ax.set_ylabel("Tempo BPM")
        ax.set_title(f"{dataset}: identical-note score fragments aligned on local beat axis")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Local beat index inside identical 48-beat score fragment")
    fig.tight_layout()
    fig_path = OUT_DIR / "k331_identical_score_fragment_aligned_tempo.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    fig2, ax = plt.subplots(1, 1, figsize=(12, 4.6))
    for dataset in ["Vienna", "ATEPP"]:
        sub = aligned[aligned["dataset"] == dataset]
        mean = sub.groupby("local_beat_idx", sort=True)["tempo_bpm"].mean()
        ax.plot(mean.index.to_numpy(), mean.to_numpy(), color=colors[dataset], linewidth=2.4, label=f"{dataset} mean of two identical fragments")
    ax.set_title("K331 identical score fragments: aligned tempo means")
    ax.set_xlabel("Local beat index inside identical 48-beat score fragment")
    ax.set_ylabel("Tempo BPM")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig2.tight_layout()
    fig2_path = OUT_DIR / "k331_identical_score_fragment_aligned_dataset_means.png"
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)

    print(fig_path)
    print(fig2_path)
    print(pd.DataFrame(stats).to_string(index=False))


if __name__ == "__main__":
    main()
