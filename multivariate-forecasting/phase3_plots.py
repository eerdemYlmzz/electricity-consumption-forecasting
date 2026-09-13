"""
Phase 3 plot: top-k feature count vs performance (RMSE and R^2), the
"top-k performansı" plot from the advisor's list. Includes k=15 (all
features, from Phase 1) as the reference point top-k is judged against.
"""
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = "results/phase3_topk_results.csv"
PLOT_PATH = "plots/phase3_topk_performance.png"


def plot_topk_performance():
    df = pd.read_csv(RESULTS_PATH).sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["k"], df["rmse"], marker="o")
    axes[0].set_xlabel("k (number of top-ranked features)")
    axes[0].set_ylabel("Test RMSE (MW)")
    axes[0].set_title("RMSE vs k")

    axes[1].plot(df["k"], df["r2"], marker="o")
    axes[1].set_xlabel("k (number of top-ranked features)")
    axes[1].set_ylabel("Test R^2")
    axes[1].set_title("R^2 vs k")

    for ax in axes:
        for _, row in df.iterrows():
            ax.axvline(row["k"], color="lightgray", linewidth=0.5, zorder=0)

    fig.suptitle("Phase 3 -- top-k SHAP-ranked features, BiGRU@72h (k=15 = all features, from Phase 1)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    plot_topk_performance()
