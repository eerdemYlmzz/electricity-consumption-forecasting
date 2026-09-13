"""
Phase 4 plot: Optuna vs Random Search -- performance (best-so-far val
RMSE across trials) and per-trial training time, the "Optuna-vs-
alternatif karsilastirmasi" plot from the advisor's list.
"""
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = "results/phase4_hpo_results.csv"
PLOT_PATH = "plots/phase4_optuna_vs_random.png"


def plot_comparison():
    df = pd.read_csv(RESULTS_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in ["optuna", "random"]:
        sub = df[df["method"] == method].sort_values("trial")
        best_so_far = sub["val_rmse"].cummin()
        axes[0].plot(sub["trial"] + 1, best_so_far, marker="o", label=method)
        axes[1].scatter(sub["trial"] + 1, sub["train_seconds"], label=method, alpha=0.7)

    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Best validation RMSE so far (MW)")
    axes[0].set_title("Convergence: Optuna vs Random Search")
    axes[0].legend()

    axes[1].set_xlabel("Trial number")
    axes[1].set_ylabel("Training time per trial (s)")
    axes[1].set_title("Per-trial cost")
    axes[1].legend()

    fig.suptitle("Phase 4 -- HPO method comparison, BiGRU@72h, k=4 features")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    plot_comparison()
