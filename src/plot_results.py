"""Generate report-ready figures from evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.summarize_results import summarize_baseline_results


def create_baseline_figure(
    results_dir: str | Path = "results",
    models_dir: str | Path = "results/milestone_models",
    output_path: str | Path = "figures/milestone_baseline_results.png",
) -> Path:
    """Create a two-panel baseline reward and goal-completion figure."""
    summary = summarize_baseline_results(results_dir, models_dir)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    colors = ["#2a9d8f" if outcome == "Learned" else "#e76f51"
              for outcome in summary["outcome"]]

    axes[0].bar(summary["seed"].astype(str), summary["mean_reward"], color=colors)
    axes[0].set_title("Mean evaluation reward")
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(0, max(11, summary["mean_reward"].max() + 1))
    axes[0].grid(axis="y", alpha=0.25)

    goal_percent = summary["goal_completion_rate"] * 100
    axes[1].bar(summary["seed"].astype(str), goal_percent, color=colors)
    axes[1].set_title("Goal completion rate")
    axes[1].set_xlabel("Random seed")
    axes[1].set_ylabel("Completed episodes (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2a9d8f", label="Learned"),
        plt.Rectangle((0, 0), 1, 1, color="#e76f51", label="Did not learn"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.suptitle("Baseline DQN evaluation across random seeds")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="results/milestone_models")
    parser.add_argument(
        "--output",
        default="figures/milestone_baseline_results.png",
    )
    args = parser.parse_args()
    print(create_baseline_figure(args.results_dir, args.models_dir, args.output))


if __name__ == "__main__":
    main()
