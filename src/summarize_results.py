"""Summarize milestone baseline evaluations for the project writeup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_baseline_results(
    results_dir: str | Path = "results",
    models_dir: str | Path = "results/milestone_models",
) -> pd.DataFrame:
    """Return one summary row per milestone baseline seed."""
    results_path = Path(results_dir)
    models_path = Path(models_dir)
    evaluation_files = sorted(results_path.glob("milestone_baseline_seed_*.csv"))
    if not evaluation_files:
        raise FileNotFoundError(f"No milestone evaluation files found in {results_path}")

    rows: list[dict[str, int | float | str]] = []
    for evaluation_file in evaluation_files:
        data = pd.read_csv(evaluation_file)
        if data.empty:
            raise ValueError(f"Evaluation file is empty: {evaluation_file}")
        seed = int(evaluation_file.stem.rsplit("_", 1)[-1])
        metadata_file = models_path / f"dqn_baseline_seed_{seed}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "training_timesteps": int(metadata["total_timesteps"]),
                "evaluation_episodes": len(data),
                "mean_reward": data["episode_reward"].mean(),
                "goal_completion_rate": data["goal_reached"].mean(),
                "mean_episode_length": data["episode_length"].mean(),
                "training_seconds": metadata["training_seconds"],
                "outcome": (
                    "Learned"
                    if data["goal_reached"].all()
                    else "Did not learn"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def write_markdown_table(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write a formatted Markdown table and a short interpretation."""
    table = summary.copy()
    table["goal_completion"] = (table["goal_completion_rate"] * 100).map(
        lambda value: f"{value:.0f}%"
    )
    table["mean_reward"] = table["mean_reward"].map(lambda value: f"{value:.1f}")
    table["mean_episode_length"] = table["mean_episode_length"].map(
        lambda value: f"{value:.1f}"
    )
    table["training_seconds"] = table["training_seconds"].map(
        lambda value: f"{value:.1f}"
    )
    table = table[
        [
            "seed",
            "training_timesteps",
            "evaluation_episodes",
            "mean_reward",
            "goal_completion",
            "mean_episode_length",
            "training_seconds",
            "outcome",
        ]
    ]
    table.columns = [
        "Seed",
        "Training steps",
        "Evaluation episodes",
        "Mean reward",
        "Goal completion",
        "Mean episode length",
        "Training time (s)",
        "Outcome",
    ]
    headers = list(table.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join(markdown_rows)
    output = (
        "# Milestone Baseline DQN Results\n\n"
        "All runs used the baseline reward condition, 200,000 training steps, "
        "and deterministic evaluation. Seeds are retained as independent runs.\n\n"
        f"{markdown_table}\n\n"
        "The successful runs learned the shortest seven-action path. The "
        "unsuccessful runs reached the 100-step time limit without reaching "
        "the goal, showing seed sensitivity in this sparse-reward task.\n"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="results/milestone_models")
    parser.add_argument(
        "--output",
        default="results/milestone_baseline_results.md",
    )
    args = parser.parse_args()
    summary = summarize_baseline_results(args.results_dir, args.models_dir)
    write_markdown_table(summary, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
