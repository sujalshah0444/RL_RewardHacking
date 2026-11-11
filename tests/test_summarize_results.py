import json
from pathlib import Path

import pandas as pd

from src.summarize_results import summarize_baseline_results, write_markdown_table


def test_summarize_and_write_baseline_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    models_dir = results_dir / "milestone_models"
    models_dir.mkdir(parents=True)

    for seed, goal_reached in [(0, True), (1, False)]:
        pd.DataFrame(
            {
                "episode_reward": [10.0 if goal_reached else 0.0],
                "episode_length": [7 if goal_reached else 100],
                "goal_reached": [goal_reached],
            }
        ).to_csv(results_dir / f"milestone_baseline_seed_{seed}.csv", index=False)
        (models_dir / f"dqn_baseline_seed_{seed}.json").write_text(
            json.dumps({"total_timesteps": 200000, "training_seconds": 1.0}),
            encoding="utf-8",
        )

    summary = summarize_baseline_results(results_dir, models_dir)
    output = tmp_path / "summary.md"
    write_markdown_table(summary, output)

    assert summary["outcome"].tolist() == ["Learned", "Did not learn"]
    assert "Goal completion" in output.read_text(encoding="utf-8")
