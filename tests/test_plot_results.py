from pathlib import Path

import pandas as pd

from src.plot_results import create_baseline_figure


def test_create_baseline_figure(tmp_path: Path) -> None:
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
            '{"total_timesteps": 200000, "training_seconds": 1.0}',
            encoding="utf-8",
        )

    output = create_baseline_figure(
        results_dir=results_dir,
        models_dir=models_dir,
        output_path=tmp_path / "figure.png",
    )

    assert output.exists()
    assert output.stat().st_size > 0
