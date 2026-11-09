from pathlib import Path

from src.evaluate import evaluate_agent
from src.train import train_agent


def test_training_and_evaluation_pipeline(tmp_path: Path) -> None:
    model_path = train_agent(
        reward_condition="baseline",
        seed=3,
        total_timesteps=64,
        output_dir=tmp_path / "models",
    )

    assert model_path.exists()
    assert model_path.with_suffix(".json").exists()

    results = evaluate_agent(
        model_path=model_path,
        reward_condition="baseline",
        episodes=2,
        seed=3,
    )

    assert len(results) == 2
    assert {
        "episode_reward",
        "episode_length",
        "goal_reached",
        "bonus_collections",
        "reward_hacked",
    }.issubset(results.columns)
