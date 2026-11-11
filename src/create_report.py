"""Create the RL milestone report as a DOCX document."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src.summarize_results import summarize_baseline_results


def _set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(8.5)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def _add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers):
        _set_cell_text(cell, header, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            _set_cell_text(cell, value)


def _add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_heading(text, level=level)
    heading.paragraph_format.space_before = Pt(8)
    heading.paragraph_format.space_after = Pt(3)


def create_report(
    results_dir: str | Path = "results",
    models_dir: str | Path = "results/milestone_models",
    figure_path: str | Path = "figures/milestone_baseline_results.png",
    output_path: str | Path = "RL_RewardHacking_Milestone1_Report.docx",
) -> Path:
    """Create and save the milestone report."""
    summary = summarize_baseline_results(results_dir, models_dir)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9.5)
    styles["Normal"].paragraph_format.space_after = Pt(4)
    styles["Normal"].paragraph_format.line_spacing = 1.05
    for style_name in ("Heading 1", "Heading 2"):
        styles[style_name].font.name = "Arial"
        styles[style_name].font.color.rgb = RGBColor(31, 78, 121)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Reward Hacking and Reward Function Design in Reinforcement Learning")
    title_run.bold = True
    title_run.font.size = Pt(15)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Milestone 1 Report: Environment, Experimental Design, and DQN Baseline").italic = True
    author = document.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.add_run("CPS 379 | Sujal Shah")

    _add_heading(document, "1. Introduction", 1)
    document.add_paragraph(
        "Reinforcement learning (RL) agents learn behavior by maximizing the reward "
        "defined by the designer. This creates a potential mismatch between the "
        "intended task and the measured objective: an agent can discover a strategy "
        "that produces high reward without accomplishing the task a person intended. "
        "This behavior is commonly called reward hacking."
    )
    document.add_paragraph(
        "This project studies reward hacking in a small, deterministic GridWorld. "
        "The research question is: how does reward-function design affect the behavior "
        "learned by an RL agent, and can a simple change—making a bonus reward "
        "collectible only once—reduce exploitation? The project compares a baseline "
        "goal reward, a repeatedly collectible bonus, and a one-time bonus."
    )

    _add_heading(document, "2. Environment and MDP Design", 1)
    document.add_paragraph(
        "The environment is a custom Gymnasium-compatible 8 by 4 GridWorld. The "
        "agent starts at (0, 0), the goal is at (7, 0), and the bonus location is "
        "at (4, 2). The layout, starting position, goal, bonus location, and "
        "100-step episode limit are identical across conditions."
    )
    document.add_paragraph(
        "The state contains the agent's x coordinate, y coordinate, facing direction, "
        "and a binary variable indicating whether the bonus has been collected. "
        "Including the bonus flag makes the one-time-bonus environment Markov. "
        "The action space is discrete: turn left, turn right, or move forward. "
        "Transitions are deterministic. Reaching the goal terminates an episode; "
        "otherwise the episode is truncated after 100 steps."
    )

    _add_heading(document, "3. Experimental Conditions", 1)
    document.add_paragraph(
        "Only the reward logic changes between conditions. The exact values were "
        "locked before the baseline experiments so that the comparison remains "
        "interpretable."
    )
    _add_table(
        document,
        ["Condition", "Goal reward", "Bonus reward", "Purpose"],
        [
            ["Baseline", "+10", "0", "Reference task behavior"],
            ["Exploitable", "+10", "+1 on every bonus entry", "Test repeated reward exploitation"],
            ["Fixed bonus", "+10", "+1 on first entry only", "Test a simple mitigation"],
        ],
    )
    document.add_paragraph(
        "The exploitable condition is intentionally designed so that the agent can "
        "receive repeated positive rewards by returning to the bonus location rather "
        "than completing the goal. The primary evidence of reward hacking will be a "
        "mismatch between high episode reward and low goal-completion rate."
    )

    _add_heading(document, "4. DQN Method and Experimental Procedure", 1)
    document.add_paragraph(
        "Agents are trained with the Stable-Baselines3 implementation of Deep "
        "Q-Networks (DQN) using an MLP policy. DQN estimates action values with a "
        "neural network and uses experience replay and a target network. The initial "
        "experiments used a discount factor of 0.99, a learning rate of 0.001, "
        "a replay buffer of 10,000 transitions, a batch size of 64, and the same "
        "exploration schedule across runs."
    )
    document.add_paragraph(
        "The milestone baseline uses 200,000 environment steps per run and 100 "
        "deterministic evaluation episodes. Four random seeds (0 through 3) were "
        "retained as separate runs. A hand-coded policy was tested first and reached "
        "the goal in seven actions, confirming that the task is reachable independent "
        "of learning."
    )

    _add_heading(document, "5. Baseline Results", 1)
    document.add_paragraph(
        "Table 1 reports the baseline DQN results. Seeds 0 and 1 learned the shortest "
        "seven-action path and reached the goal in every evaluation episode. Seeds 2 "
        "and 3 did not reach the goal and instead timed out after 100 steps. These "
        "results demonstrate that DQN can learn the environment, but performance is "
        "sensitive to random seed under the sparse goal reward."
    )
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                str(int(row["seed"])),
                f"{int(row['training_timesteps']):,}",
                str(int(row["evaluation_episodes"])),
                f"{row['mean_reward']:.1f}",
                f"{row['goal_completion_rate'] * 100:.0f}%",
                f"{row['mean_episode_length']:.1f}",
                f"{row['training_seconds']:.1f}",
                row["outcome"],
            ]
        )
    _add_table(
        document,
        [
            "Seed",
            "Steps",
            "Eval. episodes",
            "Mean reward",
            "Goal completion",
            "Mean length",
            "Training time (s)",
            "Outcome",
        ],
        rows,
    )
    document.add_picture(str(figure_path), width=Inches(5.8))
    last_paragraph = document.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption = document.add_paragraph(
        "Figure 1. Baseline DQN evaluation results across random seeds. "
        "Green bars indicate runs that learned the task; red bars indicate "
        "runs that did not reach the goal."
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    caption.runs[0].font.size = Pt(8)
    document.add_paragraph(
        "The initial 50,000-step runs produced no successful baseline policies. "
        "Increasing training to 200,000 steps allowed some seeds to discover the "
        "goal path, identifying insufficient training for the sparse-reward task as "
        "an important factor. The unsuccessful seeds are not discarded; retaining "
        "them gives a more accurate picture of training variability."
    )

    _add_heading(document, "6. Statistical Analysis Plan", 1)
    document.add_paragraph(
        "The primary outcomes will be goal-completion rate, average episode reward, "
        "and reward-hacking rate. For the final comparison across reward conditions, "
        "each random seed will be treated as one independent experimental run. "
        "The 100 episodes generated by one model will be used to calculate that "
        "run's metrics, but will not be treated as 100 independent replicates."
    )
    document.add_paragraph(
        "For each condition, I will report the mean, standard deviation, and 95% "
        "confidence interval across seeds. If assumptions of approximate normality "
        "and equal variance are reasonable, I will use a one-way ANOVA followed by "
        "Holm-corrected pairwise comparisons. If those assumptions are not reasonable, "
        "I will use a Kruskal-Wallis test followed by corrected pairwise tests. "
        "Reward hacking will be defined as repeated bonus collection in an episode "
        "without reaching the goal."
    )

    _add_heading(document, "7. Limitations and Next Steps", 1)
    document.add_paragraph(
        "The current environment is intentionally small, deterministic, and fully "
        "observable. This makes behavior easier to interpret, but limits how broadly "
        "the results can be generalized. The turn-based action space and sparse goal "
        "reward also make DQN sensitive to exploration and seed. The milestone "
        "baseline therefore establishes that the implementation works and provides "
        "a useful learning benchmark, but it is not yet evidence that one reward "
        "condition is superior."
    )
    document.add_paragraph(
        "The next step is to run the same calibrated DQN procedure for all three "
        "reward conditions across multiple seeds, then evaluate reward, goal "
        "completion, episode length, bonus collections, and reward-hacking rate. "
        "Future extensions could include partial observations, randomized layouts, "
        "different bonus values, or a comparison with PPO."
    )

    _add_heading(document, "8. Conclusion", 1)
    document.add_paragraph(
        "This milestone established a working RL environment, locked the reward "
        "conditions, and produced baseline DQN results using Stable-Baselines3. "
        "The baseline agent learned successfully for some seeds, while other seeds "
        "failed under the same budget, highlighting the importance of reporting "
        "random-seed variability. The complete reward-hacking comparison will use "
        "this baseline as its reference condition and will evaluate whether limiting "
        "the bonus to one collection improves alignment with the intended goal."
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="results/milestone_models")
    parser.add_argument("--figure", default="figures/milestone_baseline_results.png")
    parser.add_argument(
        "--output",
        default="RL_RewardHacking_Milestone1_Report.docx",
    )
    args = parser.parse_args()
    print(create_report(args.results_dir, args.models_dir, args.figure, args.output))


if __name__ == "__main__":
    main()
