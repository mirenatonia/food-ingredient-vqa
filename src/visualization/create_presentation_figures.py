"""Create presentation-ready figures for the Food Ingredient VQA project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


OUTPUT_DIR = Path("outputs/presentation_figures")

EPOCHS = list(range(1, 11))
TRAIN_LOSS = [0.1456, 0.0998, 0.0882, 0.0821, 0.0779, 0.0749, 0.0726, 0.0705, 0.0700, 0.0687]
VAL_LOSS = [0.1018, 0.0921, 0.0854, 0.0840, 0.0825, 0.0845, 0.0871, 0.0853, 0.0910, 0.0887]
VAL_MICRO_F1 = [0.2628, 0.4358, 0.4719, 0.4646, 0.5028, 0.4960, 0.5078, 0.4983, 0.5013, 0.5231]
VAL_MACRO_F1 = [0.0539, 0.1436, 0.1808, 0.1784, 0.1927, 0.2227, 0.2312, 0.2108, 0.2303, 0.2368]

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
THRESHOLD_MICRO_F1 = [0.4165, 0.4681, 0.4982, 0.5175, 0.5273, 0.5321, 0.5319, 0.5319, 0.5231]
THRESHOLD_MACRO_F1 = [0.2457, 0.2649, 0.2682, 0.2600, 0.2554, 0.2525, 0.2511, 0.2498, 0.2368]
THRESHOLD_PRECISION = [0.2909, 0.3596, 0.4155, 0.4655, 0.5074, 0.5446, 0.5751, 0.6097, 0.6353]
THRESHOLD_RECALL = [0.7333, 0.6705, 0.6219, 0.5826, 0.5489, 0.5201, 0.4948, 0.4718, 0.4446]
SELECTED_THRESHOLD = 0.35

FINAL_TEST_METRICS = {
    "micro_f1": 0.5250,
    "macro_f1": 0.2528,
    "precision": 0.5364,
    "recall": 0.5141,
}

PREVIOUS_METRICS = {
    "micro_f1": 0.4120,
    "macro_f1": 0.1310,
    "precision": 0.6661,
    "recall": 0.2982,
}

FINAL_METRICS = {
    "micro_f1": 0.5250,
    "macro_f1": 0.2528,
    "precision": 0.5364,
    "recall": 0.5141,
}

DATASET_SPLITS = {"Train": 4983, "Validation": 1067, "Test": 1068}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#fbfbfd",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_current_figure(filename: str) -> Path:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def annotate_bars(ax, bars, fmt: str = "{:.4f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def add_box(ax, x: float, y: float, text: str, width: float = 0.18, height: float = 0.12) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#eef4ff",
            "edgecolor": "#2b4c7e",
            "linewidth": 1.5,
        },
    )


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            color="#4f5d75",
        )
    )


def create_training_loss_curve() -> Path:
    plt.figure()
    plt.plot(EPOCHS, TRAIN_LOSS, marker="o", linewidth=2.2, label="Train loss", color="#1f77b4")
    plt.plot(EPOCHS, VAL_LOSS, marker="s", linewidth=2.2, label="Validation loss", color="#d62728")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(EPOCHS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    return save_current_figure("training_loss_curve.png")


def create_f1_over_epochs() -> Path:
    plt.figure()
    plt.plot(EPOCHS, VAL_MICRO_F1, marker="o", linewidth=2.2, label="Validation micro F1", color="#2ca02c")
    plt.plot(EPOCHS, VAL_MACRO_F1, marker="s", linewidth=2.2, label="Validation macro F1", color="#9467bd")
    plt.title("Validation F1 Scores Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.xticks(EPOCHS)
    plt.ylim(0.0, 0.6)
    plt.grid(True, alpha=0.3)
    plt.legend()
    return save_current_figure("f1_over_epochs.png")


def create_threshold_micro_f1() -> Path:
    plt.figure()
    plt.plot(THRESHOLDS, THRESHOLD_MICRO_F1, marker="o", linewidth=2.4, color="#1f77b4")
    plt.axvline(SELECTED_THRESHOLD, linestyle="--", linewidth=1.8, color="#d62728")
    plt.annotate(
        "Selected threshold = 0.35",
        xy=(SELECTED_THRESHOLD, 0.5321),
        xytext=(0.23, 0.545),
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        fontsize=10,
    )
    plt.title("Threshold Tuning by Validation Micro F1")
    plt.xlabel("Threshold")
    plt.ylabel("Micro F1")
    plt.grid(True, alpha=0.3)
    return save_current_figure("threshold_micro_f1.png")


def create_threshold_precision_recall() -> Path:
    plt.figure()
    plt.plot(THRESHOLDS, THRESHOLD_PRECISION, marker="o", linewidth=2.2, label="Precision", color="#ff7f0e")
    plt.plot(THRESHOLDS, THRESHOLD_RECALL, marker="s", linewidth=2.2, label="Recall", color="#17becf")
    plt.title("Precision-Recall Trade-off Across Thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return save_current_figure("threshold_precision_recall.png")


def create_threshold_all_metrics() -> Path:
    plt.figure()
    plt.plot(THRESHOLDS, THRESHOLD_MICRO_F1, marker="o", linewidth=2.0, label="Micro F1", color="#1f77b4")
    plt.plot(THRESHOLDS, THRESHOLD_MACRO_F1, marker="s", linewidth=2.0, label="Macro F1", color="#9467bd")
    plt.plot(THRESHOLDS, THRESHOLD_PRECISION, marker="^", linewidth=2.0, label="Precision", color="#ff7f0e")
    plt.plot(THRESHOLDS, THRESHOLD_RECALL, marker="D", linewidth=2.0, label="Recall", color="#2ca02c")
    plt.title("Threshold Tuning Metrics")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return save_current_figure("threshold_all_metrics.png")


def create_final_test_metrics_bar() -> Path:
    labels = list(FINAL_TEST_METRICS.keys())
    values = list(FINAL_TEST_METRICS.values())
    plt.figure()
    ax = plt.gca()
    bars = ax.bar(labels, values, color=["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"])
    ax.set_title("Final Test Performance at Threshold 0.35")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 0.65)
    ax.grid(True, axis="y", alpha=0.25)
    annotate_bars(ax, bars)
    return save_current_figure("final_test_metrics_bar.png")


def create_dataset_split_bar() -> Path:
    labels = list(DATASET_SPLITS.keys())
    values = list(DATASET_SPLITS.values())
    plt.figure()
    ax = plt.gca()
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Dataset Split Used in Our Project")
    ax.set_ylabel("Number of samples")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 40, str(value), ha="center", va="bottom", fontsize=10)
    return save_current_figure("dataset_split_bar.png")


def create_pipeline_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_axis_off()
    positions = [0.06, 0.24, 0.43, 0.60, 0.77, 0.93]
    labels = [
        "Food Image",
        "EfficientNet-B0",
        "103 Ingredient\nProbabilities",
        "Threshold 0.35",
        "Ingredient Query",
        "Yes / Maybe / No\nAnswer",
    ]
    for x, label in zip(positions, labels):
        add_box(ax, x, 0.5, label)
    for start_x, end_x in zip(positions[:-1], positions[1:]):
        add_arrow(ax, (start_x + 0.07, 0.5), (end_x - 0.08, 0.5))
    return save_current_figure("pipeline_diagram.png")


def create_label_conversion_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_axis_off()
    positions = [0.10, 0.35, 0.60, 0.86]
    labels = [
        "Segmentation Mask",
        "classes_on_image",
        "Remove background\nid 0",
        "Multi-hot vector\n(length 103)",
    ]
    for x, label in zip(positions, labels):
        add_box(ax, x, 0.5, label)
    for start_x, end_x in zip(positions[:-1], positions[1:]):
        add_arrow(ax, (start_x + 0.09, 0.5), (end_x - 0.10, 0.5))
    return save_current_figure("label_conversion_diagram.png")


def create_threshold_decision_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_axis_off()
    add_box(ax, 0.5, 0.82, "Ingredient probability", width=0.24)
    add_box(ax, 0.23, 0.40, "probability >= 0.35\nYes", width=0.22)
    add_box(ax, 0.50, 0.40, "0.20 <= probability < 0.35\nMaybe", width=0.28)
    add_box(ax, 0.77, 0.40, "probability < 0.20\nNo", width=0.22)
    add_arrow(ax, (0.46, 0.74), (0.27, 0.50))
    add_arrow(ax, (0.50, 0.74), (0.50, 0.50))
    add_arrow(ax, (0.54, 0.74), (0.73, 0.50))
    return save_current_figure("threshold_decision_diagram.png")


def create_before_after_training_comparison() -> Path:
    metrics = list(PREVIOUS_METRICS.keys())
    previous_values = [PREVIOUS_METRICS[m] for m in metrics]
    final_values = [FINAL_METRICS[m] for m in metrics]
    x = list(range(len(metrics)))
    width = 0.36
    plt.figure()
    ax = plt.gca()
    bars1 = ax.bar([i - width / 2 for i in x], previous_values, width=width, label="Subset model", color="#9ecae1")
    bars2 = ax.bar([i + width / 2 for i in x], final_values, width=width, label="Full model + tuning", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("Score")
    ax.set_title("Subset Training vs Full Training + Threshold Tuning")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    return save_current_figure("before_after_training_comparison.png")


def create_real_world_results_table() -> Path:
    rows = [
        ["Hamburger", "Bread, lettuce, steak, tomato detected well", "Minor omissions", "General Western foods work better"],
        ["Korean food", "Egg detected", "Carrot uncertain or missed", "Some visible ingredients remain low-confidence"],
        ["Pizza", "Pizza class detected", "Pepper or cheese missed, mushrooms confused", "Ingredient-level detail is harder than dish-level cues"],
        ["Kebab", "Tomato and steak detected", "Pepper confused as asparagus", "Domain shift affects fine-grained labels"],
        ["Chicken rice", "Some greens detected", "Missed chicken and rice", "Background cues can dominate the prediction"],
        ["Salad", "Tomato present visually", "Tomato below yes threshold", "Thresholding changes the user-facing answer"],
    ]
    columns = ["Image type", "Correct behavior", "Mistake", "Lesson"]
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbe9ff")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8fbff" if row % 2 == 1 else "#eef4ff")
        cell.set_edgecolor("#b0b8c2")
    ax.set_title("Real-World Qualitative Results", fontsize=16, pad=18)
    return save_current_figure("real_world_results_table.png")


def create_presentation_summary_cards() -> Path:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_axis_off()
    cards = [
        (
            0.17,
            "What worked",
            "Hamburger ingredient detection,\negg detection,\nand pizza class recognition",
            "#e8f5e9",
        ),
        (
            0.50,
            "What improved",
            "Full training + threshold tuning\nimproved micro F1\nfrom 0.412 to 0.525",
            "#e3f2fd",
        ),
        (
            0.83,
            "What failed",
            "Domain shift, Turkish dishes,\nsmall or hidden ingredients,\nand background confusion",
            "#fff3e0",
        ),
    ]
    for x, title, body, color in cards:
        ax.text(
            x,
            0.55,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=14,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": color,
                "edgecolor": "#4f5d75",
                "linewidth": 1.6,
            },
        )
    ax.set_title("Presentation Summary", fontsize=18, pad=12)
    return save_current_figure("presentation_summary_cards.png")


def create_figure_index(paths: list[Path]) -> Path:
    lines = [
        "# Figure Index",
        "",
        "| Filename | Suggested slide | What it explains |",
        "| --- | --- | --- |",
        "| training_loss_curve.png | Training results | Shows optimization stability and convergence. |",
        "| f1_over_epochs.png | Training results | Shows how validation micro and macro F1 improved over epochs. |",
        "| threshold_micro_f1.png | Threshold tuning | Shows why 0.35 was selected. |",
        "| threshold_precision_recall.png | Threshold tuning | Explains the precision-recall trade-off. |",
        "| threshold_all_metrics.png | Threshold tuning appendix | Compares all threshold metrics together. |",
        "| final_test_metrics_bar.png | Final quantitative results | Summarizes final test performance at threshold 0.35. |",
        "| dataset_split_bar.png | Dataset overview | Shows the train, validation, and test split sizes. |",
        "| pipeline_diagram.png | Method overview | Explains the inference pipeline from image to answer. |",
        "| label_conversion_diagram.png | Data preparation | Explains how segmentation labels become multi-hot training targets. |",
        "| threshold_decision_diagram.png | Demo logic | Explains the Yes / Maybe / No decision rule in the UI. |",
        "| before_after_training_comparison.png | Improvement story | Compares subset training against full training plus tuning. |",
        "| real_world_results_table.png | Qualitative analysis | Summarizes successes, failures, and lessons on internet images. |",
        "| presentation_summary_cards.png | Closing slide | Summarizes what worked, what improved, and what failed. |",
        "",
        "Generated files:",
    ]
    lines.extend([f"- `{path.name}`" for path in paths])
    output_path = OUTPUT_DIR / "figure_index.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    apply_style()
    ensure_output_dir()

    generated_paths = [
        create_training_loss_curve(),
        create_f1_over_epochs(),
        create_threshold_micro_f1(),
        create_threshold_precision_recall(),
        create_threshold_all_metrics(),
        create_final_test_metrics_bar(),
        create_dataset_split_bar(),
        create_pipeline_diagram(),
        create_label_conversion_diagram(),
        create_threshold_decision_diagram(),
        create_before_after_training_comparison(),
        create_real_world_results_table(),
        create_presentation_summary_cards(),
    ]
    index_path = create_figure_index(generated_paths)

    print("Generated presentation figures:")
    for path in generated_paths:
        print(path)
    print(index_path)


if __name__ == "__main__":
    main()
