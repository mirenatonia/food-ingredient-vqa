"""Tune a global probability threshold on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import IngredientDataset
from src.models.ingredient_model import IngredientClassifier
from src.training.train import build_transforms, load_config, run_epoch
from src.utils.metrics import multilabel_metrics


THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to config best_checkpoint.",
    )
    return parser.parse_args()


def save_threshold_results(config_path: Path, output_path: Path, best_threshold: float, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_threshold": best_threshold,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    config = load_config(config_path)
    config["inference"]["threshold"] = float(best_threshold)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    checkpoint_path = args.checkpoint or config["paths"]["best_checkpoint"]

    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])
    num_workers = int(config["dataset"]["num_workers"])
    hf_dataset_name = config["dataset"]["hf_dataset_name"]

    _, eval_transform = build_transforms(image_size)
    val_dataset = IngredientDataset(
        csv_path=config["paths"]["val_csv"],
        transform=eval_transform,
        hf_dataset_name=hf_dataset_name,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IngredientClassifier(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    _, val_logits, val_targets = run_epoch(model, val_loader, criterion, device)

    results: list[dict] = []
    best_result: dict | None = None
    for threshold in THRESHOLDS:
        metrics = multilabel_metrics(val_logits, val_targets, threshold=threshold)
        row = {
            "threshold": threshold,
            "micro_f1": float(metrics["micro_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }
        results.append(row)
        if best_result is None or row["micro_f1"] > best_result["micro_f1"]:
            best_result = row

    print(f"Checkpoint: {checkpoint_path}")
    print("Threshold tuning on validation set:")
    print("threshold | micro_f1 | macro_f1 | precision | recall")
    for row in results:
        print(
            f"{row['threshold']:.2f} | "
            f"{row['micro_f1']:.4f} | "
            f"{row['macro_f1']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f}"
        )

    assert best_result is not None
    best_threshold = float(best_result["threshold"])
    print(f"Best threshold: {best_threshold:.2f}")

    output_path = Path("outputs/threshold_tuning.json")
    save_threshold_results(config_path, output_path, best_threshold, results)
    print(f"Saved threshold tuning results to {output_path}")


if __name__ == "__main__":
    main()
