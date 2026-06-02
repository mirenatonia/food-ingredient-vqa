"""Evaluate a trained checkpoint on the test CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import IngredientDataset
from src.models.ingredient_model import IngredientClassifier
from src.training.train import run_epoch
from src.utils.metrics import multilabel_metrics


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["paths"]["best_checkpoint"]

    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])
    num_workers = int(config["dataset"]["num_workers"])
    hf_dataset_name = config["dataset"]["hf_dataset_name"]
    threshold = float(args.threshold) if args.threshold is not None else float(
        config["inference"]["threshold"]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = IngredientDataset(
        csv_path=config["paths"]["test_csv"],
        transform=eval_transform,
        hf_dataset_name=hf_dataset_name,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    if len(test_dataset) == 0:
        raise RuntimeError(
            "test.csv is empty. Rebuild the processed CSV files before evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IngredientClassifier(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_logits, test_targets = run_epoch(model, test_loader, criterion, device)
    test_metrics = multilabel_metrics(test_logits, test_targets, threshold=threshold)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
