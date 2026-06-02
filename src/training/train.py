"""Train a multi-label ingredient recognition model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import IngredientDataset
from src.models.ingredient_model import IngredientClassifier
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
    return parser.parse_args()


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create torchvision transforms for training and validation.

    Training uses small augmentations to improve generalization.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform


def maybe_subset_dataset(dataset: Dataset, subset_size: int, seed: int) -> Dataset:
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["dataset"]["batch_size"])
    num_workers = int(config["dataset"]["num_workers"])
    hf_dataset_name = config["dataset"]["hf_dataset_name"]
    subset_seed = int(config["dataset"].get("train_val_test_seed", 42))
    use_subset = bool(config["dataset"].get("use_subset", False))

    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = IngredientDataset(
        csv_path=config["paths"]["train_csv"],
        transform=train_transform,
        hf_dataset_name=hf_dataset_name,
    )
    val_dataset = IngredientDataset(
        csv_path=config["paths"]["val_csv"],
        transform=eval_transform,
        hf_dataset_name=hf_dataset_name,
    )

    if use_subset:
        train_dataset = maybe_subset_dataset(
            train_dataset,
            int(config["dataset"].get("train_subset_size", 0) or 0),
            seed=subset_seed,
        )
        val_dataset = maybe_subset_dataset(
            val_dataset,
            int(config["dataset"].get("val_subset_size", 0) or 0),
            seed=subset_seed + 1,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_logits = []
    all_targets = []

    progress_bar = tqdm(loader, leave=False)
    for batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    average_loss = total_loss / max(len(loader.dataset), 1)
    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0))
    targets_array = (
        np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 0))
    )
    return average_loss, logits_array, targets_array


def save_checkpoint(
    model: nn.Module,
    config: dict,
    output_path: str | Path,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(config)
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            "train.csv is empty. Run src/data/build_multilabel_index.py first."
        )
    if len(val_loader.dataset) == 0:
        raise RuntimeError(
            "val.csv is empty. The current training script expects validation data."
        )

    num_classes = int(config["model"]["num_classes"])
    model = IngredientClassifier(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_val_micro_f1 = -1.0
    epochs = int(config["training"]["epochs"])
    threshold = float(config["inference"]["threshold"])

    print(f"Training on device: {device}")
    print(
        f"Train samples: {len(train_loader.dataset)} | "
        f"Val samples: {len(val_loader.dataset)}"
    )
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_logits, val_targets = run_epoch(model, val_loader, criterion, device)

        val_metric_values = multilabel_metrics(
            logits=val_logits,
            targets=val_targets,
            threshold=threshold,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"micro_f1={val_metric_values['micro_f1']:.4f} | "
            f"macro_f1={val_metric_values['macro_f1']:.4f}"
        )

        if val_metric_values["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metric_values["micro_f1"]
            save_checkpoint(
                model=model,
                config=config,
                output_path=config["paths"]["best_checkpoint"],
                epoch=epoch,
                metrics=val_metric_values,
            )
            print(f"Saved new best checkpoint to {config['paths']['best_checkpoint']}")


if __name__ == "__main__":
    main()
