"""Predict ingredient probabilities for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models.ingredient_model import IngredientClassifier
from src.utils.labels import get_class_names
from src.utils.metrics import extract_top_k_ingredients


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(checkpoint_path: str | Path, num_classes: int, device: torch.device):
    model = IngredientClassifier(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_probabilities(
    image_path: str | Path,
    checkpoint_path: str | Path,
    config: dict,
    threshold: float | None = None,
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = get_class_names()
    image_size = int(config["dataset"]["image_size"])
    top_k = int(config["inference"]["top_k"])
    threshold = (
        float(threshold) if threshold is not None else float(config["inference"]["threshold"])
    )

    transform = build_inference_transform(image_size)
    model = load_model(checkpoint_path, len(class_names), device)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    probability_map = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }
    top_predictions = extract_top_k_ingredients(probabilities, class_names, top_k=top_k)
    return probability_map, top_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to a trained checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file.",
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
    threshold = float(args.threshold) if args.threshold is not None else None
    _, top_predictions = predict_probabilities(
        args.image,
        args.checkpoint,
        config,
        threshold=threshold,
    )

    effective_threshold = (
        threshold if threshold is not None else float(config["inference"]["threshold"])
    )
    print(f"Threshold: {effective_threshold:.2f}")
    print("Top predicted ingredients:")
    for ingredient_name, probability in top_predictions:
        print(f"- {ingredient_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
