"""Answer simple ingredient questions from model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from src.inference.predict import predict_probabilities
from src.utils.labels import get_candidate_labels_from_question


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def answer_food_question(
    image_path: str | Path,
    question: str,
    checkpoint_path: str | Path,
    config: dict,
    threshold: float | None = None,
    maybe_threshold: float = 0.20,
) -> Dict[str, object]:
    probability_map, top_predictions = predict_probabilities(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        config=config,
        threshold=threshold,
    )

    threshold = (
        float(threshold) if threshold is not None else float(config["inference"]["threshold"])
    )
    candidate_labels = get_candidate_labels_from_question(question)

    if candidate_labels is None:
        return {
            "answer": "I could not match that question to a supported ingredient name.",
            "confidence": 0.0,
            "target_ingredient": None,
            "top_predictions": top_predictions,
        }

    best_label = None
    best_probability = -1.0
    for label in candidate_labels:
        probability = probability_map.get(label, 0.0)
        if probability > best_probability:
            best_label = label
            best_probability = probability

    if best_probability >= threshold:
        answer = "Yes"
    elif best_probability >= maybe_threshold:
        answer = "Maybe"
    else:
        answer = "No"

    return {
        "answer": answer,
        "confidence": float(best_probability),
        "target_ingredient": best_label,
        "yes_threshold": float(threshold),
        "maybe_threshold": float(maybe_threshold),
        "top_predictions": top_predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to an input image.")
    parser.add_argument("--question", type=str, required=True, help="A yes/no ingredient question.")
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
    parser.add_argument(
        "--maybe-threshold",
        type=float,
        default=0.20,
        help="Probability threshold for a Maybe answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = answer_food_question(
        image_path=args.image,
        question=args.question,
        checkpoint_path=args.checkpoint,
        config=config,
        threshold=args.threshold,
        maybe_threshold=args.maybe_threshold,
    )

    yes_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config["inference"]["threshold"])
    )
    print(f"Question: {args.question}")
    print(f"Target ingredient: {result['target_ingredient']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Yes threshold: {yes_threshold:.2f}")
    print(f"Maybe threshold: {args.maybe_threshold:.2f}")
    print("Top predicted ingredients:")
    for ingredient_name, probability in result["top_predictions"]:
        print(f"- {ingredient_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
