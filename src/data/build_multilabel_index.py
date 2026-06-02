"""Build CSV index files for multi-label ingredient recognition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from datasets import DatasetDict, load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.labels import get_num_classes


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def labels_to_multi_hot(label_ids: Sequence[int], num_classes: int) -> np.ndarray:
    """
    Convert dataset class ids into a model target vector.

    FoodSeg103 uses background id 0. Model outputs are zero-based for the 103
    foreground classes, so class id 1 maps to output index 0.
    """
    multi_hot = np.zeros(num_classes, dtype=np.float32)
    for class_id in sorted(set(int(x) for x in label_ids)):
        if class_id <= 0:
            continue
        class_index = class_id - 1
        if 0 <= class_index < num_classes:
            multi_hot[class_index] = 1.0
    return multi_hot


def encode_label_indices(label_ids: Sequence[int]) -> str:
    """Store only foreground label ids in a compact CSV-friendly string."""
    label_ids = sorted(set(int(x) for x in label_ids if int(x) > 0))
    return " ".join(str(x) for x in label_ids)


def mask_to_label_ids(mask_value) -> List[int]:
    """
    Read a segmentation mask and extract unique class ids.

    The function accepts a PIL image, a numpy array, or a path-like object.
    """
    if isinstance(mask_value, Image.Image):
        mask_array = np.array(mask_value)
    elif isinstance(mask_value, np.ndarray):
        mask_array = mask_value
    else:
        mask_array = np.array(Image.open(mask_value))

    unique_ids = np.unique(mask_array).tolist()
    return [int(class_id) for class_id in unique_ids if int(class_id) != 0]


def build_rows_from_hf_split(
    split_dataset,
    split_name: str,
    num_classes: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for index, sample in enumerate(tqdm(split_dataset, desc=f"Indexing {split_name}")):
        # Prefer the dataset-provided classes_on_image field because it is fast
        # and already represents image-level labels.
        label_ids: Optional[Iterable[int]] = sample.get("classes_on_image")

        # Defensive fallback: if classes_on_image is missing, derive labels from
        # the segmentation mask column instead.
        if label_ids is None:
            mask_value = sample.get("label") or sample.get("mask")
            if mask_value is None:
                print(
                    f"[WARN] Sample {index} in split '{split_name}' has no "
                    "'classes_on_image' and no obvious mask column."
                )
                continue
            label_ids = mask_to_label_ids(mask_value)

        label_ids = [int(x) for x in label_ids if int(x) != 0]

        row = {
            "sample_id": int(sample.get("id", index)),
            "hf_row_idx": index,
            "split": split_name,
            "source": "huggingface",
            "image_path": "",
            "label_indices": encode_label_indices(label_ids),
            "multi_hot": " ".join(
                str(int(x))
                for x in labels_to_multi_hot(label_ids=label_ids, num_classes=num_classes)
            ),
        }
        rows.append(row)

    return rows


def detect_local_split_dirs(local_dataset_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Search a few common folder layouts for FoodSeg103 image and mask directories.

    TODO: Adjust this if your local download uses different folder names.
    """
    candidates = {
        "train": {
            "images": [
                local_dataset_dir / "Images" / "train",
                local_dataset_dir / "images" / "train",
                local_dataset_dir / "train" / "images",
            ],
            "masks": [
                local_dataset_dir / "Annotations" / "train",
                local_dataset_dir / "annotations" / "train",
                local_dataset_dir / "train" / "masks",
                local_dataset_dir / "labels" / "train",
            ],
        },
        "validation": {
            "images": [
                local_dataset_dir / "Images" / "val",
                local_dataset_dir / "images" / "val",
                local_dataset_dir / "validation" / "images",
                local_dataset_dir / "val" / "images",
            ],
            "masks": [
                local_dataset_dir / "Annotations" / "val",
                local_dataset_dir / "annotations" / "val",
                local_dataset_dir / "validation" / "masks",
                local_dataset_dir / "val" / "masks",
                local_dataset_dir / "labels" / "val",
            ],
        },
    }

    detected: Dict[str, Dict[str, Path]] = {}
    for split_name, split_candidates in candidates.items():
        image_dir = next((p for p in split_candidates["images"] if p.exists()), None)
        mask_dir = next((p for p in split_candidates["masks"] if p.exists()), None)
        if image_dir and mask_dir:
            detected[split_name] = {"images": image_dir, "masks": mask_dir}

    return detected


def build_rows_from_local_split(
    image_dir: Path,
    mask_dir: Path,
    split_name: str,
    num_classes: int,
) -> List[Dict[str, object]]:
    image_paths = sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    rows: List[Dict[str, object]] = []
    for image_path in tqdm(image_paths, desc=f"Indexing local {split_name}"):
        stem = image_path.stem
        mask_candidates = [
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.jpg",
            mask_dir / image_path.name,
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)

        if mask_path is None:
            print(f"[WARN] Missing mask for image: {image_path}")
            continue

        label_ids = mask_to_label_ids(mask_path)
        rows.append(
            {
                "sample_id": stem,
                "hf_row_idx": "",
                "split": split_name,
                "source": "local",
                "image_path": str(image_path.resolve()),
                "label_indices": encode_label_indices(label_ids),
                "multi_hot": " ".join(
                    str(int(x))
                    for x in labels_to_multi_hot(
                        label_ids=label_ids, num_classes=num_classes
                    )
                ),
            }
        )

    return rows


def save_rows(rows: List[Dict[str, object]], output_csv: Path) -> None:
    dataframe = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    print(f"Saved {len(dataframe)} rows to {output_csv}")


def split_validation_rows(
    validation_rows: List[Dict[str, object]],
    val_fraction: float,
    seed: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not validation_rows:
        return [], []

    if len(validation_rows) < 2:
        return validation_rows, validation_rows

    val_rows, test_rows = train_test_split(
        validation_rows,
        test_size=1.0 - val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return list(val_rows), list(test_rows)


def build_from_huggingface(config: dict) -> None:
    dataset_name = config["dataset"]["hf_dataset_name"]
    processed_dir = Path(config["paths"]["processed_data_dir"])
    num_classes = int(config["model"]["num_classes"])
    seed = int(config["dataset"]["train_val_test_seed"])
    val_fraction = float(config["dataset"]["val_fraction_from_validation"])

    print(f"Loading Hugging Face dataset: {dataset_name}")
    dataset: DatasetDict = load_dataset(dataset_name)
    print(f"Available splits: {list(dataset.keys())}")

    train_rows = []
    validation_rows = []

    if "train" in dataset:
        train_rows = build_rows_from_hf_split(dataset["train"], "train", num_classes)

    if "validation" in dataset:
        validation_rows = build_rows_from_hf_split(
            dataset["validation"], "validation", num_classes
        )
    elif "val" in dataset:
        validation_rows = build_rows_from_hf_split(dataset["val"], "validation", num_classes)
    else:
        print("[WARN] No validation split found. Validation and test CSVs will be empty.")

    val_rows, test_rows = split_validation_rows(validation_rows, val_fraction, seed)

    # We relabel the saved split field to match the CSV filename consumers.
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"

    save_rows(train_rows, processed_dir / "train.csv")
    save_rows(val_rows, processed_dir / "val.csv")
    save_rows(test_rows, processed_dir / "test.csv")


def build_from_local(config: dict) -> None:
    local_dataset_dir = Path(config["dataset"]["local_dataset_dir"])
    processed_dir = Path(config["paths"]["processed_data_dir"])
    num_classes = int(config["model"]["num_classes"])
    seed = int(config["dataset"]["train_val_test_seed"])
    val_fraction = float(config["dataset"]["val_fraction_from_validation"])

    print(f"Searching local dataset folder: {local_dataset_dir}")
    if not local_dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset directory was not found: {local_dataset_dir}"
        )

    detected = detect_local_split_dirs(local_dataset_dir)
    print("Detected local split directories:")
    print(detected)

    if "train" not in detected:
        raise RuntimeError(
            "Could not find a local training image/mask folder pair. "
            "Please update detect_local_split_dirs() for your folder layout."
        )

    train_rows = build_rows_from_local_split(
        detected["train"]["images"],
        detected["train"]["masks"],
        "train",
        num_classes,
    )

    validation_rows: List[Dict[str, object]] = []
    if "validation" in detected:
        validation_rows = build_rows_from_local_split(
            detected["validation"]["images"],
            detected["validation"]["masks"],
            "validation",
            num_classes,
        )

    val_rows, test_rows = split_validation_rows(validation_rows, val_fraction, seed)
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"

    save_rows(train_rows, processed_dir / "train.csv")
    save_rows(val_rows, processed_dir / "val.csv")
    save_rows(test_rows, processed_dir / "test.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the project config file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["huggingface", "local"],
        help="Override the dataset source from the config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    if int(config["model"]["num_classes"]) != get_num_classes():
        raise ValueError(
            "Config num_classes does not match FoodSeg103 label list length. "
            "Please keep model.num_classes equal to 103."
        )

    source = args.source or config["dataset"]["source"]
    print(f"Building CSV index from source: {source}")

    if source == "huggingface":
        build_from_huggingface(config)
    else:
        build_from_local(config)


if __name__ == "__main__":
    main()
