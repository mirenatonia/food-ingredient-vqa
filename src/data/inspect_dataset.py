"""Inspect the Hugging Face FoodSeg103 dataset structure."""

from __future__ import annotations

from pprint import pprint

from datasets import load_dataset


HF_DATASET_NAME = "tejasmishra77/FoodSeg103"


def describe_value(value) -> str:
    """Return a short type summary for debugging."""
    if hasattr(value, "size"):
        return f"{type(value).__name__}(size={getattr(value, 'size', None)})"
    if isinstance(value, list):
        preview = value[:5]
        return f"list(len={len(value)}, preview={preview})"
    return f"{type(value).__name__}: {value}"


def main() -> None:
    print(f"Loading dataset: {HF_DATASET_NAME}")
    dataset = load_dataset(HF_DATASET_NAME)

    print("\nDataset object:")
    print(dataset)

    print("\nSplit names:")
    print(list(dataset.keys()))

    for split_name, split_dataset in dataset.items():
        print(f"\n--- Split: {split_name} ---")
        print(split_dataset)
        print("Column names:", split_dataset.column_names)
        print("Features:")
        pprint(split_dataset.features)

        sample = split_dataset[0]
        print("\nSample keys:")
        print(list(sample.keys()))

        print("\nSample value types:")
        for key, value in sample.items():
            print(f"  {key}: {describe_value(value)}")

        print("\nOnly showing one sample from the first split for readability.")
        break


if __name__ == "__main__":
    main()
