"""Label utilities for FoodSeg103 ingredient prediction."""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# FoodSeg103 contains 103 foreground classes. We exclude background id 0.
FOODSEG103_INGREDIENTS: List[str] = [
    "candy",
    "egg tart",
    "french fries",
    "chocolate",
    "biscuit",
    "popcorn",
    "pudding",
    "ice cream",
    "cheese butter",
    "cake",
    "wine",
    "milkshake",
    "coffee",
    "juice",
    "milk",
    "tea",
    "almond",
    "red beans",
    "cashew",
    "dried cranberries",
    "soy",
    "walnut",
    "peanut",
    "egg",
    "apple",
    "date",
    "apricot",
    "avocado",
    "banana",
    "strawberry",
    "cherry",
    "blueberry",
    "raspberry",
    "mango",
    "olives",
    "peach",
    "lemon",
    "pear",
    "fig",
    "pineapple",
    "grape",
    "kiwi",
    "melon",
    "orange",
    "watermelon",
    "steak",
    "pork",
    "chicken duck",
    "sausage",
    "fried meat",
    "lamb",
    "sauce",
    "crab",
    "fish",
    "shellfish",
    "shrimp",
    "soup",
    "bread",
    "corn",
    "hamburg",
    "pizza",
    "hanamaki baozi",
    "wonton dumplings",
    "pasta",
    "noodles",
    "rice",
    "pie",
    "tofu",
    "eggplant",
    "potato",
    "garlic",
    "cauliflower",
    "tomato",
    "kelp",
    "seaweed",
    "spring onion",
    "rape",
    "ginger",
    "okra",
    "lettuce",
    "pumpkin",
    "cucumber",
    "white radish",
    "carrot",
    "asparagus",
    "bamboo shoots",
    "broccoli",
    "celery stick",
    "cilantro mint",
    "snow peas",
    "cabbage",
    "bean sprouts",
    "onion",
    "pepper",
    "green beans",
    "French beans",
    "king oyster mushroom",
    "shiitake",
    "enoki mushroom",
    "oyster mushroom",
    "white button mushroom",
    "salad",
    "other ingredients",
]

QUESTION_ALIASES: Dict[str, List[str]] = {
    "cheese": ["cheese butter"],
    "butter": ["cheese butter"],
    "tomato": ["tomato"],
    "rice": ["rice"],
    "meat": ["steak", "pork", "chicken duck", "sausage", "fried meat", "lamb"],
    "chicken": ["chicken duck"],
    "duck": ["chicken duck"],
    "mushroom": [
        "king oyster mushroom",
        "shiitake",
        "enoki mushroom",
        "oyster mushroom",
        "white button mushroom",
    ],
}


def get_num_classes() -> int:
    """Return the number of foreground ingredient classes."""
    return len(FOODSEG103_INGREDIENTS)


def get_class_names() -> List[str]:
    """Return label names in model output order."""
    return FOODSEG103_INGREDIENTS


def index_to_name(index: int) -> str:
    """Convert a zero-based model output index into a human-readable class name."""
    return FOODSEG103_INGREDIENTS[index]


def name_to_index(name: str) -> int:
    """Look up a class index by exact class name."""
    return FOODSEG103_INGREDIENTS.index(name)


def normalize_text(text: str) -> str:
    """Lowercase and remove extra punctuation for simple keyword matching."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_ingredient_from_question(question: str) -> Optional[str]:
    """Map a free-form yes/no question to one supported ingredient target."""
    normalized_question = normalize_text(question)

    for alias, mapped_labels in QUESTION_ALIASES.items():
        if alias in normalized_question:
            return mapped_labels[0]

    for ingredient_name in FOODSEG103_INGREDIENTS:
        ingredient_tokens = normalize_text(ingredient_name)
        if ingredient_tokens and ingredient_tokens in normalized_question:
            return ingredient_name

    return None


def get_candidate_labels_from_question(question: str) -> Optional[List[str]]:
    """
    Return one or more model labels that should answer the question.

    This helps generic words like "meat" map to several ingredient labels.
    """
    normalized_question = normalize_text(question)

    for alias, mapped_labels in QUESTION_ALIASES.items():
        if alias in normalized_question:
            return mapped_labels

    ingredient_name = extract_ingredient_from_question(question)
    if ingredient_name is None:
        return None
    return [ingredient_name]
