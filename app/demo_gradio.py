"""Simple Gradio app for food ingredient VQA."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gradio as gr
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.food_vqa import answer_food_question


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
DEFAULT_MAYBE_THRESHOLD = 0.20


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def format_top_predictions(top_predictions: list[tuple[str, float]]) -> str:
    return "\n".join(
        f"{ingredient_name}: {probability:.4f}"
        for ingredient_name, probability in top_predictions
    )


def run_demo(image: Image.Image, question: str):
    if image is None:
        return (
            question or "",
            "",
            "Please upload an image first.",
            0.0,
            0.0,
            DEFAULT_MAYBE_THRESHOLD,
            "No predictions yet.",
        )

    if not CHECKPOINT_PATH.exists():
        return (
            question or "",
            "",
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train the model first.",
            0.0,
            0.0,
            DEFAULT_MAYBE_THRESHOLD,
            "No predictions yet.",
        )

    config = load_config()
    yes_threshold = float(config["inference"]["threshold"])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        image.convert("RGB").save(temp_path)
        result = answer_food_question(
            image_path=temp_path,
            question=question,
            checkpoint_path=CHECKPOINT_PATH,
            config=config,
            maybe_threshold=DEFAULT_MAYBE_THRESHOLD,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return (
        question,
        result["target_ingredient"] or "",
        result["answer"],
        result["confidence"],
        result.get("yes_threshold", yes_threshold),
        result.get("maybe_threshold", DEFAULT_MAYBE_THRESHOLD),
        format_top_predictions(result["top_predictions"]),
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Food Ingredient VQA")
        gr.Markdown(
            "Upload a food photo, ask an ingredient query such as tomato, cheese, rice, or egg, and inspect the model's confidence."
        )

        image_input = gr.Image(type="pil", label="Food image")
        question_input = gr.Textbox(
            label="Ingredient query",
            value="tomato",
            placeholder="Enter a simple ingredient query like tomato, cheese, rice, or egg",
        )
        run_button = gr.Button("Ask")

        query_output = gr.Textbox(label="Question / Query")
        ingredient_output = gr.Textbox(label="Target ingredient")
        answer_output = gr.Textbox(label="Answer")
        confidence_output = gr.Number(label="Confidence")
        yes_threshold_output = gr.Number(label="Yes threshold")
        maybe_threshold_output = gr.Number(label="Maybe threshold")
        top_predictions_output = gr.Textbox(label="Top predicted ingredients", lines=8)

        run_button.click(
            fn=run_demo,
            inputs=[image_input, question_input],
            outputs=[
                query_output,
                ingredient_output,
                answer_output,
                confidence_output,
                yes_threshold_output,
                maybe_threshold_output,
                top_predictions_output,
            ],
        )
        question_input.submit(
            fn=run_demo,
            inputs=[image_input, question_input],
            outputs=[
                query_output,
                ingredient_output,
                answer_output,
                confidence_output,
                yes_threshold_output,
                maybe_threshold_output,
                top_predictions_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7861)
