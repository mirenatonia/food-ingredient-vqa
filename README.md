# Food Ingredient VQA

This project predicts visible ingredients from food images and answers simple ingredient queries such as:

- tomato
- cheese
- rice
- egg

The model is based on EfficientNet-B0 and was fine-tuned on FoodSeg103.

## Main Idea

Instead of classifying a food image into a single dish category, this project treats food understanding as a multi-label ingredient recognition problem.

Image → ingredient probabilities → ingredient query → Yes / Maybe / No answer

## Dataset

Dataset: FoodSeg103

FoodSeg103 is originally an ingredient segmentation dataset. We converted segmentation labels into image-level multi-label targets.

Segmentation mask → classes_on_image → remove background id 0 → multi-hot vector of length 103

## Model

- Architecture: EfficientNet-B0
- Output: 103 ingredient logits
- Activation: Sigmoid
- Loss: BCEWithLogitsLoss
- Task: Multi-label classification

## Training

Final training:

- Train images: 4983
- Validation images: 1067
- Test images: 1068
- Epochs: 10
- Batch size: 16
- GPU: NVIDIA RTX 3050 Laptop GPU

## Final Results

Threshold selected by validation tuning: 0.35

Final test results:

- Micro F1: 0.5250
- Macro F1: 0.2528
- Precision: 0.5364
- Recall: 0.5141

## Demo

Run the Gradio interface:

```bash
python app/demo_gradio.py
