import os
import json
import gc

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


print("FOOD CLASSIFIER FILE LOADED", flush=True)

# Reduce CPU thread usage to help on low-memory hosting.
torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.pth")
CLASSES_PATH = os.path.join(MODEL_DIR, "food_classes.json")

# Render free instances normally run inference on the CPU.
device = torch.device("cpu")


# Load food category names.
if not os.path.isfile(CLASSES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASSES_PATH}"
    )

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError("food_classes.json must contain a non-empty list.")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


print("Loading trained food model on CPU...", flush=True)

# Build the same model architecture used during training.
model = models.mobilenet_v3_small(weights=None)

input_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    input_features,
    len(class_names)
)

# Load the trained weights directly onto the CPU.
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
del state_dict

model.eval()

# Use the same preprocessing as the training/validation pipeline.
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("TRAINED FOOD MODEL LOADED SUCCESSFULLY", flush=True)
print("Food classes:", class_names, flush=True)


def predict_food(image_path):
    """Predict one of the trained food categories."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_tensor = None
    outputs = None
    probabilities = None

    try:
        # DIP: correct orientation and convert to RGB.
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original)
            image = image.convert("RGB")

            # Resize before creating the tensor.
            image = image.resize((224, 224))

            image_tensor = image_transform(image).unsqueeze(0)

            image.close()

        # Neural-network inference without gradient calculations.
        with torch.inference_mode():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            top_count = min(5, len(class_names))
            scores, indices = torch.topk(
                probabilities,
                top_count
            )

            # Copy the small result list before releasing tensors.
            score_values = scores.tolist()
            index_values = indices.tolist()

        alternatives = []

        for score, index in zip(score_values, index_values):
            alternatives.append({
                "label": class_names[index].replace("_", " "),
                "confidence": round(score * 100, 2)
            })

        best = alternatives[0]

        # Keep uncertain predictions clearly marked.
        uncertain = best["confidence"] < 80

        print("TOP FOOD PREDICTIONS:", alternatives, flush=True)
        print("PREDICTED FOOD:", best["label"], flush=True)
        print("CONFIDENCE:", best["confidence"], flush=True)
        print("PREDICTION UNCERTAIN:", uncertain, flush=True)

        return {
            "prediction": best["label"],
            "confidence": best["confidence"],
            "alternatives": alternatives,
            "uncertain": uncertain
        }

    finally:
        # Release temporary image and inference tensors.
        del image_tensor

        if outputs is not None:
            del outputs

        if probabilities is not None:
            del probabilities

        gc.collect()


if __name__ == "__main__":
    image_path = input(
        "Enter the full path of a food image: "
    ).strip().strip('"')

    try:
        result = predict_food(image_path)
        print("\nFinal result:", result)
    except Exception as error:
        print("Error testing image:", error)