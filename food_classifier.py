import os
import json
import gc

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.pth")
CLASSES_PATH = os.path.join(MODEL_DIR, "food_classes.json")

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device("cpu")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isfile(CLASSES_PATH):
    raise FileNotFoundError(f"Class names not found: {CLASSES_PATH}")

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError("food_classes.json must contain a non-empty list.")

print("Loading MobileNetV3-Small...", flush=True)

model = models.mobilenet_v3_small(weights=None)
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(class_names))

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
del state_dict

model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Neural-network model loaded.", flush=True)
print("Classes:", class_names, flush=True)


def predict_food(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_tensor = None
    outputs = None
    probabilities = None

    try:
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image = image.resize((224, 224))
            image_tensor = image_transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            top_count = min(5, len(class_names))
            scores, indices = torch.topk(probabilities, top_count)

            score_values = scores.tolist()
            index_values = indices.tolist()

        alternatives = [
            {
                "label": class_names[index].replace("_", " "),
                "confidence": round(score * 100, 2)
            }
            for score, index in zip(score_values, index_values)
        ]

        best = alternatives[0]

        result = {
            "prediction": best["label"],
            "confidence": best["confidence"],
            "alternatives": alternatives,
            "uncertain": best["confidence"] < 80
        }

        print("TOP PREDICTIONS:", alternatives, flush=True)
        return result

    finally:
        if image_tensor is not None:
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
        print("\nPrediction result:", predict_food(image_path))
    except Exception as error:
        print("Prediction error:", error)