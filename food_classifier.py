
import os
import json
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

print("FOOD CLASSIFIER FILE LOADED", flush=True)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.pth")
CLASSES_PATH = os.path.join(MODEL_DIR, "food_classes.json")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the seven food categories
if not os.path.isfile(CLASSES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASSES_PATH}"
    )

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run python train_model.py first."
    )

print("Loading your trained food model...", flush=True)

# Build the same architecture used during training
model = models.mobilenet_v3_small(weights=None)

input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(
    input_features, len(class_names)
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Prepare uploaded images in the same way as validation images
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

print("TRAINED FOOD MODEL LOADED SUCCESSFULLY", flush=True)
print("Food classes:", class_names, flush=True)


def predict_food(image_path):
    """Predict one of the seven food categories."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # DIP: correct image orientation and convert to RGB
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original)
        image = image.convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Neural-network prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_count = min(5, len(class_names))
    scores, indices = torch.topk(probabilities, top_count)

    alternatives = []

    for score, index in zip(scores.tolist(), indices.tolist()):
        alternatives.append({
            "label": class_names[index].replace("_", " "),
            "confidence": round(score * 100, 2)
        })

    best = alternatives[0]
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
if __name__ == "__main__":
    image_path = input("Enter the full path of your Vegetable Pulao image: ").strip().strip('"')

    try:
        result = predict_food(image_path)
        print("\nFinal result:", result)
    except Exception as error:
        print("Error testing image:", error)