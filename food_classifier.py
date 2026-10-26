import os
import json
import gc

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.onnx")
CLASSES_PATH = os.path.join(MODEL_DIR, "food_classes.json")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

if not os.path.isfile(CLASSES_PATH):
    raise FileNotFoundError(f"Class names not found: {CLASSES_PATH}")

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError("food_classes.json must contain a non-empty list.")

# Limit ONNX Runtime's CPU threads to reduce resource usage.
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

print("Loading ONNX food model...", flush=True)

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded successfully.", flush=True)
print("Classes:", class_names, flush=True)


def predict_food(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_array = None
    input_tensor = None
    outputs = None
    probabilities = None

    try:
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image = image.resize((224, 224))

            image_array = np.asarray(
                image, dtype=np.float32
            ) / 255.0

        # Match the normalization used during training.
        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        )
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        image_array = (image_array - mean) / std

        # Convert HWC image layout to NCHW.
        input_tensor = np.transpose(
            image_array, (2, 0, 1)
        )[None, :, :, :].astype(np.float32)

        outputs = session.run(
            None,
            {input_name: input_tensor}
        )

        logits = outputs[0][0]

        # Stable softmax.
        shifted_logits = logits - np.max(logits)
        exp_values = np.exp(shifted_logits)
        probabilities = exp_values / np.sum(exp_values)

        top_indices = np.argsort(probabilities)[::-1][
            :min(5, len(class_names))
        ]

        alternatives = [
            {
                "label": class_names[int(index)].replace("_", " "),
                "confidence": round(
                    float(probabilities[index]) * 100, 2
                )
            }
            for index in top_indices
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
        del image_array
        del input_tensor
        del outputs
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