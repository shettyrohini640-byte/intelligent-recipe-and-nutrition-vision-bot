
import os
import json
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 15
PATIENCE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. IMAGE TRANSFORMATIONS
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.12
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. LOAD THE DATASETS
# --------------------------------------------------

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(
        f"Validation folder not found: {VAL_DIR}"
    )

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

class_names = train_dataset.classes

if val_dataset.classes != class_names:
    raise ValueError(
        "Training and validation folders must contain "
        "the same class folders."
    )

if len(class_names) < 2:
    raise ValueError(
        "At least two food categories are required."
    )

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError(
        "The training and validation datasets cannot be empty."
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Classes:", class_names)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# 4. LOAD THE PRETRAINED NEURAL NETWORK
# --------------------------------------------------

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

# Freeze the feature layers first.
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Fine-tune the last three feature blocks.
for parameter in model.features[-3:].parameters():
    parameter.requires_grad = True

# Train the classifier.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

# Match the classifier output to our food categories.
input_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    input_features,
    len(class_names)
)

model = model.to(device)


# --------------------------------------------------
# 5. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = Adam([
    {
        "params": model.features[-3:].parameters(),
        "lr": 0.00001
    },
    {
        "params": model.classifier.parameters(),
        "lr": 0.0001
    }
])


# --------------------------------------------------
# 6. TRAINING AND EARLY STOPPING
# --------------------------------------------------

best_accuracy = -1.0
best_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for phase in ("train", "validation"):

        if phase == "train":
            model.train()

            # Keep the frozen feature layers in evaluation mode.
            model.features.eval()

            # Train only the last three feature blocks.
            model.features[-3:].train()

            loader = train_loader

        else:
            model.eval()
            loader = val_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):

                outputs = model(images)
                loss = criterion(outputs, labels)

                predictions = outputs.argmax(dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            count = labels.size(0)

            total_loss += loss.item() * count

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += count

        epoch_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(
            f"{phase.title()} loss: {epoch_loss:.4f} | "
            f"accuracy: {accuracy:.1%}"
        )

        # Save the best-performing validation checkpoint.
        if phase == "validation":

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_weights = copy.deepcopy(
                    model.state_dict()
                )

                epochs_without_improvement = 0

                print("Validation accuracy improved.")

            else:
                epochs_without_improvement += 1

    # This check is outside the phase loop, so it stops
    # the outer training loop correctly.
    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping: validation accuracy "
            "has not improved for "
            f"{PATIENCE} consecutive epochs."
        )

        break


# --------------------------------------------------
# 7. SAVE THE BEST MODEL
# --------------------------------------------------

model.load_state_dict(best_weights)

model_path = os.path.join(
    MODEL_DIR,
    "food_classifier.pth"
)

classes_path = os.path.join(
    MODEL_DIR,
    "food_classes.json"
)

torch.save(model.state_dict(), model_path)

with open(classes_path, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("\nTraining complete!")
print(f"Best validation accuracy: {best_accuracy:.1%}")
print(f"Model saved: {model_path}")
print(f"Classes saved: {classes_path}")
