import os
import json
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 15
PATIENCE = 4

torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not os.path.isdir(VAL_DIR):
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

class_names = train_dataset.classes

if val_dataset.classes != class_names:
    raise ValueError("Training and validation classes do not match.")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training and validation datasets cannot be empty.")

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

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

for parameter in model.features.parameters():
    parameter.requires_grad = False

for parameter in model.features[-3:].parameters():
    parameter.requires_grad = True

input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(class_names))

for parameter in model.classifier.parameters():
    parameter.requires_grad = True

model = model.to(device)

from collections import Counter

class_counts = Counter(train_dataset.targets)

class_weights = torch.tensor(
    [
        len(train_dataset) / (len(class_names) * class_counts[i])
        for i in range(len(class_names))
    ],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Class weights:", class_weights.cpu().tolist())

optimizer = Adam([
    {"params": model.features[-3:].parameters(), "lr": 0.00001},
    {"params": model.classifier.parameters(), "lr": 0.0001}
])

best_accuracy = -1.0
best_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for phase in ("train", "validation"):
        if phase == "train":
            model.train()
            model.features.eval()
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
            total_correct += (predictions == labels).sum().item()
            total_samples += count

        epoch_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        print(
            f"{phase.title()} loss: {epoch_loss:.4f} | "
            f"accuracy: {accuracy:.1%}"
        )

        if phase == "validation":
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print("Validation accuracy improved.")
            else:
                epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping.")
        break

model.load_state_dict(best_weights)

torch.save(
    model.state_dict(),
    os.path.join(MODEL_DIR, "food_classifier.pth")
)

with open(
    os.path.join(MODEL_DIR, "food_classes.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(class_names, file, indent=4)

print("\nTraining complete!")
print(f"Best validation accuracy: {best_accuracy:.1%}")
print("Neural-network model saved.")