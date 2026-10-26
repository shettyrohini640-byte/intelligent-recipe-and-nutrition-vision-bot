import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_PATH = os.path.join(BASE_DIR, "models", "food_classifier.pth")

torch.set_num_threads(1)
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

classes = dataset.classes

model = models.mobilenet_v3_small(weights=None)
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, len(classes))

state = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)
model.load_state_dict(state)
model.to(device)
model.eval()

correct = [0] * len(classes)
total = [0] * len(classes)
confusion = [[0] * len(classes) for _ in classes]

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels.tolist(), predictions.tolist()):
            total[actual] += 1
            confusion[actual][predicted] += 1

            if actual == predicted:
                correct[actual] += 1

overall_correct = sum(correct)
overall_total = sum(total)

print("\nVALIDATION RESULTS")
print("-" * 55)

for i, name in enumerate(classes):
    if total[i]:
        accuracy = correct[i] / total[i] * 100
        print(
            f"{name}: {correct[i]}/{total[i]} correct "
            f"({accuracy:.1f}%)"
        )
    else:
        print(f"{name}: No validation images")

print("-" * 55)
print(
    f"Overall accuracy: {overall_correct}/{overall_total} "
    f"({overall_correct / overall_total * 100:.1f}%)"
)

print("\nCONFUSION TABLE")
print("Rows = actual food; columns = predicted food")
print("Column order:", classes)

for name, row in zip(classes, confusion):
    print(f"{name}: {row}")