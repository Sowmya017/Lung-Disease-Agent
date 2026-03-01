import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ==========================
# Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR = "formatted_dataset"
BATCH_SIZE = 16
EPOCHS = 12
PATIENCE = 3

# ==========================
# Transforms (Medical Safe)
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ==========================
# Weighted Random Sampler
# ==========================
labels = [label for _, label in train_dataset]
class_sample_count = np.array(
    [len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)]
)

print("Class Distribution:", class_sample_count)

weights = 1. / class_sample_count
samples_weight = np.array([weights[t] for t in labels])
samples_weight = torch.from_numpy(samples_weight).double()

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# Model: EfficientNet-B0
# ==========================
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

# Freeze backbone first
for param in model.features.parameters():
    param.requires_grad = False

model.to(device)

# ==========================
# FOCAL LOSS
# ==========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, factor=0.5
)

best_auc = 0
counter = 0

# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(EPOCHS):

    # Unfreeze after 4 epochs
    if epoch == 4:
        print("Unfreezing backbone layers...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    train_loss = 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")

    # ==========================
    # VALIDATION WITH TTA
    # ==========================
    model.eval()
    all_probs = []
    true = []

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            # TTA: original + flipped
            outputs1 = model(images)
            outputs2 = model(torch.flip(images, dims=[3]))

            probs1 = torch.softmax(outputs1, dim=1)
            probs2 = torch.softmax(outputs2, dim=1)

            probs = (probs1 + probs2) / 2

            all_probs.extend(probs.cpu().numpy())
            true.extend(labels_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    preds = np.argmax(all_probs, axis=1)

    acc = np.mean(preds == np.array(true))

    # AUC (multi-class)
    try:
        auc = roc_auc_score(true, all_probs, multi_class="ovr")
    except:
        auc = 0

    print(f"Validation Accuracy: {acc*100:.2f}%")
    print(f"Validation AUC: {auc:.4f}")

    scheduler.step(auc)

    # Early stopping on AUC
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_lung_model.pth")
        counter = 0
        print("Best model saved (based on AUC).")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ==========================
# FINAL EVALUATION
# ==========================
print("\nLoading best saved model...\n")

model.load_state_dict(torch.load("best_lung_model.pth"))
model.eval()

all_probs = []
true = []

with torch.no_grad():
    for images, labels_batch in val_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        true.extend(labels_batch.cpu().numpy())

all_probs = np.array(all_probs)
preds = np.argmax(all_probs, axis=1)

acc = np.mean(preds == np.array(true))
auc = roc_auc_score(true, all_probs, multi_class="ovr")

print(f"Best Model Accuracy: {acc*100:.2f}%")
print(f"Best Model AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(true, preds, target_names=train_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(true, preds))

print("\nTraining Complete.")
print("Best AUC Achieved:", best_auc)