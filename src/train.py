import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CephDataset, get_transforms
from model import SimpleUNet



# ---------------- CONFIG ----------------
images_dir = "data/images"
annots_dir = "data/annotations"
output_model_dir = "models"
os.makedirs(output_model_dir, exist_ok=True)

num_epochs = 50
batch_size = 4
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_landmarks = 19  # change according to your dataset

# --------------- DATASET ----------------
train_dataset = CephDataset(
    images_dir,
    annots_dir,
    transform=get_transforms(),
    num_landmarks=num_landmarks
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# --------------- MODEL ------------------
model = SimpleUNet(n_channels=3, n_classes=num_landmarks)

model = model.to(device)

# --------------- LOSS & OPTIMIZER --------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --------------- TRAINING ----------------
best_loss = float("inf")
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for images, heatmaps in train_loader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(output_model_dir, "best_model.pth"))
        print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")

print("Training completed!")
