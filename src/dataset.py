import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy")
    )


class CephDataset(Dataset):
    def __init__(self, images_dir, annots_dir, transform=None, num_landmarks=19):
        self.images_dir = images_dir
        self.annots_dir = annots_dir
        self.transform = transform
        self.num_landmarks = num_landmarks

        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(
            self.annots_dir,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        # Load image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Load landmarks
        landmarks = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x = float(parts[1])
                y = float(parts[2])
                x = min(max(x, 0), w - 1)
                y = min(max(y, 0), h - 1)
                landmarks.append([x, y])

        landmarks = landmarks[:self.num_landmarks]

        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks)
            image = transformed["image"]
            landmarks = transformed["keypoints"]

        heatmaps = np.zeros((self.num_landmarks, 256, 256), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks):
            heatmaps[i] = generate_gaussian_heatmap(
                256, 256, x, y, sigma=4
    )

        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return image, torch.tensor(heatmaps)

def generate_gaussian_heatmap(h, w, cx, cy, sigma=4):
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)
    y = y[:, np.newaxis]

    heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap


