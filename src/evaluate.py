import os
import cv2
from utils import read_landmarks, denormalize_landmarks, euclidean_distance, generate_excel_report
import torch
from model import SimpleUNet
from dataset import get_transforms
import numpy as np

# ---------------- CONFIG ----------------
images_dir = "data/images"
annots_dir = "data/annotations"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

model_path = "models/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_landmarks = 19  # change according to your dataset

# --------------- MODEL ------------------
model = SimpleUNet(n_channels=3, n_classes=num_landmarks)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --------------- TRANSFORM ----------------
transform = get_transforms()

# --------------- EVALUATION ----------------
all_errors = []

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(images_dir, img_file)
    annot_path = os.path.join(annots_dir, os.path.splitext(img_file)[0] + ".txt")

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    H, W = image.shape

    # Transform
    import albumentations as A
    augmented = transform(image=image_color)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_heatmaps = model(img_tensor)
        pred_heatmaps = pred_heatmaps.squeeze(0).cpu().numpy()

    # Convert heatmaps to landmark coordinates
    pred_landmarks = {}
    for i in range(pred_heatmaps.shape[0]):
        heatmap = pred_heatmaps[i]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        pred_landmarks[f"L{i+1}"] = (x / heatmap.shape[1], y / heatmap.shape[0])

    pred_landmarks = denormalize_landmarks(pred_landmarks, W, H)

    # Load GT landmarks
    gt_landmarks = read_landmarks(annot_path)

    # Compute Euclidean errors
    errors = []
    for key in gt_landmarks.keys():
        gt = gt_landmarks[key]
        pred = pred_landmarks[key]
        errors.append(euclidean_distance(gt, pred))
    all_errors.append(errors)

    # Save combined image
    from utils import save_comparison_image
    output_path = os.path.join(output_dir, img_file)
    save_comparison_image(image_color, gt_landmarks, pred_landmarks, output_path)

# Generate Excel report
landmark_names = [f"L{i+1}" for i in range(num_landmarks)]
output_excel = os.path.join(output_dir, "accuracy_report.xlsx")
generate_excel_report(all_errors, landmark_names, output_excel)

print(f"Evaluation completed! Report saved at {output_excel}")
