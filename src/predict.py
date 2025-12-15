import cv2
import torch
import numpy as np
from model import SimpleUNet
from dataset import get_transforms

MODEL_PATH = "models/best_model.pth"
IMAGE_PATH = "data/images/image1.jpg"
NUM_LANDMARKS = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (MUST MATCH TRAINING)
model = SimpleUNet(n_channels=3, n_classes=NUM_LANDMARKS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load image
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Preprocess
transform = get_transforms()
transformed = transform(image=image, keypoints=[])
input_img = transformed["image"].unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    heatmaps = model(input_img)[0].cpu().numpy()

# Extract landmarks
predicted_landmarks = []
for i in range(NUM_LANDMARKS):
    y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
    predicted_landmarks.append((x, y))

# Draw landmarks
output = image.copy()
for x, y in predicted_landmarks:
    cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

cv2.imwrite("output_prediction.jpg", output)
print("Prediction saved as output_prediction.jpg")
