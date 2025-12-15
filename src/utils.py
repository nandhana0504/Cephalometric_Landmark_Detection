import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_landmarks(txt_path):
    """
    Reads landmark coordinates from a text file.
    Expected format per line:
    landmark_name x y
    """
    landmarks = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                name, x, y = parts
                landmarks[name] = (float(x), float(y))
    return landmarks


def normalize_landmarks(landmarks, image_width, image_height):
    """
    Normalize landmark coordinates to range [0, 1]
    """
    normalized = {}
    for name, (x, y) in landmarks.items():
        normalized[name] = (x / image_width, y / image_height)
    return normalized


def denormalize_landmarks(landmarks, image_width, image_height):
    """
    Convert normalized landmarks back to pixel coordinates
    """
    denorm = {}
    for name, (x, y) in landmarks.items():
        denorm[name] = (x * image_width, y * image_height)
    return denorm


def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    """
    Draw landmarks on an image
    """
    img = image.copy()
    for _, (x, y) in landmarks.items():
        cv2.circle(img, (int(x), int(y)), 3, color, -1)
    return img


def euclidean_distance(p1, p2):
    """
    Compute Euclidean distance between two points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def save_comparison_image(
    image,
    gt_landmarks,
    pred_landmarks,
    output_path
):
    """
    Save a combined image:
    Left: Ground truth landmarks
    Right: Predicted landmarks
    """
    gt_img = draw_landmarks(image, gt_landmarks, color=(0, 255, 0))
    pred_img = draw_landmarks(image, pred_landmarks, color=(0, 0, 255))

    combined = np.hstack([gt_img, pred_img])
    cv2.imwrite(output_path, combined)


def generate_excel_report(
    results,
    landmark_names,
    output_excel_path
):
    """
    Generate Excel report of landmark errors (mm)
    Highlight errors > 2 mm
    """
    df = pd.DataFrame(results, columns=landmark_names)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Accuracy")

        worksheet = writer.sheets["Accuracy"]
        for row in range(2, len(df) + 2):
            for col in range(1, len(landmark_names) + 1):
                cell = worksheet.cell(row=row, column=col)
                if cell.value is not None and cell.value > 2:
                    cell.font = cell.font.copy(color="FF0000")
