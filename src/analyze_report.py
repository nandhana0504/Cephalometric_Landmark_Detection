# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # ---------------- CONFIG ----------------
# excel_path = "outputs/accuracy_report.xlsx"  # path to your Excel report
# output_dir = "outputs"
# os.makedirs(output_dir, exist_ok=True)

# # ---------------- READ EXCEL ----------------
# df = pd.read_excel(excel_path)
# print("First 5 rows of the report:")
# print(df.head())

# # ---------------- CALCULATE STATISTICS ----------------
# mean_errors = df.mean()
# max_errors = df.max()
# overall_mean_error = df.values.flatten().mean()

# print("\nMean error per landmark:")
# print(mean_errors)
# print("\nMaximum error per landmark:")
# print(max_errors)
# print(f"\nOverall mean error across all landmarks: {overall_mean_error:.2f} pixels")

# # ---------------- PLOT ----------------
# plt.figure(figsize=(10,6))
# mean_errors.plot(kind='bar', color='skyblue')
# plt.ylabel("Mean Error (pixels)")
# plt.xlabel("Landmarks")
# plt.title("Average Euclidean Error per Landmark")
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Save plot image
# plot_path = os.path.join(output_dir, "landmark_errors_plot.png")
# plt.savefig(plot_path)
# plt.show()

# print(f"\nPlot saved at: {plot_path}")









import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
excel_path = "outputs/accuracy_report.xlsx"  # path to your Excel report
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------------- READ EXCEL ----------------
df = pd.read_excel(excel_path)
print("First 5 rows of the report:")
print(df.head())

# ---------------- CALCULATE STATISTICS ----------------
mean_errors = df.mean()
max_errors = df.max()
overall_mean_error = df.values.flatten().mean()

print("\nMean error per landmark:")
print(mean_errors)
print("\nMaximum error per landmark:")
print(max_errors)
print(f"\nOverall mean error across all landmarks: {overall_mean_error:.2f} pixels")

# ---------------- HIGHLIGHT TOP 3 ----------------
top3_landmarks = mean_errors.sort_values(ascending=False).head(3)
print("\nTop 3 landmarks with highest mean error:")
print(top3_landmarks)

# ---------------- PLOT ----------------
plt.figure(figsize=(12,6))
bars = plt.bar(mean_errors.index, mean_errors.values, color='skyblue')

# Highlight top 3 landmarks in red
for i, landmark in enumerate(mean_errors.index):
    if landmark in top3_landmarks.index:
        bars[i].set_color('red')

plt.ylabel("Mean Error (pixels)")
plt.xlabel("Landmarks")
plt.title("Average Euclidean Error per Landmark (Top 3 in Red)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot image
plot_path = os.path.join(output_dir, "landmark_errors_plot_top3.png")
plt.savefig(plot_path)
plt.show()

print(f"\nPlot saved at: {plot_path}")
