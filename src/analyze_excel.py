import pandas as pd

# Path to your Excel report
excel_path = "outputs/accuracy_report.xlsx"

# Read the Excel file
df = pd.read_excel(excel_path)

# Print first few rows to see the data
print("First 5 rows:")
print(df.head())

# Calculate mean, max, and min errors per landmark
mean_errors = df.mean()
max_errors = df.max()
min_errors = df.min()

print("\nMean error per landmark:")
print(mean_errors)

print("\nMax error per landmark:")
print(max_errors)

print("\nMin error per landmark:")
print(min_errors)

# Optional: save the summary to a new Excel file
summary_path = "outputs/accuracy_summary.xlsx"
summary = pd.DataFrame({
    "Mean Error": mean_errors,
    "Max Error": max_errors,
    "Min Error": min_errors
})
summary.to_excel(summary_path)
print(f"\nSummary saved at: {summary_path}")
