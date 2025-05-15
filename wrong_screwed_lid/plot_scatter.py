import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Load your dataframe
df = pd.read_excel("wrong_screwed_lid/data/inner_circle_df.xlsx")  

# Metrics to analyze
metrics = ['ellipse_area', 'area_ratio', 'ellipse_area2', 'area_ratio2', 'diff', 'angle']

# Directory to save the plots
output_folder = "wrong_screwed_lid/data3/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Degree of polynomial
poly_degree = 2

# Plot scatter for each metric against `centre_y`
for metric in metrics:
    plt.figure(figsize=(8, 6))

    # Original data points (blue)
    plt.scatter(df['centre_y'], df[metric], color='blue', edgecolors='black', alpha=0.5, label="Original Data")

    # Fit a polynomial curve
    coeffs = np.polyfit(df['centre_y'], df[metric], poly_degree)  # Fit polynomial
    poly_func = np.poly1d(coeffs)  # Create polynomial function

    # Generate smooth curve values
    x_curve = np.linspace(df['centre_y'].min(), df['centre_y'].max(), 100)
    y_curve = poly_func(x_curve)

    # Plot polynomial curve
    plt.plot(x_curve, y_curve, color='green', linewidth=2, label=f"Polynomial Fit (Degree {poly_degree})")

    # Labels and title
    plt.title(f"Scatter Plot of {metric} vs. centre_y")
    plt.xlabel('centre_y')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_folder, f"{metric}_vs_centre_y.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot for {metric} vs centre_y at: {plot_path}")
