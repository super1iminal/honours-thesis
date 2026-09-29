import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Loop through your range
for x in range(0, 6):
    data_points = pd.read_csv(f'../plot_info/states_{x}.txt', skiprows=1, header=None, names=['index', 'x', 'v'])
    ellipse_params = pd.read_csv(f'../plot_info/ellipses_{x}.txt', header=None, names=['index', 'a', 'b', 'theta'])
    
    # Create figure and axis with fixed size for consistent output
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot data points
    ax.scatter(data_points['x'], data_points['v'], s=1, c='blue', label='Data Points')
    
    # Plot ellipses
    for idx, row in ellipse_params.iterrows():
        a = row['a']
        b = row['b']
        theta = row['theta']
        ellipse = Ellipse(
            (0, 0),
            width=2 * a,
            height=2 * b,
            angle=np.degrees(theta),
            edgecolor='red',
            facecolor='none',
            lw=1,  # Reduced thickness from 2 to 1
            label='Fitted Ellipse' if idx == 0 else ''
        )
        ax.add_patch(ellipse)
    
    # Set aspect ratio and labels
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_title(f'Fitted Ellipses for t={x}')  # Updated title format
    ax.legend()
    
    # Save the figure with the requested filename
    plt.savefig(f'../plots/ellipses_{x}.png')
    plt.close(fig)  # Close the figure to avoid displaying all at once and free memory