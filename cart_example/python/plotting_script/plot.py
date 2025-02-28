import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Loop through your range
for x in range(0, 5):
    # Load your data
    data_points = pd.read_csv(f'plot_info/states_{x}.txt', skiprows=1, header=None, names=['index', 'x', 'y'])
    ellipse_params = pd.read_csv(f'plot_info/ellipses_{x}.txt', header=None, names=['index', 'a', 'b', 'theta'])

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot data points
    ax.scatter(data_points['x'], data_points['y'], s=1, c='blue', label='Data Points')

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
            lw=2,
            label='Fitted Ellipse' if idx == 0 else ''
        )
        ax.add_patch(ellipse)

    # Set aspect ratio and labels
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Scatter Plot with Fitted Ellipse(s) for t={x}')
    ax.legend()

# Display all plots at once
plt.show()
