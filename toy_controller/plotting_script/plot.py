import argparse
import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

PLOT_INFO = '../plot_info'
PLOTS = '../plots'

parser = argparse.ArgumentParser(description='Plot sampled states and fitted ellipses for each time step.')
parser.add_argument('-t', type=int, default=None,
                    help='time horizon of the run to plot (plots t = 0 .. t-1). '
                         'Defaults to every time step found in plot_info.')
args = parser.parse_args()

if args.t is not None:
    time_steps = list(range(args.t))
else:
    # Only plot time steps that have both a states file and an ellipses file
    found = [int(re.search(r'states_(\d+)\.txt$', f).group(1))
             for f in glob.glob(os.path.join(PLOT_INFO, 'states_*.txt'))]
    time_steps = sorted(t for t in found
                        if os.path.exists(os.path.join(PLOT_INFO, f'ellipses_{t}.txt')))

os.makedirs(PLOTS, exist_ok=True)

for x in time_steps:
    # states files have no header row
    data_points = pd.read_csv(f'{PLOT_INFO}/states_{x}.txt', header=None, names=['index', 'x', 'v'])
    ellipse_params = pd.read_csv(f'{PLOT_INFO}/ellipses_{x}.txt', header=None, names=['index', 'a', 'b', 'theta'])

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
    plt.savefig(f'{PLOTS}/ellipses_{x}.png')
    plt.close(fig)  # Close the figure to avoid displaying all at once and free memory
