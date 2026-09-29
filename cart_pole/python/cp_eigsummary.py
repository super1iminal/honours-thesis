import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def generate_eigenvalue_stats_table():
    """
    Generate a statistical summary table for eigenvalues of saturated and unsaturated Jacobians.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the statistics
    """
    # Load the data
    filename = "cp_jacobians.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    sat_Jacobians = data["sat_Jacobians"]
    unsat_Jacobians = data["unsat_Jacobians"]
    
    # Calculate statistics for saturated Jacobians
    sat_real_parts = []
    for J in sat_Jacobians:
        # Get eigenvalues (sorted by real part)
        eigs = get_eigs(J)
        # Get real parts
        real_parts = get_reals(eigs)
        sat_real_parts.extend(real_parts)
    
    sat_real_parts = np.array(sat_real_parts)
    sat_avg = np.mean(sat_real_parts)
    sat_median = np.median(sat_real_parts)
    sat_std = np.std(sat_real_parts)
    sat_min = np.min(sat_real_parts)
    sat_max = np.max(sat_real_parts)
    
    # Calculate statistics for unsaturated Jacobians
    unsat_real_parts = []
    for J in unsat_Jacobians:
        # Get eigenvalues (sorted by real part)
        eigs = get_eigs(J)
        # Get real parts
        real_parts = get_reals(eigs)
        unsat_real_parts.extend(real_parts)
    
    unsat_real_parts = np.array(unsat_real_parts)
    unsat_avg = np.mean(unsat_real_parts)
    unsat_median = np.median(unsat_real_parts)
    unsat_std = np.std(unsat_real_parts)
    unsat_min = np.min(unsat_real_parts)
    unsat_max = np.max(unsat_real_parts)
    
    # Create a DataFrame
    stats_df = pd.DataFrame({
        "Statistic": ["Average", "Median", "Std Deviation", "Min", "Max"],
        "Saturated": [sat_avg, sat_median, sat_std, sat_min, sat_max],
        "Unsaturated": [unsat_avg, unsat_median, unsat_std, unsat_min, unsat_max]
    })
    
    return stats_df

def plot_eigenvalue_stats_table():
    """
    Create a matplotlib plot with a table showing eigenvalue statistics
    for saturated and unsaturated Jacobians.
    """
    stats_df = generate_eigenvalue_stats_table()
    
    # Convert values to strings formatted to 4 decimal places
    formatted_data = stats_df.copy()
    formatted_data["Saturated"] = formatted_data["Saturated"].map(lambda x: f"{x:.4f}")
    formatted_data["Unsaturated"] = formatted_data["Unsaturated"].map(lambda x: f"{x:.4f}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create the table
    table = ax.table(
        cellText=formatted_data.values,
        colLabels=formatted_data.columns,
        loc='center',
        cellLoc='center',
        colColours=['#e6f3ff', '#e6f3ff', '#e6f3ff'],
        rowLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add a title
    plt.title('Eigenvalue Statistics (Real Parts)', fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.savefig('eigenvalue_stats_table.png')
    plt.show()

# Re-using these functions from the original code
def get_eigs(J):
    """Get eigenvalues of a Jacobian matrix, sorted by real part"""
    if hasattr(J, 'detach'):
        # It's a PyTorch tensor
        eigs = np.linalg.eig(J.detach().numpy())[0]
    else:
        # It's already a NumPy array
        eigs = np.linalg.eig(J)[0]
    
    idx = np.argsort(eigs.real)
    return eigs[idx]  # sort eigenvalues

def get_reals(eigs):
    """Get real parts of eigenvalues"""
    return eigs.real

# Run the analysis
if __name__ == "__main__":
    plot_eigenvalue_stats_table()