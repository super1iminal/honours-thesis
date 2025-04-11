import torch
from torch.autograd import functional as AF
import numpy as np
import pickle
import matplotlib.pyplot as plt
from enum import Enum
from collections import defaultdict

class MODE(Enum):
    FAIL = 0    # The state violates a safety constraint.  We won't compute any successors 
                # of such a state.  All states in the remaining modes satisfy the safety
                # constraints.
                
    SAT = 1     # the actuator value is saturated at it's maximum or minimum value
    
    POS = 2     # the Jacobian has at least one eigenvalue with a positive real part.
    
    RNEG = 3    # all eigenvalues of the Jacobian are real and negative.
                
    J2 = 4      # There are two complex-conjugate pairs of eigenvalues,
                # both have negative real parts.
                
    J1 = 5     # all eigenvalues of the Jacobian have negative real parts.
                # There is one complex-conjugate pair of eigenvalues,
                # and two purely real real eigenvalues.
    

def cart_pole_consts():
    return {"g"      : 9.8,   # m/s^2, acceleration of gravity (earth's surface assumed)
            "l_pole" : 1.0,   # m,  length of the pole
            "m_cart" : 1.0,   # kg, mass of the cart
            "m_pole" : 0.1,   # kg, mass of the pole
            "dt"     : 0.05,  # s,  size of time step.
            "f_lo"   : -30.0, # minimum force to apply to cart.  newtons
            "f_hi"   : +30.0, # maximum force to apply to cart.  newtons
            # a system state is a vector: s = [x, x_dot, theta, theta_dot]
            # The controller is parameterized by a vector, w
            "w"      : torch.tensor([0.6234, 1.8060, 34.6404, 11.8123], requires_grad=False)
        }
        
def get_max_start_cart_vel():
    return np.sqrt(
                2.0*
                (abs(cart_pole_consts()["f_hi"])/(cart_pole_consts()["m_cart"]+cart_pole_consts()["m_pole"]))*
                (cart_pole_constraints()["x"][1] - cart_pole_constraints()["x"][0])
            ) # currently around 23m/s

def get_max_start_pole_vel(): # crude approx
    return 2*np.sqrt(
            abs(cart_pole_consts()["f_hi"])/(cart_pole_consts()["m_pole"]*cart_pole_consts()["l_pole"])
        )

def cart_pole_constraints():
    return {"x"      : np.array([-5, 5]),
            "theta"  : np.array([-1, 1]), 
        }

def implied_cp_constraints():
    x_dot = np.array([-get_max_start_cart_vel(), get_max_start_cart_vel()])
    theta_dot = np.array([-get_max_start_pole_vel(), get_max_start_pole_vel()])
    return {"x_dot"      : x_dot,
            "theta_dot"  : theta_dot, 
        }

        

def float_thresh():
    return 1e-9

def violates_constraints(state: torch.tensor):
    constraints = cart_pole_constraints()
    if (state[0] < constraints["x"][0]) or (state[0] > constraints["x"][1]):
        return True
    if (state[2] < constraints["theta"][0]) or (state[2] > constraints["theta"][1]):
        return True
    return False


def cp_step(state: torch.tensor):  # s is a cart-pole state, should have requires_grad=True for torch to track all operations
    x, x_dot, theta, theta_dot = state
    cp_consts = cart_pole_consts()
    [g, l_pole, m_cart, m_pole, dt, f_lo, f_hi, w] = \
            [ cp_consts[what] for what in
              ["g", "l_pole", "m_cart", "m_pole", "dt", "f_lo", "f_hi", "w"]]

    u = torch.dot(w, state)
    f = min(max(u, f_lo), f_hi)  # force applied to the cart
    
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    th_dot_2 = theta_dot * theta_dot
    
    m_eff = m_cart + m_pole * sin_th * sin_th
    
    ddxdtt = (f + sin_th*m_pole*(l_pole * th_dot_2 - g * cos_th)) / m_eff
    ddthdtt = (
        -f * cos_th
        - m_pole*l_pole*th_dot_2*sin_th*cos_th
        + sin_th*g*(m_cart + m_pole)
    ) / (l_pole * m_eff)
    
    ddt = torch.stack([x_dot, ddxdtt, theta_dot, ddthdtt])
        
    next_state = state + ddt * dt
    
    return next_state, f

def ode_step(state: torch.tensor):  # s is a cart-pole state, should have requires_grad=True for torch to track all operations
    x, x_dot, theta, theta_dot = state
    cp_consts = cart_pole_consts()
    [g, l_pole, m_cart, m_pole, dt, f_lo, f_hi, w] = \
            [ cp_consts[what] for what in
              ["g", "l_pole", "m_cart", "m_pole", "dt", "f_lo", "f_hi", "w"]]

    u = torch.dot(w, state)
    f = min(max(u, f_lo), f_hi)  # force applied to the cart
    
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    th_dot_2 = theta_dot * theta_dot
    
    m_eff = m_cart + m_pole * sin_th * sin_th
    
    ddxdtt = (f + sin_th*m_pole*(l_pole * th_dot_2 - g * cos_th)) / m_eff
    ddthdtt = (
        -f * cos_th
        - m_pole*l_pole*th_dot_2*sin_th*cos_th
        + sin_th*g*(m_cart + m_pole)
    ) / (l_pole * m_eff)
    
    ddt = torch.stack([x_dot, ddxdtt, theta_dot, ddthdtt])
    
    return ddt

def get_Jacobian(state: torch.tensor):
    return AF.jacobian(ode_step, state)

def is_saturated(state: torch.tensor):
    _, f = cp_step(state)
    cp_consts = cart_pole_consts()
    [f_lo, f_hi] = [cp_consts[what] for what in ["f_lo", "f_hi"]]
    if (f == f_lo) or (f == f_hi):
        return True
    return False

def get_eigs(J: torch.tensor):
    if hasattr(J, 'detach'):
        # It's a PyTorch tensor
        eigs = np.linalg.eig(J.detach().numpy())[0]
    else:
        # It's already a NumPy array
        eigs = np.linalg.eig(J)[0]
    
    idx = np.argsort(eigs.real)
    return eigs[idx]  # sort eigenvalues

def get_cc_pairs(eigs: np.ndarray) -> np.ndarray:
    used = set()
    cc_pairs = []
    n = len(eigs)
    for i in range(n):
        if i in used:
            continue
        eig = eigs[i]
        if abs(eig.imag) < float_thresh():
            continue
        
        for j in range(i + 1, n):
            if j in used:
                continue
            other = eigs[j]
            
            if abs(other.imag) < float_thresh():
                continue
            
            if (abs(eig.real - other.real) < float_thresh()) and (abs(eig.imag + other.imag) < float_thresh()):
                cc_pairs.append([eig, other])
                used.add(i)
                used.add(j)
                break
    return cc_pairs
    
    
def get_reals(eigs: np.ndarray):
    return eigs.real

# returns sorted list of absolute values of array, ascending
def abs_sort(arr: np.ndarray): # helper
    return np.sort(np.abs(arr))

def real_spread_inv(reals: np.ndarray):
    abs_sort_reals = abs_sort(reals)
    if (abs_sort_reals[-1] < float_thresh()):
        return 0.0 
    else:
        return abs_sort_reals[0]/abs_sort_reals[-1] # smallest divided by largest (to avoid div by 0 errors)

def real_separation_inv(reals: np.ndarray):
    abs_sort_reals = abs_sort(reals)
    if (abs_sort_reals[1] < float_thresh()):
        return 0.0 
    else:
        return abs_sort_reals[0]/abs_sort_reals[1] # smallest divided by second smallest (to avoid div by 0 errors)
    
def num_pos_reals(reals: np.ndarray):
    count = 0
    for real in reals:
        if real > 0.0:
            count += 1
    return count
    
def generate_cp_states(n_x=5, n_xdot=5, n_theta=5, n_thetadot=5): # 4D grid of states, n is number of grid divisions
    c = cart_pole_constraints()
    ic = implied_cp_constraints()
    
    # (start, end, num divisions)
    x_vals       = np.linspace(c["x"][0],          c["x"][1],          n_x)
    xdot_vals    = np.linspace(ic["x_dot"][0],     ic["x_dot"][1],     n_xdot)
    theta_vals   = np.linspace(c["theta"][0],      c["theta"][1],      n_theta)
    thetadot_vals   = np.linspace(ic["theta_dot"][0], ic["theta_dot"][1], n_thetadot)
    
    # cartesian product
    X, X_dot, Theta, Theta_dot = np.meshgrid(x_vals, xdot_vals, theta_vals, thetadot_vals, indexing = 'ij')
    
    # flatten each dim
    X_f = X.ravel()
    X_dot_f = X_dot.ravel()
    Theta_f = Theta.ravel()
    Theta_dot_f = Theta_dot.ravel()
    
    # stack into shape (N, 4)
    grid_np = np.vstack([X_f, X_dot_f, Theta_f, Theta_dot_f]).T
    
    grid_torch = torch.from_numpy(grid_np).float()
    return grid_torch # each row is a state

def get_J(n_x=5, n_xdot=5, n_theta=5, n_thetadot=5):
    states = generate_cp_states(n_x, n_xdot, n_theta, n_thetadot)
    Jacobians = []
    for i in range(states.shape[0]):
        Jacobians.append(get_Jacobian(states[i]))
    
    return states, Jacobians

def classify_state(state, J):
    if violates_constraints(state):
        return MODE.FAIL
    
    if is_saturated(state):
        return MODE.SAT
    
    eigs = get_eigs(J)
    
    reals = get_reals(eigs)
    
    if (num_pos_reals(reals) > 0):
        return MODE.POS
    
    num_ccs = len(get_cc_pairs(eigs))
    
    if (num_ccs == 0):
        return MODE.RNEG
    
    if (num_ccs == 1):
        return MODE.J1
    
    if (num_ccs == 2):
        return MODE.J2

# storage unit is just a funny name
def save_storage_unit(stoage_unit: dict, filename: str = "storage_unit.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(stoage_unit, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_storage_unit(filename: str = "storage_unit.pkl"):
    with open(filename, "rb") as f:
        storage_unit = pickle.load(f)
    return storage_unit
    
def make_storage_unit(n_x = 5, n_xdot = 5, n_theta = 5, n_thetadot = 5):
    
    states, Jacobians = get_J(n_x, n_xdot, n_theta, n_thetadot)
    states_class = [classify_state(state, J) for (state, J) in zip(states, Jacobians)]
    
    # sort all based on mode (class)
    class_values = np.array([mode.value for mode in states_class])
    idx = np.argsort(class_values)
    
    states_class = np.array(states_class)[idx]
    states = torch.from_numpy(np.array(states)[idx])
    Jacobians = np.array(Jacobians)[idx]
    
    # store all states over time
    states_over_time = [[(state, c)] for state, c in zip(states, states_class)]
    
    # copy to store starting states, reduandant due to states_over_time, but still potentially useful
    start_states = states.clone()
    start_states_class = states_class.copy()
    
    # find possible transitions and store all state progressions
    possible_transitions = defaultdict(list)
    
    for i in range(states.shape[0]):
        count = 0
        if (states_class[i] == MODE.FAIL):
            possible_transitions[(MODE.FAIL, MODE.FAIL)].append(1)
            continue
        for j in range(20):
            count += 1
            next_state, _ = cp_step(states[i])
            next_J = get_Jacobian(next_state)
            next_class = classify_state(next_state, next_J)
            states_over_time[i].append((next_state.clone(), next_class))
            if next_class != states_class[i]:
                possible_transitions[(states_class[i], next_class)].append((states[i].clone(), next_state.clone(), count))
                states_class[i] = next_class
                count = 0
                if next_class == MODE.FAIL:
                    states[i] = next_state
                    Jacobians[i] = next_J
                    break
            states[i] = next_state
            Jacobians[i] = next_J
    
    
    storage_unit = {
        "start_states": start_states,
        "end_states": states,
        "start_modes": start_states_class,
        "end_modes": states_class,
        "states_over_time": states_over_time,
        "possible_transitions": possible_transitions
    }
    return storage_unit

def data_processing(load:bool = False, save:bool = False, filename = "storage_unit.pkl"):    
    if load and save:
        exit(-1)
    
    if load:
        storage_unit = load_storage_unit(filename)
    else:
        n_x = 10
        n_xdot = 10
        n_theta = 10
        n_thetadot = 10
        storage_unit = make_storage_unit(n_x, n_xdot, n_theta, n_thetadot)
        if save:
            save_storage_unit(storage_unit, filename)
            
    [start_states, end_states, start_states_modes, end_states_modes, states_over_time, possible_transitions] = \
            [ storage_unit[what] for what in
              ["start_states", "end_states", "start_modes", "end_modes", "states_over_time", "possible_transitions"]]
    
    plot_mode_distribution(start_states_modes, end_states_modes)
    
    plot_transition_stats_table(possible_transitions)
    
    eig_data = [get_eigs(get_Jacobian(state)) for state in end_states[np.where(end_states_modes == MODE.J1)]]
    sorted_eig_data = np.array(sorted(eig_data, key=lambda x: abs(x[0].real), reverse=True))
    plot_eigenvalues(sorted_eig_data, title="J1 Eigenvalues")
    plt.show()
    
    
def plot_eigenvalues(eig_data, title="Eigenvalues on Complex Plane", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Process each state's eigenvalues
    for eigs in eig_data:
        # Convert to complex numbers if they're not already
        eigs_array = np.array(eigs, dtype=complex)
        
        # Sort real and complex eigenvalues
        real_eigs = []
        complex_eigs = []
        
        for eig in eigs_array:
            if abs(eig.imag) < float_thresh():
                real_eigs.append(eig)
            else:
                complex_eigs.append(eig)
        
        # Sort real eigenvalues by magnitude (largest first)
        real_eigs.sort(key=lambda x: abs(x.real), reverse=True)
        
        # Plot the larger magnitude real eigenvalue in green
        if len(real_eigs) > 0:
            ax.scatter(real_eigs[0].real, real_eigs[0].imag, 
                      color='green', 
                      marker='x',
                      s=40,  # Smaller point size
                      alpha=0.8)
            
        # Plot the smaller magnitude real eigenvalue in red
        if len(real_eigs) > 1:
            ax.scatter(real_eigs[1].real, real_eigs[1].imag, 
                      color='red', 
                      marker='x',
                      s=40,  # Smaller point size
                      alpha=0.8)
        
        # Plot complex eigenvalues in black
        for eig in complex_eigs:
            ax.scatter(eig.real, eig.imag, 
                      color='black', 
                      marker='x',
                      s=40,  # Smaller point size
                      alpha=0.8)
    
    # Draw the real and imaginary axes
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', 
                   markeredgecolor='green', markersize=10, label='Larger magnitude real eigenvalue'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', 
                   markeredgecolor='red', markersize=10, label='Smaller magnitude real eigenvalue'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black', 
                   markeredgecolor='black', markersize=10, label='Complex eigenvalues')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_mode_distribution(start_states_modes, end_states_modes, title="Mode Distribution"):
    """
    Creates a bar chart showing the number of states in each mode at the beginning and end.
    
    Args:
        start_states_modes: Array of MODE enums for starting states
        end_states_modes: Array of MODE enums for ending states
        title: Plot title
        
    Returns:
        The matplotlib figure object
    """
    # Count occurrences of each mode
    modes = list(MODE)
    start_counts = [np.sum(start_states_modes == mode) for mode in modes]
    end_counts = [np.sum(end_states_modes == mode) for mode in modes]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create positions for the bars
    x = np.arange(len(modes))
    width = 0.35
    
    # Create the bars
    ax.bar(x - width/2, start_counts, width, label='Initial States', color='skyblue')
    ax.bar(x + width/2, end_counts, width, label='Final States', color='orange')
    
    # Add labels and title
    ax.set_xlabel('Mode', fontsize=12)
    ax.set_ylabel('Number of States', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([mode.name for mode in modes])
    ax.legend()
    
    # Add values on top of each bar
    for i, v in enumerate(start_counts):
        ax.text(i - width/2, v + 0.5, str(v), ha='center')
    for i, v in enumerate(end_counts):
        ax.text(i + width/2, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    return fig


def plot_transition_stats_table(possible_transitions, title="Transition Statistics"):
    """
    Creates a table showing statistics for each transition.
    
    Args:
        possible_transitions: Dictionary of transitions and their data
        title: Table title
        
    Returns:
        The matplotlib figure object
    """
    # Get sorted transitions
    keys = list(possible_transitions.keys())
    idx = np.argsort(np.array([k[0].value for k in keys]))
    pt_keys_sorted = [keys[i] for i in idx]
    
    # Prepare data for the table
    rows = []
    
    for pt in pt_keys_sorted:
        pt_data = np.array([info[2] for info in possible_transitions[pt]])
        
        row = [
            f"{pt[0].name} → {pt[1].name}",
            f"{pt_data.shape[0]}",
            f"{np.min(pt_data)}",
            f"{np.max(pt_data)}",
            f"{np.average(pt_data):.2f}",
            f"{np.std(pt_data):.2f}",
            f"{np.median(pt_data)}"
        ]
        rows.append(row)
    
    # Calculate appropriate figure height based on number of rows
    fig_height = max(6, len(rows) * 0.5 + 2)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    
    # Column headers
    col_labels = ['Transition', 'Count', 'Min Steps', 'Max Steps', 'Avg Steps', 'Std Dev', 'Median Steps']
    
    # Create the table
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15],
        colColours=['#f0f0f0'] * len(col_labels)
    )
    
    # Set title
    ax.set_title(title, pad=20, fontsize=14)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
    
    plt.tight_layout()
    return fig

# def cp_example(n=20, s0 = [-0.1, 0.01, 0.2, 0.03]):
#    s = s0
#    for i in range(n):
#        s, f = cp_step(s)
#        print("{0:3d}: f={1:6.4f}, s = [{2:6.4f}, {3:6.4f}, {4:6.4f}, {5:6.4f}]".
#              format(i, f, s[0], s[1], s[2], s[3]))
#    return s

# 1. 
# saturated -> unsaturated/fail in bounded number of steps? probably, histogram
# once unsaturated, do you ever get back into saturated? proably not, we can check

# eigs w positive real parts -> all negative/fail? in bounded number of steps?
# vice versa but also get to saturated?

# 2.
# visualize 2 conjugate (negative) -> origin (1 conjugate, 2 reals), by eig similarity
# if cant differentiate. split time step in 2
# when meeting at real axis, cant do this, because equidistant
# arrows between

# 3.
# find connected set of 2 real, 1 conjugate, all neg, unsaturated (R0)
# fine set of 100x100x100x100
# each block can hold a 16x16x16x16 (or whatever fits in memory) block
# flood fill algo, lineraze indices, 
# for each set of 4 find if its got the desired props
# also for neighbors
# set to smallest index




# newton step on long pos solves
# multiply deriv by inverse of jacobian
# if deriv is 0, then no lyapunov function
# could also sample in a bounding box of all 12 pos
# then resample on longest ones
# first check if it remains in POS


# intro that's a description of area
# paragraph or 2 to tell the reader what we're doin
# challenge: work in the real world
# most of introudction from thesis
# if we could establish this, it would give a much

# one claim: some of these problems benefit from large-scale simulation
# two codebases


# providing insight into characterizing state space

# three contributions that i would give that would establish the main claim

# possibly related work

# conclusions and future work, dumping ground


# can we get to the subset of J1 that converge to origin
# subset: eig near -7, -2, and complex conjugate
# looking over all J1_C ^, find mean and std from -7, -2, then do the same for complex conjugate pair
  # project one eigenvector into another jac's basis, see how far it is from unit
# make sure its close


import time

start_time = time.time()
data_processing(load=True)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
