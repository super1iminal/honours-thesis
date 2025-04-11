import torch
from torch.autograd import functional as AF
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    if (state[1] < constraints["theta"][0]) or (state[1] > constraints["theta"][1]):
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

def get_J_by_sat(n_x=5, n_xdot=5, n_theta=5, n_thetadot=5):
    states = generate_cp_states(n_x, n_xdot, n_theta, n_thetadot)
    sat_Jacobians = []
    unsat_Jacobians = []
    for i in range(states.shape[0]):
        if is_saturated(states[i]):
            sat_Jacobians.append(get_Jacobian(states[i]))
        else:
            unsat_Jacobians.append(get_Jacobian(states[i]))
    
    return np.array(sat_Jacobians), np.array(unsat_Jacobians) # shape: (n, 4, 4)


def get_specific_summary_stats(Js, n):
    # general data
    J_mean = Js.mean(axis = 0)
    J_std = Js.std(axis = 0)
    
    J_ieig = [-1.0]*n
    
    J_pos = [-1.0]*n
    
    J_spreads = [-1.0]*n # -1 means no data, but should have data for everything
    
    J_seps = [-1.0]*n # -1 means no data, but should have data for everything
    
    
    for i in range(n):
        J = Js[i]
        eigs = get_eigs(J)
        # number of imag eigs
        n_ieig = len(get_cc_pairs(eigs))
        J_ieig[i] = n_ieig
        
        # number of positive real parts:
        reals = get_reals(eigs)
        n_pos = num_pos_reals(reals)
        J_pos[i] = n_pos
        
        # get spread
        J_spreads[i] = real_spread_inv(reals)
        
        # get seperation
        J_seps[i] = real_separation_inv(reals)
        
    return J_mean, J_std, J_ieig, J_pos, J_spreads, J_seps
            
        
# used for pos and ieig
def plot_bar_counts(values, possible_vals, title_str): # make bar chart for how many times a possible val occurs
    counts = [sum(v == pv for v in values) for pv in possible_vals]
    plt.figure()
    plt.bar(range(len(possible_vals)), counts)
    plt.xticks(range(len(possible_vals)), [str(pv) for pv in possible_vals])
    plt.title(title_str)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()


# data is array of n continuous values
# group is array of n group labels
# clustered bar chart
def plot_hist_clustered(data, group, group_vals, nbins, title_str):
    data = np.array(data)
    group = np.array(group)

    # if data is constant or empty, handle gracefully
    if len(data) == 0 or abs(data.max() - data.min()) < 1e-12:
        print(f"No variation or empty data for '{title_str}', skipping plot.")
        return

    # choose bin edges
    nbins = max(1, nbins)
    bin_edges = np.linspace(data.min(), data.max(), nbins+1)

    # count how many items in each bin for each group
    hist_counts = np.zeros((len(group_vals), nbins), dtype=int)

    for idx_g, gv in enumerate(group_vals):
        # subset data for this group value
        subset = data[group == gv]
        counts, _ = np.histogram(subset, bins=bin_edges)
        hist_counts[idx_g, :] = counts

    # now let's do the side-by-side bar chart
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]) / (len(group_vals) + 1)

    plt.figure()
    for idx_g, gv in enumerate(group_vals):
        # shift the x positions for group gv
        shift = -0.5*(len(group_vals)-1)*width + idx_g*width
        xs = bin_centers + shift
        ys = hist_counts[idx_g, :]
        plt.bar(xs, ys, width=width, label=f"group={gv}")

    plt.title(title_str)
    plt.xlabel("Data value")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    
    
def make_summary_stats():
    # setup+
    n_x = 31
    n_xdot = 31
    n_theta = 31
    n_thetadot = 31
    
    # get data, both should be shape (sat_n, 4, 4) and (unsat_n, 4, 4)
    # sat_Jacobians, unsat_Jacobians = get_J_by_sat(n_x, n_xdot, n_theta, n_thetadot)
    
    # saving and loading
    
    #dict_Jacobians = {
    #    "sat_Jacobians": sat_Jacobians,
    #    "unsat_Jacobians": unsat_Jacobians,
    #}
    
    filename = "cp_jacobians.pkl"
    #with open(filename, "wb") as f:
    #    pickle.dump(dict_Jacobians, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    #print("data saved to {0}".format(filename))
    
    # to read:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    sat_Jacobians = data["sat_Jacobians"]
    unsat_Jacobians = data["unsat_Jacobians"]
    
    # analysis
    
    # some basic naming info (ordered):
        
        # sat_ is for saturated data, unsat_ is for unsaturated data
        # g_ is for general data and
        # ieig_ is for number of eigenvalues pairs with imaginary parts (conjugates)
        # pos_ is for number of positive real parts:
    
    
    sat_n = sat_Jacobians.shape[0]
    unsat_n = unsat_Jacobians.shape[0]
    g_n = sat_n + unsat_n
    
    print("g_n:", g_n)
    print("sat_n:", sat_n)
    print("unsat_n:", unsat_n)
        
    if sat_n > 0:
        sat_mean, sat_std, sat_ieig, sat_pos, sat_spreads, sat_seps = get_specific_summary_stats(sat_Jacobians, sat_n)    
        
        print("\n=== SAT ===")
        print("Mean:\n", sat_mean)
        print("Std:\n", sat_std)
        
        plot_bar_counts(sat_ieig, [0,1,2], "SAT: Number of complex conjugate pairs (sat_ieig)")
        plot_bar_counts(sat_pos, [0,1,2,3,4], "SAT: Number of positive real parts (sat_pos)")
        
        possible_ieig = [0,1,2]
        plot_hist_clustered(sat_spreads, sat_ieig, possible_ieig, nbins=10,
                            title_str="SAT: Inverse of Spreads grouped by number of conjugate pairs")
        plot_hist_clustered(sat_seps, sat_ieig, possible_ieig, nbins=10,
                            title_str="SAT: Inverse of Separations grouped by number of conjugate pairs")
        
        # 3) histograms of sat_spreads, sat_seps, grouped by sat_pos in {0,1,2,3,4}
        possible_pos = [0,1,2,3,4]
        plot_hist_clustered(sat_spreads, sat_pos, possible_pos, nbins=10,
                            title_str="SAT: Inverse of Spreads grouped by number of positive real-part eigenvalues")
        plot_hist_clustered(sat_seps, sat_pos, possible_pos, nbins=10,
                            title_str="SAT: Inverse of Separations grouped by number of positive real-part eigenvalues")
        
        

            
    if unsat_n > 0:
        unsat_mean, unsat_std, unsat_ieig, unsat_pos, unsat_spreads, unsat_seps = get_specific_summary_stats(unsat_Jacobians, unsat_n)
        
        print("\n=== UNSAT ===")
        print("Mean:\n", unsat_mean)
        print("Std:\n", unsat_std)
        
        plot_bar_counts(unsat_ieig, [0,1,2], "UNSAT: Number of complex conjugate pairs (unsat_ieig)")
        plot_bar_counts(unsat_pos, [0,1,2,3,4], "UNSAT: Number of positive real parts (unsat_pos)")

        # hist of spreads / seps grouped by unsat_ieig
        possible_ieig = [0,1,2]
        plot_hist_clustered(unsat_spreads, unsat_ieig, possible_ieig, nbins=10,
                            title_str="UNSAT: Inverse of Spreads grouped by number of conjugate pairs")
        plot_hist_clustered(unsat_seps, unsat_ieig, possible_ieig, nbins=10,
                            title_str="UNSAT: Inverse of Separations grouped by number of conjugate pairs")

        # hist of spreads / seps grouped by unsat_pos
        possible_pos = [0,1,2,3,4]
        plot_hist_clustered(unsat_spreads, unsat_pos, possible_pos, nbins=10,
                            title_str="UNSAT: Inverse of Spreads grouped by number of positive real-part eigenvalues")
        plot_hist_clustered(unsat_seps, unsat_pos, possible_pos, nbins=10,
                            title_str="UNSAT: Inverse of Separations grouped by number of positive real-part eigenvalues")
    

#def cp_example(n=20, s0 = [-0.1, 0.01, 0.2, 0.03]):
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

import time

start_time = time.time()
make_summary_stats()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
