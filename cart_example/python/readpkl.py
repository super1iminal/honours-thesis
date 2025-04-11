import pickle
import numpy as np

filename = "cp_jacobians.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)
    sat_Jacobians = data["sat_Jacobians"] # np.array, shape sat_n, 4, 4
    unsat_Jacobians = data["unsat_Jacobians"] # np.array, shape unsat_n, 4, 4
    
print("Number of saturated sampels: ", sat_Jacobians.shape)
print("Number of unsaturated samples: ", unsat_Jacobians.shape)
