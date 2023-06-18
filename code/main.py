import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_IRFs

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_paths = ['../models/final_balanced_transfers.yml']
varlist = ['y', 'C', 'G', 'R', 'Rr', 'Rstar',
           'Top10C', 'Top10A', 'RBr', 'MPC', 'T', 'z']
shock = ("e_z", 0.01)
save_dir = "tech_shock"
plot_IRFs(model_paths, varlist, shock, save_dir)
