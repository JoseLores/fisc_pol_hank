import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_shocks

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_path = '../models/balanced_tau.yml'

varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'MPC', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l']

shock_name = 'e_g'
shock_vals = [0.01, 0.02, 0.03, 0.04, 0.05]
save_dir = "shocks_try"
Palette = 'Blues'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
IRF_list = plot_shocks(model_path, varlist, shock_name,
                       shock_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
