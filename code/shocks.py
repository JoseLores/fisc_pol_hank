import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_shocks

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_path = '../models/debt_shock_T.yml'

varlist = ['y', 'C', 'D', 'G', 'pi', 'n', 'DIV', 'MPC', 'Z', 'R', 'RBr',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l']

shock_name = 'e_d'
shock_vals = [0.0025, 0.005, 0.0075, 0.01, 0.0125]
save_dir = "deficit_shocks_transfers"
Palette = 'Purples'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
IRF_list = plot_shocks(model_path, varlist, shock_name,
                       shock_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
