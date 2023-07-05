import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_dif_params

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_path = '../models/debt_financed_tau.yml'

varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'MPC', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l']

param_name = 'psi'
shock = ("e_g", 0.01)
param_vals = [20, 40, 60, 80, 100]
save_dir = "psi"
Palette = 'Blues'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
IRF_list = plot_dif_params(model_path, varlist, shock,
                           param_name, param_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
