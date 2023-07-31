import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_dif_params

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_path = '../models/baseline.yml'

varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'Z', 'R', 'piwn',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l', 'D']

##########################################################################

param_name = 'phi_pi'
param_name_latex = r'$\phi_{\pi}$'
shock = ("e_t", 0.01)
param_vals = jnp.arange(0., 2.11, 0.3)
save_dir = "phi_pi_robustness"
Palette = 'Blues'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
IRF_list = plot_dif_params(model_path, varlist, shock,
                           param_name, param_name_latex, param_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)

##########################################################################

param_name = 'gamma_B'
param_name_latex = r'$\gamma_{B}$'
shock = ("e_t", 0.01)
param_vals = jnp.arange(0., 1.21, 0.2)
save_dir = "gamma_B_robustness"
Palette = 'Reds'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
IRF_list = plot_dif_params(model_path, varlist, shock,
                           param_name, param_name_latex, param_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
