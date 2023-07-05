import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_IRFs, plot_wealth_distribution, plot_IRFs_init_cond

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

model_paths = ['../models/debt_financed_tau.yml', '../models/good_MPC_debt.yml',
               '../models/div_debt.yml']  # , '../models/debt_financed_tau.yml'
model_names = ['MPC = 11', 'MPC = 0.25', 'MPC = 0.47']
varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'MPC', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l', 'D']
shock = ("e_t", 0.01)
save_dir = "transfers_def"
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
models, IRF_list = plot_IRFs(
    model_paths, model_names, varlist, shock, 50, save_dir)
plot_wealth_distribution(IRF_list, models, save_dir)
