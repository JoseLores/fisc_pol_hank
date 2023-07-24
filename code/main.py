import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_IRFs, plot_wealth_distribution, plot_IRFs_init_cond

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# model_paths = ['../models/balanced_tau.yml', '../models/balanced_T.yml',
#    '../models/debt_financed_tau.yml']  # , '../models/debt_financed_tau.yml'


# model_names = ['Labor taxes financed', 'Transfers Financed', 'Debt Financed']

####################################################################################

# model_paths = ['../models/transition_deficit_B.yml']
# model_names = ['Deficit Financed']

############################################################################################

model_paths = ['../models/debt_financed_tau.yml', '../models/good_MPC_debt.yml',
               '../models/div_debt.yml']  # , '../models/debt_financed_tau.yml'


model_names = ['MPC = 15', 'MPC = 26', 'MPC = 40']

varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l', 'D']
varnames = ['Output', 'Consumption', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'Post-tax Average Income',
            'Interest Rate', 'Bonds', 'Government Revenue', 'Real Wage', 'Transfers', 'Ex-post Interest Rate', 'Average Tax Rate on Labor', 'Deficit']
shock = ("e_t", 0.01)
save_dir = "transfers_def"
# Palette = 'Blues'
# models, IRF_list = plot_IRFs(model_paths, varlist, shock, save_dir)
# initial_conditions = {'B': 1.001, 'G': 1.01}
models, IRF_list = plot_IRFs(
    model_paths, model_names, varlist, varnames, shock, 50, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
