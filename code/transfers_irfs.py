import jax.numpy as jnp
import econpizza as ep
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_IRFs

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)


model_path = '../models/baseline.yml'

model_specifications = ['Low-MPC', 'Mid-MPC', 'High-MPC']

steady_state_targets = {
    'B': [1.75183783, 0.97122222, 0.26094991],
    'beta': [0.95382926, 0.93880593, 0.90293616],
    'gamma_B': [0.55, 0.4, 0.15]
    # Add more variables as needed
}


varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l']
varnames = ['Output', 'Consumption', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'Post-tax Average Income',
            'Interest Rate', 'Real Bonds', 'Government Revenue', 'Real Wage', 'Transfers', 'Ex-post Interest Rate', 'Average Tax Rate on Labor']
shock = ("e_t", 0.01)
save_dir = "transfers_IRFs"


models, IRF_list = plot_IRFs(
    model_path, model_specifications, steady_state_targets, varlist, varnames, shock, 50, save_dir)


########################################################################

model_path = '../models/baseline.yml'

model_specifications = ['Low-MPC', 'Mid-MPC', 'High-MPC']

steady_state_targets = {
    'B': [1.75183783, 0.97122222, 0.26094991],
    'beta': [0.95382926, 0.93880593, 0.90293616],
    'gamma_B': [0, 0, 0],
    'phi_pi': [0, 0, 0],
    # Add more variables as needed
}


varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'Z', 'R',
           'B', 'revenue', 'w', 'T', 'RBr', 'tau_l']
varnames = ['Output', 'Consumption', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'Post-tax Average Income',
            'Interest Rate', 'Real Bonds', 'Government Revenue', 'Real Wage', 'Transfers', 'Ex-post Interest Rate', 'Average Tax Rate on Labor']
shock = ("e_t", 0.01)
save_dir = "self_financed_deficits"


models, IRF_list = plot_IRFs(
    model_path, model_specifications, steady_state_targets, varlist, varnames, shock, 50, save_dir)
