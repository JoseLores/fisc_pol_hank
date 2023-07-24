import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_transition, plot_consumption_changes_skill, plot_consumption_changes_deciles


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# ['../models/balanced_tau.yml', '../models/balanced_T.yml'] # , '../models/debt_financed_tau.yml'
model_path = '../models/transition_good_MPC.yml'
varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'D', 'revenue', 'Z',
           'B', 'R', 'RBr', 'w', 'tau_l', 'beta', 'T']
varnames = ['Output', 'Consumption', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'Deficit', 'Government Revenue',
            'Post-tax Average Income', 'Bonds', 'Nominal Interest Rate', 'Ex-post Interest Rate', 'Real Wage', 'Average Tax Rate on Labor', 'Discount Factor', 'Transfers']

save_dir = "steady_state_transitions_MPC26"

new_states = {'tau_p': 0.132}  # , 'Rstar': 1.03**25

irfs = plot_transition(model_path, varlist, varnames, new_states, 50, save_dir)

plot_consumption_changes_skill(model_path, new_states, save_dir)

plot_consumption_changes_deciles(model_path, new_states, save_dir)
