import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_transition, plot_consumption_changes_skill, plot_consumption_changes_deciles


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)


model_paths = ['../models/transition_baseline.yml',
               '../models/transition_adjust_tau.yml']

varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'D', 'revenue', 'Z',
           'B', 'R', 'RBr', 'w', 'tau_l', 'T']

varnames = ['Output', 'Consumption', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'Deficit', 'Government Revenue',
            'Post-tax Average Income', 'Real Bonds', 'Nominal Interest Rate', 'Ex-post Interest Rate', 'Real Wage', 'Average Tax Rate on Labor', 'Transfers']

save_dir = ['steady_state_transitions', 'steady_state_transitions_robustness']

new_states = {'tau_p': 0.132}

for i, model in enumerate(model_paths):

    irfs = plot_transition(model, varlist, varnames,
                           new_states, 50, save_dir[i])

    plot_consumption_changes_skill(model, new_states, save_dir[i])

    plot_consumption_changes_deciles(model, new_states, save_dir[i])
