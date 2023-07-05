import jax.numpy as jnp  # use jax.numpy instead of normal numpy
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os

from plotting_functions import plot_transition, plot_wealth_distribution, plot_IRFs_init_cond

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# ['../models/balanced_tau.yml', '../models/balanced_T.yml'] # , '../models/debt_financed_tau.yml'
model_path = '../models/transition_deficit_B.yml'
varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'D', 'revenue', 'Z',
           'B', 'R', 'RBr', 'w', 'tau_l', 'Top10A', 'Top10C']
save_dir = "transition_def_tau_rhog9"

new_states = {'tau_p': 0.132}  # , 'Rstar': 1.03**25

irfs = plot_transition(model_path, varlist, new_states, 50, save_dir)

# model_path = '../models/transition_balanced_B.yml'  #['../models/balanced_tau.yml', '../models/balanced_T.yml'] # , '../models/debt_financed_tau.yml'
# varlist = ['y', 'C', 'G', 'pi', 'n', 'DIV', 'D', 'revenue',
#             'B', 'Rstar', 'R', 'Rr', 'RBr', 'w', 'tau_l',  'T']
# save_dir = "transition_tau_p_B_fixed_moresense"

# new_states = {'tau_p': 0.132} # , 'Rstar': 1.03**25

# irfs = plot_transition(model_path, varlist, new_states, save_dir)
