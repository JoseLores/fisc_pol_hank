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

varlist = ['y', 'C', 'D', 'G', 'pi', 'n', 'DIV', 'Z', 'R', 'RBr',
           'B', 'revenue', 'w', 'T', 'tau_l']

varnames = ['Output', 'Consumption', 'Deficit', 'Government Consumption', 'Inflation', 'Labor Supply', 'Profits', 'After-tax income',
            'Nominal Interest Rate',  'Ex-post Interest Rate', 'Real Bonds', 'Government Revenue', 'Real Wage', 'Transfers', 'Average Tax Rate on Labor']

shock_name = 'e_d'
shock_vals = [0.00025, 0.0005, 0.00075, 0.001, 0.00125]
save_dir = "deficit_shocks_transfers"
Palette = 'Purples'

IRF_list = plot_shocks(model_path, varlist, varnames, shock_name,
                       shock_vals, 50, Palette, save_dir)
# plot_wealth_distribution(IRF_list, models, save_dir)
