
import jax.numpy as jnp  # use jax.numpy instead of normal numpy
# a nice backend for batch plotting with matplotlib
from grgrlib import figurator, grplot
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy

# TODO: create tables directory automatically

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# functions shoul be in another file

b = 1


def compute_mult(intervention, baseline, model):

    ind_y = model['variables'].index('y')
    ind_T = model['variables'].index('T')

    # Calculate the changes in output and government spending
    output_difference = jnp.sum(
        intervention[1:80, ind_y] - baseline[1:80, ind_y])
    government_transfers_difference = jnp.sum(
        intervention[1:80, ind_T] - baseline[1:80, ind_T])

    # Compute the fiscal multipliers
    fiscal_multiplier = output_difference / government_transfers_difference

    return fiscal_multiplier


def impulse_response(shock_T, shock_beta, model):

    x0 = model['stst'].copy()
    x0['beta'] *= shock_beta

    baseline, _ = model.find_path(init_state=x0.values(), maxit=70)
    shock = ('e_t', shock_T-1)

    intervention, _ = model.find_path(
        init_state=x0.values(), shock=shock, maxit=70)

    return baseline, intervention


def generate_latex_table(results, changes):
    column_names = ['Shock'] + ['Baseline' if val is None else f"$\{key}={val}$"
                                for name, change in changes.items() for key, val in (change or {}).items()]

    latex_table = '\\begin{tabular}{l' + \
        '*{%d}{c}' % (len(column_names)) + '}\n'
    latex_table += '\\toprule\n'
    latex_table += ' & '.join(column_names) + '\\\\\n'
    latex_table += '\\midrule\n'

    for i, (shock, data) in enumerate(results.items()):
        row_data = [
            f'{data[column_name]:.2f}' for column_name in changes.keys()]
        latex_table += f'{shock} & ' + ' & '.join(row_data) + ' \\\\\n'
        if i < len(results) - 1:
            latex_table += '\\addlinespace[0.5em]\n'

    latex_table += '\\bottomrule\n'
    latex_table += '\\end{tabular}'

    return latex_table


##############################################################################################
model_paths = [
    # '../models/balanced_tau.yml',
    # '../models/balanced_T.yml',
    '../models/debt_financed_tau.yml',  # g rule
    # '../models/g_rule_tau.yml', # g rule #
    # '../models/good_MPC.yml',
    '../models/good_MPC_debt.yml',
    # '../models/div_rich.yml',
    '../models/div_debt.yml'
]

results = {}

# The changes to be made for each case
changes = {
    'baseline': None,
    'generous': {'tau_p': 0.132},
    'regresive': {'tau_p': 0.084},
    'high_psiw': {'psi': 67},
    'Hawkish': {'phi_pi': 1.5},
    'ZLB': {'Rstar': 1, 'pi': 1}  # fails
}

for i, model_path in enumerate(model_paths):
    print(
        f'-------------------------- model {i+1} ---------------------------------')
    # Load the model configuration
    base_dict = ep.parse(model_path)

    for case, change in changes.items():
        print(
            f'------------------------ change {change} --------------------------------')

        # Clone the base configuration
        hank_dict = copy.deepcopy(base_dict)

        # Modify the configuration as needed
        if change is not None:
            for key, value in change.items():
                hank_dict['steady_state']['fixed_values'][key] = value

        # Load and solve the model
        model = ep.load(hank_dict)
        stst_result = model.solve_stst(maxit=40)

        # Compute the multipliers
        if case == 'ZLB':
            baseline, intervention = impulse_response(1.01, 1.0002, model)
        else:
            baseline, intervention = impulse_response(1.01, b, model)

        multiplier = compute_mult(intervention, baseline, model)

        # Save the results for this model
        if i not in results:
            results[i] = {}
        results[i][case] = multiplier

# Generate the LaTeX table
latex_table = generate_latex_table(results, changes)

with open('../tables/multiplier_table_transfers_new.tex', 'w') as f:
    f.write(latex_table)
