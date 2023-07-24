
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
    ind_G = model['variables'].index('G')

    # Calculate the changes in output and government spending
    output_difference = jnp.sum(
        intervention[1:200, ind_y] - baseline[1:200, ind_y])
    government_spending_difference = jnp.sum(
        intervention[1:200, ind_G] - baseline[1:200, ind_G])

    # Compute the fiscal multipliers
    fiscal_multiplier = output_difference / government_spending_difference

    return fiscal_multiplier


def impulse_response(shock_G, shock_beta, model):

    x0 = model['stst'].copy()
    x0['beta'] *= shock_beta

    baseline, _ = model.find_path(init_state=x0.values())

    x0['G'] *= shock_G

    intervention, _ = model.find_path(init_state=x0.values())

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
    '../models/balanced_tau.yml',
    '../models/balanced_T.yml',
    '../models/debt_financed_tau.yml',  # g rule
    # '../models/g_rule_tau.yml', # g rule
    '../models/good_MPC.yml',
    '../models/good_MPC_debt.yml',
    '../models/div_rich.yml',
    '../models/div_debt.yml'
]

results = {}

# The changes to be made for each case
changes = {
    'baseline': None,
    'generous': {'tau_p': 0.132},
    'regresive': {'tau_p': 0.084},
    'high_psiw': {'psi': 67},
    'Hawkish': {'phi_pi': 1.7},
    'ZLB': {'Rstar': 1, 'pi': 1}
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
            baseline, intervention = impulse_response(1.01, 1.0005, model)
        else:
            baseline, intervention = impulse_response(1.01, b, model)

        multiplier = compute_mult(intervention, baseline, model)

        # Save the results for this model
        if i not in results:
            results[i] = {}
        results[i][case] = multiplier

# Generate the LaTeX table
latex_table = generate_latex_table(results, changes)

with open('../tables/multiplier_table_G_def2.tex', 'w') as f:
    f.write(latex_table)


# model_1 = '../models/balanced_tau.yml'
# model_2 = '../models/balanced_T.yml'
# model_3 = '../models/debt_financed_tau.yml'
# model_4 = '../models/debt_financed_T.yml'
# model_5 = '../models/good_MPC.yml'
# model_6 = '../models/div_rich.yml'

# # hank_baseline_path = '../models/hank_baseline.yml'

# # Model 1
# hank1_dict = ep.parse(hank_baseline_path)
# # compile the model
# hank_baseline = ep.load(hank1_dict)
# stst_result = hank_baseline.solve_stst(maxit=40)

# # Model 2
# print('Model 2')
# hank1_dict = ep.parse(hank_baseline_path)
# hank1_dict['steady_state']['fixed_values']['tau_p'] = 0.181
# hank_generous = ep.load(hank1_dict)
# stst_result = hank_generous.solve_stst(maxit=40)

# # Model 3
# print('Model 3')
# hank1_dict = ep.parse(hank_baseline_path)
# hank1_dict['steady_state']['fixed_values']['tau_p'] = 0.84 # optimal
# hank_regresive = ep.load(hank1_dict)
# stst_result = hank_regresive.solve_stst(maxit=40)


# # Model 4
# hank1_dict = ep.parse(hank_baseline_path)
# hank1_dict['steady_state']['fixed_values']['psi'] = 110
# hank4 = ep.load(hank1_dict)
# stst_result = hank4.solve_stst(maxit=40)


# baseline, intervention = impulse_response(1.01, b, hank_baseline)
# baseline_g, intervention_g = impulse_response(1.01, b, hank_generous)
# baseline_r, intervention_r = impulse_response(1.01, b, hank_regresive)
# baseline_psiw, intervention_psiw = impulse_response(1.01, b, hank4)
# # baseline_psi, intervention_psi = impulse_response(1.01, b, hank4)
# # baseline_hmm, intervention_hmm = impulse_response(1.01, b, hank5)

# multiplier = compute_mult(intervention, baseline, hank_baseline)
# multiplier_g = compute_mult(intervention_g, baseline_g, hank_generous)
# multiplier_r = compute_mult(intervention_r, baseline_r, hank_regresive)
# multiplier_psiw = compute_mult(intervention_psiw, baseline_psiw, hank4)
# # multiplier_psi = compute_mult(intervention_psiw, baseline_psiw, hank4)
# # multiplier_hmm = compute_mult(intervention_psiw, baseline_psiw, hank5)

# # Save the results in the dictionary
# results[0] = {'baseline': multiplier, 'generous': multiplier_g, 'regresive': multiplier_r,
#                     'high_psiw': multiplier_psiw}  # , 'higher_psi': multiplier_psi, 'HHM_calib': multiplier_hmm

# latex_table = generate_latex_table(results)

# with open('../tables/multiplier_table_transfers.tex', 'w') as f:
#     f.write(latex_table)


# beta_shocks = [0.99,  1.,  1.01]
# shock_names = ['big expansion', 'steady state', 'big recession']

# # Dictionary to store the results
# results = {}

# for b, name in zip(beta_shocks, shock_names):

#     # # Model 2
# # hank2_dict = ep.parse(hank_generous_path)
# # # compile the model
# # hank_generous = ep.load(hank2_dict)
# # stst_result = hank_generous.solve_stst(maxit=40)

# # # Model 3
# # hank3_dict = ep.parse(hank_regresive_path)
# # # compile the model
# # hank_regresive = ep.load(hank3_dict)
# # stst_result = hank_regresive.solve_stst(maxit=40)


# # # Model 4
# # hank1_dict = ep.parse(hank_baseline_path)
# # hank1_dict['steady_state']['fixed_values']['psi']= 200
# # hank4 = ep.load(hank1_dict)
# # stst_result = hank4.solve_stst(maxit=40)

# # # Model 5
# # hank1_dict = ep.parse(hank_baseline_path)
# # hank1_dict['steady_state']['fixed_values']['psi_w']= 300
# # hank1_dict['steady_state']['fixed_values']['psi']= 300
# # hank5 = ep.load(hank1_dict)
# # stst_result = hank5.solve_stst(maxit=40)
