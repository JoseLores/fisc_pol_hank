
import jax.numpy as jnp  # use jax.numpy instead of normal numpy
# a nice backend for batch plotting with matplotlib
from grgrlib import figurator, grplot
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# functions shoul be in another file


def compute_mult(intervention, baseline, model):

    ind_y = model['variables'].index('y')
    ind_G = model['variables'].index('G')

    # Calculate the changes in output and government spending
    output_difference = jnp.sum(
        intervention[1:31, ind_y] - baseline[1:31, ind_y])
    government_spending_difference = jnp.sum(
        intervention[1:31, ind_G] - baseline[1:31, ind_G])

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


def generate_latex_table(results):
    latex_table = '\\begin{tabular}{|l|c|c|c|}\n'
    latex_table += '\\hline\n'
    latex_table += 'Shock & Baseline  & Poor Pay & Rich Pay & Higher $\\psi_w$\\\\\n'
    latex_table += '\\hline\n'

    for shock, data in results.items():
        baseline_multiplier = data['baseline']
        generous_multiplier = data['generous']
        higher_psiw = data['high_psiw']
        regresive = data['regresive']
        latex_table += f'{shock} & {baseline_multiplier:.2f} & {generous_multiplier:.2f} & {regresive:.2f} & {higher_psiw:.2f} \\\\\n'

    latex_table += '\\hline\n'
    latex_table += '\\end{tabular}'

    return latex_table


##############################################################################################
hank_baseline_path = '../models/final_balanced_transfers.yml'
# hank_generous_path = '../models/transfers_generous.yml'
# hank_regresive_path = '../models/transfers_regresive.yml'

# Model 1
hank1_dict = ep.parse(hank_baseline_path)
# compile the model
hank_baseline = ep.load(hank1_dict)
stst_result = hank_baseline.solve_stst(maxit=40)

# Model 2
hank1_dict = ep.parse(hank_baseline_path)
hank1_dict['steady_state']['fixed_values']['tau_p'] = 0.15
hank_generous = ep.load(hank1_dict)
stst_result = hank_generous.solve_stst(maxit=40)

# Model 3
hank1_dict = ep.parse(hank_baseline_path)
hank1_dict['steady_state']['fixed_values']['tau_p'] = -0.05
hank_regresive = ep.load(hank1_dict)
stst_result = hank_regresive.solve_stst(maxit=40)

# # Model 2
# hank2_dict = ep.parse(hank_generous_path)
# # compile the model
# hank_generous = ep.load(hank2_dict)
# stst_result = hank_generous.solve_stst(maxit=40)

# # Model 3
# hank3_dict = ep.parse(hank_regresive_path)
# # compile the model
# hank_regresive = ep.load(hank3_dict)
# stst_result = hank_regresive.solve_stst(maxit=40)

# Model 4
hank1_dict = ep.parse(hank_baseline_path)
hank1_dict['steady_state']['fixed_values']['psi_w'] = 300
hank4 = ep.load(hank1_dict)
stst_result = hank4.solve_stst(maxit=40)

# # Model 4
# hank1_dict = ep.parse(hank_baseline_path)
# hank1_dict['steady_state']['fixed_values']['psi']= 200
# hank4 = ep.load(hank1_dict)
# stst_result = hank4.solve_stst(maxit=40)

# # Model 5
# hank1_dict = ep.parse(hank_baseline_path)
# hank1_dict['steady_state']['fixed_values']['psi_w']= 300
# hank1_dict['steady_state']['fixed_values']['psi']= 300
# hank5 = ep.load(hank1_dict)
# stst_result = hank5.solve_stst(maxit=40)


beta_shocks = [0.99,  1.,  1.01]
shock_names = ['big expansion', 'steady state', 'big recession']

# Dictionary to store the results
results = {}

for b, name in zip(beta_shocks, shock_names):
    baseline, intervention = impulse_response(1.01, b, hank_baseline)
    baseline_g, intervention_g = impulse_response(1.01, b, hank_generous)
    baseline_r, intervention_r = impulse_response(1.01, b, hank_regresive)
    baseline_psiw, intervention_psiw = impulse_response(1.01, b, hank4)
    # baseline_psi, intervention_psi = impulse_response(1.01, b, hank4)
    # baseline_hmm, intervention_hmm = impulse_response(1.01, b, hank5)

    multiplier = compute_mult(intervention, baseline, hank_baseline)
    multiplier_g = compute_mult(intervention_g, baseline_g, hank_generous)
    multiplier_r = compute_mult(intervention_r, baseline_r, hank_regresive)
    multiplier_psiw = compute_mult(intervention_psiw, baseline_psiw, hank4)
    # multiplier_psi = compute_mult(intervention_psiw, baseline_psiw, hank4)
    # multiplier_hmm = compute_mult(intervention_psiw, baseline_psiw, hank5)

    # Save the results in the dictionary
    results[name] = {'baseline': multiplier, 'generous': multiplier_g, 'regresive': multiplier_r,
                     'high_psiw': multiplier_psiw}  # , 'higher_psi': multiplier_psi, 'HHM_calib': multiplier_hmm

latex_table = generate_latex_table(results)

with open('../tables/multiplier_table_balanced_transfers.tex', 'w') as f:
    f.write(latex_table)
