
import jax.numpy as jnp
import econpizza as ep
import matplotlib.pyplot as plt
import os
import copy
import pickle

from utilities import _create_directory
from multipliers_functions import compute_mult, impulse_response

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

_create_directory('figures')

##############################################################################################
# Function for generating latex table


def _results_to_latex_table(results, shock_names, model_specifications):
    policy_names = [
        'Labor taxes',
        r'$\Delta G = - \Delta T$',
        'Debt'
    ]

    table = r"""
    \begin{table}[t!]
    \centering
    \begin{threeparttable}
    \caption{Fiscal Multipliers for Different Shocks and Financing Methods}
    \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{3}{c}{T Shock} & \multicolumn{3}{c}{G Shock} \\
    \cline{2-4} \cline{5-7}
    & Low-MPC & Mid-MPC & High-MPC & Low-MPC & Mid-MPC & High-MPC \\
    \midrule
    """
    for policy_name in policy_names:
        row = f"{policy_name} "
        for shock_name in shocks:
            for specification_name in model_specifications:
                try:
                    multiplier = results[(
                        shock_name, specification_name, policy_name)]
                except KeyError:
                    multiplier = ''
                row += f"& {multiplier:.2f} "
        row += r'\\'
        table += row + '\n'

    table += r"""
    \bottomrule
    \end{tabular}
    \label{table:3}
    \begin{tablenotes}
    \item[] T shock is a 1\% increase in uniform transfers. G shock is a 1\% increase in government expenditure. $\Delta G = - \Delta T$ means that the increase in transfers is financed by cuts in public expenditure and vice versa. Debt means that it follows the proposed deficit rule.
    \end{tablenotes}
    \end{threeparttable}
    \end{table}
    """
    return table


##############################################################################################
model_paths = [
    '../models/balanced_tau.yml',
    '../models/balanced_T.yml',
    '../models/balanced_G.yml',
    '../models/baseline.yml'
]

model_specifications = ['Low-MPC', 'Mid-MPC', 'High-MPC']
steady_state_betas = [0.95382926, 0.93880593, 0.90293616]
steady_state_B = [1.75183783, 0.97122222, 0.26094991]
steady_state_gamma = [0.55, 0.4, 0.15]

results = {}

policy_names = [
    'Labor taxes',
    r'$\Delta G = - \Delta T$',
    r'$\Delta G = - \Delta T$',
    'Debt and Labor taxes'
]

shock_names = ['T shock', 'G shock']

shocks = {
    'T shock': 'T',
    'G shock': 'G'
}

for i, model_path in enumerate(model_paths):
    print(
        f'-------------------------- model {i+1} ---------------------------------')

    # Load the model configuration
    base_dict = ep.parse(model_path)

    for z, specification_name in enumerate(model_specifications):

        # Clone the base configuration
        hank_dict = copy.deepcopy(base_dict)
        hank_dict['steady_state']['fixed_values']['B'] = steady_state_B[z]
        hank_dict['steady_state']['fixed_values']['beta'] = steady_state_betas[z]
        hank_dict['steady_state']['fixed_values']['gamma_B'] = steady_state_gamma[z]

        # Load and solve the model
        model = ep.load(hank_dict)
        stst_result = model.solve_stst(maxit=40)

        for j, shock_name, shock_type in zip(range(len(shocks)), shocks.keys(), shocks.values()):

            # no T shock when budget adjusts T and same for G
            if (shock_type == 'T' and i == 1) or (shock_type == 'G' and i == 2):
                continue

            baseline, intervention = impulse_response(
                1.01, shock_type, 1, model)
            # no shock to beta => baseline is the steady state

            multiplier = compute_mult(
                shock_type, intervention, baseline, model)

            # Save the results for this model and shock
            results[(shock_name, specification_name,
                     policy_names[i])] = multiplier

# Save dictionary to a Pickle file
with open('data.pickle', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)


# Generate the LaTeX table
latex_table = _results_to_latex_table(
    results, shock_names, model_specifications)

with open('../bld/figures/Table_3.tex', 'w') as f:
    f.write(latex_table)
