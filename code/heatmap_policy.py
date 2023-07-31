
import jax.numpy as jnp
from grgrlib import figurator, grplot
import econpizza as ep  # pizza
import matplotlib.pyplot as plt
import os
import copy
import itertools
import pickle
import plotly.graph_objects as go
import plotly.io as pio

# TODO: create tables directory automatically

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

# functions should be in another file


def compute_mult(shock_type, intervention, baseline, model):

    ind_y = model['variables'].index('y')
    ind = model['variables'].index(shock_type)

    # Calculate the changes in output and government spending
    output_difference = jnp.sum(
        intervention[1:200, ind_y] - baseline[1:200, ind_y])
    government_shock_difference = jnp.sum(
        intervention[1:200, ind] - baseline[1:200, ind])

    # Compute the fiscal multipliers
    fiscal_multiplier = output_difference / government_shock_difference

    return fiscal_multiplier


def impulse_response(shock_size, shock_type,  shock_beta, model):

    x0 = model['stst'].copy()
    x0['beta'] *= shock_beta

    baseline, _ = model.find_path(init_state=x0.values())

    x0[shock_type] *= shock_size

    intervention, _ = model.find_path(init_state=x0.values())

    return baseline, intervention


##############################################################################################
model_path = '../models/baseline.yml'

gamma_B_list = jnp.arange(0., 1.21, 0.2)

phi_pi_list = jnp.arange(0., 2.11, 0.3)

results = {}


base_dict = ep.parse(model_path)


for gamma_B, phi_pi in itertools.product(gamma_B_list, phi_pi_list):

    print(f'--------  gamma_B: {gamma_B}, phi_pi: {phi_pi}  -----------')
    # hank_dict['steady_state']['fixed_values']['gamma_B'] = 0.4
    # hank_dict['steady_state']['fixed_values']['beta'] = 0.93880593

    # copy base configuration
    hank_dict = copy.deepcopy(base_dict)

    hank_dict['steady_state']['fixed_values']['gamma_B'] = gamma_B
    hank_dict['steady_state']['fixed_values']['phi_pi'] = phi_pi

    # Load and solve the model
    model = ep.load(hank_dict)
    stst_result = model.solve_stst(maxit=40)

    try:
        baseline, intervention = impulse_response(1.01, 'T',  1, model)
        multiplier = compute_mult('T', intervention, baseline, model)
    except Exception:  # For unstable models
        multiplier = jnp.nan

    # Save the result for this combination
    results[(float(gamma_B), float(phi_pi))] = multiplier

# Save dictionary to a Pickle file
with open('data.pickle', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)

######################################################################


# Convert into LaTeX format for plotly
phi_pi_latex = [rf"$\phi_{{\pi}} = {x:.1f}$" for x in phi_pi_list]
gamma_B_latex = [rf"$\gamma_{{B}} = {x:.1f}$        " for x in gamma_B_list]

# Initialize the multipliers array
multipliers = jnp.zeros((len(gamma_B_list), len(phi_pi_list)))

for i, gamma_B in enumerate(gamma_B_list):
    for j, phi_pi in enumerate(phi_pi_list):
        # Update the array
        multipliers = multipliers.at[i, j].set(
            results.get((float(gamma_B), float(phi_pi)), jnp.nan))

multipliers = jnp.where((jnp.isnan(multipliers)) | (multipliers > 10) | (
    multipliers < -10), jnp.nan, multipliers)  # replace if the model exploded

fig = go.Figure(data=go.Heatmap(
    z=multipliers,
    x=phi_pi_latex,
    y=gamma_B_latex,
    colorscale='mint',
    hoverongaps=False,
    zauto=False,
    zmin=jnp.nanmin(multipliers).item(),
    zmax=jnp.nanmax(multipliers).item(),
))


fig.update_traces(
    showscale=True,
    hovertemplate='Monetary Policy response: %{x}<br>Fiscal Policy Response: %{y}<br>Multiplier: %{z:.2f}<extra></extra>'
)

annotations = []
for y in range(len(gamma_B_latex)):
    for x in range(len(phi_pi_latex)):
        val = multipliers[y, x]
        if not jnp.isnan(val) and val <= 10:
            annotations.append(dict(
                xref='x1', yref='y1',
                x=phi_pi_latex[x], y=gamma_B_latex[y],
                text='{:.2f}'.format(val),
                font=dict(family='Arial', size=16,
                          color='rgba(0,0,0,1)'),
                showarrow=False))

fig.update_layout(annotations=annotations)

# Set layout
fig.update_layout(
    # xaxis_title='Monetary Policy response',
    # yaxis_title='Tax Progressivity',
    xaxis_nticks=len(phi_pi_latex),  # Ensure all x-axis labels are displayed
    yaxis_nticks=len(gamma_B_latex),  # Ensure all y-axis labels are displayed
    xaxis=dict(
        title_font=dict(size=18),  # Increase x-axis label font size
        tickfont=dict(size=16),  # Increase x-axis tick font size
    ),
    yaxis=dict(
        title_font=dict(size=18),  # Increase y-axis label font size
        tickfont=dict(size=16),  # Increase y-axis tick font size
        automargin=True,
        # tickangle=-50
    ),
    autosize=False,
    width=800,  # width in pixels
    height=600,  # height in pixels
    margin=dict(l=120, r=50, b=50, t=50, pad=4),
    plot_bgcolor='white',  # This sets the background color
    paper_bgcolor='white',  # This sets the paper (margin) color
)

# Save the figure
try:
    pio.write_image(fig, '../tables/heatmap_policy_2.pdf')
    print("Figure saved as PDF.")
except:
    fig.write_image('../tables/heatmap_policy_2.png')
    print("Figure saved as PNG.")
