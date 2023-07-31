
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

psi_list = [20, 50, 80, 110, 140]

psiw_list = [90, 180, 250, 320, 390]

results = {}


base_dict = ep.parse(model_path)


for psi, psiw in itertools.product(psi_list, psiw_list):

    print(f'--------  psi: {psi}, psiw: {psiw}  -----------')
    # hank_dict['steady_state']['fixed_values']['gamma_B'] = 0.4
    # hank_dict['steady_state']['fixed_values']['beta'] = 0.93880593

    # copy base configuration
    hank_dict = copy.deepcopy(base_dict)

    hank_dict['steady_state']['fixed_values']['psi'] = psi
    hank_dict['steady_state']['fixed_values']['psi_w'] = psiw

    # Load and solve the model
    model = ep.load(hank_dict)
    stst_result = model.solve_stst(maxit=40)

    baseline, intervention = impulse_response(1.01, 'T',  1, model)

    multiplier = compute_mult('T', intervention, baseline, model)

    # Save the result for this combination
    results[(psi, psiw)] = multiplier

# Save dictionary to a Pickle file
with open('data.pickle', 'wb') as pickle_file:
    pickle.dump(results, pickle_file)

######################################################################


# Convert into LaTeX format for plotly
psiw_latex = [rf"$\psi_w = {x}$" for x in psiw_list]
# add white space so it does not overlap with the heatmap
psi_latex = [rf"$\psi={x}$      " for x in psi_list]

# Initialize the multipliers array
multipliers = jnp.zeros((len(psi_list), len(psiw_list)))

for i, psi in enumerate(psi_list):
    for j, psiw in enumerate(psiw_list):
        # Update the array
        multipliers = multipliers.at[i, j].set(
            results.get((psi, psiw), jnp.nan))

fig = go.Figure(data=go.Heatmap(
    z=multipliers,  # Multipliers
    x=psiw_latex,  # x-axis labels
    y=psi_latex,  # y-axis labels
    colorscale='brwnyl',  # brwnyl, aggrnyl, tealgrn, mint
    hoverongaps=False,  # Do not display hover text for missing values
    zauto=False,  # Do not automatically adjust color range
    zmin=jnp.min(multipliers).item(),  # Minimum color value
    zmax=jnp.max(multipliers).item(),  # Maximum color value
))

# # Display the actual z values on the heatmap
# fig.update_traces(showscale=True,
#                   hovertemplate='Monetary Policy response: %{x}<br>Tax Progressivity: %{y}<br>Multiplier: %{z}<extra></extra>')

fig.update_traces(
    showscale=True,
    hovertemplate='Monetary Policy response: %{x}<br>Tax Progressivity: %{y}<br>Multiplier: %{z:.2f}<extra></extra>'
)

annotations = []
for y in range(len(psi_latex)):
    for x in range(len(psiw_latex)):
        annotations.append(dict(
            xref='x1', yref='y1',
            x=psiw_latex[x], y=psi_latex[y],
            text='{:.3f}'.format(multipliers[y, x]),
            font=dict(family='Arial', size=14,
                      color='rgba(0,0,0,1)'),
            showarrow=False))

fig.update_layout(annotations=annotations)

# Set layout
fig.update_layout(
    # xaxis_title='Monetary Policy response',
    # yaxis_title='Tax Progressivity',
    xaxis_nticks=len(psiw_latex),  # Ensure all x-axis labels are displayed
    yaxis_nticks=len(psi_latex),  # Ensure all y-axis labels are displayed
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
)

# Save the figure
try:
    pio.write_image(fig, '../tables/heatmap_sticky.pdf')
    print("Figure saved as PDF.")
except:
    fig.write_image('../tables/heatmap_sticky.png')
    print("Figure saved as PNG.")
