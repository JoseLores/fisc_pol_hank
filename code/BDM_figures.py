
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
    ind_omega = model['variables'].index('omega')

    # Calculate the changes in output and government spending
    output_difference = jnp.sum(
        intervention[1:21, ind_y] - baseline[1:21, ind_y])
    government_spending_difference = jnp.sum(
        intervention[1:21, ind_G] - baseline[1:21, ind_G])

    # Compute the fiscal multipliers
    fiscal_multiplier = output_difference / government_spending_difference

    return fiscal_multiplier

# 3


BDM_file = '../models/tank/BDM_final.yml'
BDM_dict = ep.parse(BDM_file)
# compile the model
BDM = ep.load(BDM_file)
# stst_result = BDM.solve_stst(2.36e-08) # due to gammas approx in their model we have to decrease tolerance by 1.36e-08
# x0 = BDM['stst'].copy()

# # # variables to track
# variables = ['u']
# ind_u = [BDM['variables'].index(v) for v in variables]

# Useful objects
rho_omega = 0.8
# UI_change = jnp.arange(0.001, 1.6, 0.05)
UI_change = jnp.arange(0.2, 1.6, 0.05)
mult_exp_high = jnp.zeros(len(UI_change))
mult_exp_low = jnp.zeros(len(UI_change))
mult_con_high = jnp.zeros(len(UI_change))
mult_con_low = jnp.zeros(len(UI_change))
UI_proportion_ss = jnp.zeros(len(UI_change))

# loop trough UI changes
for i, UI in enumerate(UI_change):

    BDM = ep.load(BDM_file)  # reload the model
    BDM["steady_state"]["fixed_values"]["C_u"] *= UI
    stst_result = BDM.solve_stst(1e-07)
    x0 = BDM['stst'].copy()
    UI_proportion_ss = UI_proportion_ss.at[i].set(x0['b_proportion'])

    # create a grid for the shocks
    beta_shock = jnp.arange(0.99, 1.5, 0.01)
    multiplier_expansion = jnp.zeros(len(beta_shock))
    multiplier_contraction = jnp.zeros(len(beta_shock))
    u_base = jnp.zeros(len(beta_shock))
    # beta_base = jnp.zeros(len(beta_shock))

    for j, grid_value in enumerate(beta_shock):

        # create a first state of the economy with a shock
        x0 = BDM['stst'].copy()  # reset x0

        x0['beta'] = beta_shock[j]

        baseline, _ = BDM.find_path(init_state=x0.values())

        # # reverse engineering x0['omega] so that t=1 value 1% shock
        omega_t1 = x0['omega']*1.01
        x0['omega'] = omega_t1**(1/rho_omega)*0.2**((rho_omega-1)/rho_omega)

        government_expansion, _ = BDM.find_path(init_state=x0.values())

        # negative G shock
        x0['omega'] = 0.2
        omega_t1 = x0['omega']*0.99
        x0['omega'] = omega_t1**(1/rho_omega)*0.2**((rho_omega-1)/rho_omega)

        government_contraction, _ = BDM.find_path(init_state=x0.values())

        # save unemployment
        u_base = u_base.at[j].set(baseline[1, BDM['variables'].index('u')])

        # Compute multipliers
        multiplier_expansion = multiplier_expansion.at[j].set(
            compute_mult(government_expansion, baseline, BDM))
        multiplier_contraction = multiplier_contraction.at[j].set(
            compute_mult(government_contraction, baseline, BDM))

    # take index for the multiplier
    low_unemp = jnp.argmax(u_base > 0.05)
    high_unemp = jnp.argmax(u_base > 0.08)

    # save multipliers of interest
    mult_exp_high = mult_exp_high.at[i].set(multiplier_expansion[high_unemp])
    mult_exp_low = mult_exp_low.at[i].set(multiplier_expansion[low_unemp])
    mult_con_high = mult_con_high.at[i].set(multiplier_contraction[high_unemp])
    mult_con_low = mult_con_low.at[i].set(multiplier_contraction[low_unemp])


# plot the results
# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(UI_proportion_ss, mult_exp_high, label='Expansion, high unemployment')
ax.plot(UI_proportion_ss, mult_exp_low, label='Expansion, low unemployment')
ax.plot(UI_proportion_ss, mult_con_high,
        label='Contraction, high unemployment')
ax.plot(UI_proportion_ss, mult_con_low, label='Contraction, low unemployment')

# Add a vertical dotted line at UI_proportion_SS=0.4
ax.axvline(x=0.4, linestyle='--')

# Add text annotation at UI_proportion_SS=0.4
ax.text(0.45, ax.get_ylim()[1]*0.6, 'BDM',
        horizontalalignment='center', fontsize=12)

ax.set_xlabel(
    'Unemployment insurance as percentage of employed income in steady state')
ax.set_ylabel('Fiscal multiplier')
ax.legend()

fig.savefig('../bld/multipliers3.pdf')
