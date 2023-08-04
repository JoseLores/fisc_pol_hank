
import jax.numpy as jnp
import econpizza as ep


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
