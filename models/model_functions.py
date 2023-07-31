# -*- coding: utf-8 -*-
"""functions for the one-asset HANK model without labor choice. Heavily inspired by https://github.com/shade-econ/sequence-jacobian/#sequence-space-jacobian
"""

import jax
import jax.numpy as jnp
from grgrjax import jax_print
from econpizza.utilities.interp import interpolate


def egm_init(a_grid, skills_grid):
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2


# GHH utility
def egm_step(Wa_p, a_grid, skills_grid, z_grid, n,  T, R, beta, sigma_c, sigma_l, tau_p, chi):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Wa_p

    # consumption can be readily obtained from MUC and MU of labor
    labor_inc = z_grid[:, None]

    c_nextgrid = ux_nextgrid**(-1/sigma_c) + (1-tau_p)*labor_inc/(1 + sigma_l)

    # jax_print(labor_inc)

    # get consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = R * a_grid

    c = interpolate(lhs, rhs, c_nextgrid)

    # get todays distribution of assets
    a = rhs + labor_inc + T[:, None] - c

    # fix consumption and labor for constrained households
    c = jnp.where(a < a_grid[0], labor_inc + rhs +
                  T[:, None] - a_grid[0], c)
    a = jnp.where(a < a_grid[0], a_grid[0], a)

    uc = (c - (1-tau_p)*labor_inc/(1 + sigma_l)) ** (-sigma_c)
    # calculate new MUC
    Wa = R * uc

    # uce = uc * skills_grid[:, None] ** (1-tau_p) / jnp.sum(skills_stationary * (skills_grid ** (1-tau_p)))

    uce = uc * skills_grid[:, None] ** (1-tau_p)

    return Wa, a, c, uce


def compute_weighted_mpc(c, a, a_grid, r, e_grid, tau_p):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = jnp.empty_like(c)
    post_return = r * a_grid

    mpc = mpc.at[:, 1:-1].set((c[:, 2:] - c[:, 0:-2]) /
                              (post_return[2:] - post_return[:-2]))
    mpc = mpc.at[:, 0].set((c[:, 1] - c[:, 0]) /
                           (post_return[1] - post_return[0]))
    mpc = mpc.at[:, -1].set((c[:, -1] - c[:, -2]) /
                            (post_return[-1] - post_return[-2]))

    mpc = jnp.where(a == a_grid[0], 1, mpc)
    mpc = mpc * e_grid[:, None]**(1-tau_p)

    return mpc


def transfers(skills_stationary, T, Div, skills_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    rule_div = skills_grid
    # rule_tranf = 1/skills_grid
    # rule_tranf = skills_grid
    # rule_tranf = jnp.array([1.2, 1.1, 0.9, 0.8 ])
    rule_tranf = jnp.ones_like(skills_grid)

    div = Div / jnp.sum(skills_stationary * rule_div) * rule_div
    transf = T / jnp.sum(skills_stationary * rule_tranf) * rule_tranf

    T = div + transf

    return T


def post_tax_lincome(y_grid, tau_l, tau_p):
    # tau_p is the progressivity of the tax system = 0 is equivalent to previous model
    z_grid = (1-tau_l)*y_grid**(1-tau_p)
    return z_grid
