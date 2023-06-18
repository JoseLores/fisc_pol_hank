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

    # this without tau works but it should be (1-tau_p)*labor_inc/(1 + sigma_l)
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

    # uc = (c - labor_inc/(1 + sigma_l)) ** (-sigma_c)

    # uc = (c - chi*skills_grid[:, None]**(1-tau_p)*n**(1+sigma_l)/(1+sigma_l)) ** (-sigma_c) # does not work ;(
    # uce = skills_grid[:, None] * uc

    # this without tau works but it should be (1-tau_p)*labor_inc/(1 + sigma_l)
    uc = (c - (1-tau_p)*labor_inc/(1 + sigma_l)) ** (-sigma_c)
    # calculate new MUC
    Wa = R * uc

    return Wa, a, c


def compute_weighted_mpc(c, a, a_grid, r, e_grid):
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
    mpc = mpc * e_grid[:, None]

    return mpc


# GHH utility
# def egm_step(Wa_p, a_grid, z_grid, T, R, beta, sigma_c, sigma_l):
#     """A single backward step via EGM
#     """

#     # MUC as implied by next periods value function
#     ux_nextgrid = beta * Wa_p

#     # consumption can be readily obtained from MUC and MU of labor
#     c_nextgrid = ux_nextgrid**(-1/sigma_c) + z_grid/(1 + sigma_l)

#     # get consumption in grid space
#     lhs = c_nextgrid - z_grid + a_grid[None, :] - T[:, None]
#     rhs = R * a_grid

#     c = interpolate(lhs, rhs, c_nextgrid)

#     # get todays distribution of assets
#     a = rhs + z_grid + T[:, None] - c

#     # fix consumption and labor for constrained households
#     c = jnp.where(a < a_grid[0], z_grid + rhs +
#                   T[:, None] - a_grid[0], c)
#     a = jnp.where(a < a_grid[0], a_grid[0], a)


#     uc = (c - z_grid/(1 + sigma_l)) ** (-sigma_c)
#     # uce = skills_grid[:, None] * uc

#     # calculate new MUC
#     Wa = R * uc

#     return Wa, a, c #, uce

def labor_supply(w, tau_l, tau_p):
    """Labor supply as a function of wage and tax rates
    """
    return


def transfers(skills_stationary, T, skills_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    # rule_div= skills_grid
    # rule_tranf = 1/skills_grid
    # rule_tranf = skills_grid
    # rule_tranf = jnp.array([1.2, 1.1, 0.9, 0.8 ])
    rule_tranf = jnp.ones_like(skills_grid)

    # div = Div / jnp.sum(skills_stationary * rule_div) * rule_div
    transf = T / jnp.sum(skills_stationary * rule_tranf) * rule_tranf

    # T = div + transf

    return transf


def make_grids_pdf(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, e_pdf, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, e_pdf, Pi, a_grid


def pre_tax_lincome(w, N, e_grid):
    Y = w*N
    y_grid = Y * e_grid[:, None]
    return y_grid  # pre tax income


def pre_tax_lincome_cyclical(w, N, e_grid, e_stationary, zeta):
    Y = w*N
    y_grid = Y * e_grid ** (1 + zeta * jnp.log(Y)) / \
        jnp.vdot(e_grid ** (1 + zeta * jnp.log(Y)), e_stationary)
    return y_grid  # pre tax income


def post_tax_lincome(y_grid, tau_l, tau_p):
    # tau_p is the progressivity of the tax system = 0 is equivalent to previous model
    z_grid = (1-tau_l)*y_grid**(1-tau_p)
    return z_grid
