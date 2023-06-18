from grgrlib import grbar3d
from grgrlib import figurator, grplot
import econpizza as ep
import matplotlib.pyplot as plt
import os


from utilities import _create_directory


def plot_IRFs(model_paths, varlist, shock, save_dir):
    models = []
    steady_states = []

    # Load models and solve steady states
    for path in model_paths:
        model = ep.load(path)
        _ = model.solve_stst()
        models.append(model)
        steady_states.append(model["stst"].copy())

    # Find variables index
    inds_models = [[model["variables"].index(
        v) for v in varlist] for model in models]

    # Find IRFs
    paths, flags = zip(*[model.find_path(shock=shock) for model in models])

    # Create directories for saving the plots
    _create_directory(save_dir)

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, path in enumerate(paths):
            model = models[j]
            ss = steady_states[j]
            inds = inds_models[j]
            # We dont want interest rate as a percentage deviation
            if varlist[i] in {'R', 'Rn', 'Rr', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'tau_l', 'MPC', 'lowest_q_mpc'}:
                ax.plot(
                    (path[0:30, inds[i]]),
                    marker="o",
                    linestyle="-",
                    label=f"Model {j}",
                    alpha=0.9,
                )
                ax.set_ylabel("Value")
            else:
                # plot as % deviation from Steady State
                ax.plot(
                    (path[0:30, inds[i]] - ss[varlist[i]]) /
                    ss[varlist[i]] * 100,
                    marker="o",
                    linestyle="-",
                    label=f"Model {j}",
                    alpha=0.9,
                )
                ax.set_ylabel("Percent")

            ax.set_title(varlist[i], size="18")
            ax.set_xlabel("Quarters")
            ax.legend()
            plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
            plt.close()


def show_irfs(irfs_list, variables, labels=[" "], ylabel=r"Percentage points (dev. from ss)", T_plot=50, figsize=(18, 6)):
    if len(irfs_list) != len(labels):
        labels = [" "] * len(irfs_list)
    n_var = len(variables)
    fig, ax = plt.subplots(1, n_var, figsize=figsize, sharex=True)
    for i in range(n_var):
        # plot all irfs
        for j, irf in enumerate(irfs_list):
            ax[i].plot(100 * irf[variables[i]][:50], label=labels[j])
        ax[i].set_title(variables[i])
        ax[i].set_xlabel(r"$t$")
        if i == 0:
            ax[i].set_ylabel(ylabel)
        ax[i].legend()
    plt.show()
