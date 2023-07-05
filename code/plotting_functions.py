from grgrlib import grbar3d
from grgrlib import figurator, grplot
import econpizza as ep
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from matplotlib import cm


from utilities import _create_directory


def plot_IRFs_init_cond(model_paths, varlist, init_conds, save_dir):
    models = []
    steady_states = []
    IRF_list = []

    # Load models and solve steady states
    for path in model_paths:
        model = ep.load(path)
        _ = model.solve_stst()
        models.append(model)
        steady_states.append(model["stst"].copy())

    for x0 in steady_states:
        for key, value in init_conds.items():
            x0[key] *= value

    # Find variables index
    inds_models = [[model["variables"].index(
        v) for v in varlist] for model in models]

    # Find IRFs
    paths, flags = zip(*[model.find_path(init_state=steady_states[i].values())
                       for i, model in enumerate(models)])

    [IRF_list.append(path) for path in paths]

    # Create directories for saving the plots
    _create_directory(save_dir)

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, path in enumerate(paths):
            model = models[j]
            ss = steady_states[j]
            inds = inds_models[j]
            # We don't want interest rate as a percentage deviation
            if varlist[i] in {'R', 'Rn', 'Rr', 'pi', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'tau_l', 'MPC', 'lowest_q_mpc'}:
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

    return models, IRF_list


def plot_IRFs(model_paths, model_names, varlist, shock, T, save_dir):
    models = []
    steady_states = []
    IRF_list = []

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

    [IRF_list.append(path) for path in paths]

    # Create directories for saving the plots
    _create_directory(save_dir)

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, path in enumerate(paths):
            model = models[j]
            ss = steady_states[j]
            inds = inds_models[j]
            # We don't want interest rate as a percentage deviation
            if varlist[i] in {'R', 'Rn', 'Rr', 'pi', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'MPC', 'lowest_q_mpc'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    marker="o",
                    linestyle="-",
                    label=model_names[j],
                    alpha=0.9,
                )
                ax.set_ylabel("Value")
            elif varlist[i] in {'D', 'deficit'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    marker="o",
                    linestyle="-",
                    label=model_names[j],
                    alpha=0.9,
                )
                ax.axhline(0, color='red', linestyle='dotted')  # Add this line
                ax.set_ylabel("Value")
            else:
                # plot as % deviation from Steady State
                ax.plot(
                    (path[0:T, inds[i]] - ss[varlist[i]]) /
                    ss[varlist[i]] * 100,
                    marker="o",
                    linestyle="-",
                    label=model_names[j],
                    alpha=0.9,
                )
                ax.set_ylabel("Percent")

        ax.set_title(varlist[i], size="18")
        ax.set_xlabel("Quarters")
        ax.legend()
        plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
        plt.close()

    return models, IRF_list


def plot_shocks(model_path, varlist, shock_name, shock_vals, T, Palette, save_dir):
    IRF_list = []
    shocks_list = []

    model = ep.load(model_path)
    _ = model.solve_stst()
    ss = model["stst"].copy()

    for v in shock_vals:
        s = (shock_name, v)
        shocks_list.append(s)

    # Find IRFs
    paths, flags = zip(*[model.find_path(shock=shock)
                       for shock in shocks_list])

    [IRF_list.append(path) for path in paths]

    # Find variables index
    inds = [model["variables"].index(v) for v in varlist]

    # Create directories for saving the plots
    _create_directory(save_dir)

    # Define colormap
    cmap = cm.get_cmap(Palette)

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, path in enumerate(paths):
            color = cmap(float(j+1)/len(paths))   # Use colormap
            ss = steady_states[j]
            # We don't want interest rate as a percentage deviation
            if varlist[i] in {'R', 'Rn', 'Rr', 'pi', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'MPC', 'lowest_q_mpc'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    linestyle="-",
                    label=f"{shock_name} = {shock_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.set_ylabel("Value")
            elif varlist[i] in {'D', 'deficit'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    linestyle="-",
                    label=f"{shock_name} = {shock_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.axhline(0, color='red', linestyle='dotted')  # Add this line
                ax.set_ylabel("Value")
            else:
                # plot as % deviation from Steady State
                ax.plot(
                    (path[0:T, inds[i]] - ss[varlist[i]]) /
                    ss[varlist[i]] * 100,
                    linestyle="-",
                    label=f"{shock_name} = {shock_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.set_ylabel("Percent")

        ax.set_title(varlist[i], size="18")
        ax.set_xlabel("Quarters")
        ax.legend()
        plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
        plt.close()

    return IRF_list


def plot_dif_params(model_path, varlist, shock, param_name, param_vals, T, Palette, save_dir):

    IRF_list = []
    steady_states = []

    model_dict = ep.parse(model_path)

    for v in param_vals:
        model_dict['steady_state']['fixed_values'][param_name] = v
        model = ep.load(model_dict)
        _ = model.solve_stst()
        steady_states.append(model["stst"].copy())
        x, __name__ = model.find_path(shock=shock)
        IRF_list.append(x)

    # Find variables index
    inds = [model["variables"].index(v) for v in varlist]

    # Create directories for saving the plots
    _create_directory(save_dir)

    # Define colormap
    cmap = cm.get_cmap(Palette)

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, path in enumerate(IRF_list):
            color = cmap(float(j+1)/len(IRF_list))   # Use colormap
            ss = steady_states[j]
            # We don't want interest rate as a percentage deviation
            if varlist[i] in {'R', 'Rn', 'Rr', 'pi', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'MPC', 'lowest_q_mpc'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    linestyle="-",
                    # label=fr"{param_name} = {param_vals[j]}", # reads latex
                    label=f"{param_name} = {param_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.set_ylabel("Value")
            elif varlist[i] in {'D', 'deficit'}:
                ax.plot(
                    (path[0:T, inds[i]]),
                    linestyle="-",
                    label=f"{param_name} = {param_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.axhline(0, color='red', linestyle='dotted')  # Add this line
                ax.set_ylabel("Value")
            else:
                # plot as % deviation from Steady State
                ax.plot(
                    (path[0:T, inds[i]] - ss[varlist[i]]) /
                    ss[varlist[i]] * 100,
                    linestyle="-",
                    label=f"{param_name} = {param_vals[j]}",
                    alpha=0.9,
                    linewidth=2.3,
                    color=color
                )
                ax.set_ylabel("Percent")

        ax.set_title(varlist[i], size="18")
        ax.set_xlabel("Quarters")
        ax.legend()
        plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
        plt.close()

    return IRF_list


def plot_wealth_distribution(IRF_list, model_list, save_dir):
    # colors = ['blue', 'red']  # different color for each model color=colors[i],
    # alphas = [0.5, 0.3]  # different transparency for each model

    for i, path in enumerate(IRF_list):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        het_vars = model_list[i].get_distributions(path)
        dist = het_vars['dist']
        a_grid = model_list[i]['context']['a_grid']

        # create coordinate arrays for the surface plot
        X, Y = jnp.meshgrid(a_grid, jnp.arange(30))
        Z = dist[..., :30].sum(0)

        # plot
        ax.bar3d(X.flatten(), Y.flatten(), jnp.zeros(
            len(Z.flatten())), 0.2, 0.2, Z.flatten(), alpha=0.5)

        # set axis labels
        ax.set_xlabel('Wealth')
        ax.set_ylabel('Time')
        ax.set_zlabel('Share')

        # rotate
        ax.view_init(azim=40)

        # save the plot
        plt.savefig(os.path.join("../bld", save_dir,
                    f"wealth_distribution_model_{i}.pdf"))
        plt.close()


def plot_transition(model, varlist, new_states, T, save_dir):

    # Create directories for saving the plots
    _create_directory(save_dir)

    model_dict = ep.parse(model)
    # compile the model
    mod = ep.load(model_dict)
    _ = mod.solve_stst(maxit=60)
    x0 = mod['stst'].copy()

    for key, value in new_states.items():
        model_dict['steady_state']['fixed_values'][key] = value

    mod_2 = ep.load(model_dict)
    _ = mod_2.solve_stst(maxit=40)
    y0 = mod_2['stst'].copy()

    xst, _ = mod_2.find_path(init_state=x0.values())

    inds = [mod_2["variables"].index(v) for v in varlist]

    # Plot IRFs
    for i in range(len(varlist)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            (xst[:T, inds[i]]),
            marker="o",
            linestyle="-",
            # label=f"Model {j}",
            alpha=0.9,
        )
        ax.hlines(y=x0[varlist[i]], xmin=0, xmax=T,
                  colors='xkcd:steel blue', linestyles='dashed', label='Old SS')
        ax.hlines(y=y0[varlist[i]], xmin=0, xmax=T,
                  colors='xkcd:sage green', linestyles='dashed', label='New SS')

        ax.set_ylabel("Value")

        ax.set_title(varlist[i], size="18")
        ax.set_xlabel("Quarters")
        ax.legend()
        plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
        plt.close()

    return xst
