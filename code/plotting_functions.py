from grgrlib import grbar3d
from grgrlib import figurator, grplot
import econpizza as ep
from econpizza.tools import percentile
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


def plot_IRFs(model_paths, model_names, varlist, varnames, shock, T, save_dir):

    if len(model_paths) != len(model_names):
        raise ValueError("Number of models and model names must be the same")

    if len(varnames) != len(varlist):
        raise ValueError("Length of varlist and varnames must be the same")

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
            if varlist[i] in {'R', 'Rn', 'Rr', 'Rstar', 'Top10C', 'Top10A', 'RBr', 'MPC', 'lowest_q_mpc'}:
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

        ax.set_title(varnames[i], size="22")
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
                ax.axhline(0, color='k', linestyle='dotted')  # Add this line
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

        ax.set_title(varlist[i], size="22")
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

        ax.set_title(varlist[i], size="22")
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


def plot_transition(model, varlist, varnames, new_states, T, save_dir):

    if len(varlist) != len(varnames):
        raise ValueError("varlist and varnames must be of same length")

    # Create directories for saving the plots
    _create_directory(save_dir)

    model_dict = ep.parse(model)
    # compile the model
    mod = ep.load(model_dict)
    _ = mod.solve_stst(maxit=40)
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
            linewidth=2.5,
            color='xkcd:steel blue',
            # label=f"Model {j}",
            # color = (0.2, 0.2, 0.2),
            alpha=0.9,
        )
        ax.hlines(y=x0[varlist[i]], xmin=-2, xmax=T+2,
                  colors='xkcd:sage green', linestyles='dashed', label='Old SS', linewidth=2)  # xkcd:steel blue
        ax.hlines(y=y0[varlist[i]], xmin=-2, xmax=T+2,
                  colors='#fd5956', linestyles='dashed', label='New SS', linewidth=2)  # xkcd:sage green

        # ax.set_ylabel("Value")
        ax.set_xlim(-2, T+2)

        ax.set_title(varnames[i], size="22")
        ax.set_xlabel("Quarters")
        ax.legend()  # fontsize='large'
        plt.savefig(os.path.join("../bld", save_dir, varlist[i] + ".pdf"))
        plt.close()

    return xst


def plot_consumption_changes_skill(model, new_states, save_dir):

    # Create directories for saving the plots
    _create_directory(save_dir)

    model_dict = ep.parse(model)
    # compile the model
    mod = ep.load(model_dict)
    _ = mod.solve_stst(maxit=40)
    x0 = mod['stst'].copy()

    for key, value in new_states.items():
        model_dict['steady_state']['fixed_values'][key] = value

    mod_2 = ep.load(model_dict)
    _ = mod_2.solve_stst(maxit=40)
    y0 = mod_2['stst'].copy()

    c0 = mod['steady_state']['decisions']['c']
    dist0 = mod['steady_state']['distributions'][0]

    c1 = mod_2['steady_state']['decisions']['c']
    dist1 = mod_2['steady_state']['distributions'][0]

    avg_c0 = jnp.sum(dist0 * c0, axis=1)
    avg_c1 = jnp.sum(dist1 * c1, axis=1)

    percentage_change = ((avg_c1 - avg_c0) / avg_c0) * 100

    deciles = jnp.arange(1, 1+len(c0[:, 0]))

    bar_width = 0.8  # Width of the bars
    zero_line_width = 2.0  # Width of the zero line

    fig, ax = plt.subplots()
    bars = ax.bar(deciles, percentage_change, width=bar_width, align='center')

    # Set the 0 value line
    ax.axhline(y=0, color='red', linewidth=zero_line_width)

    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dashed', alpha=0.5)
    ax.set_axisbelow(True)

    # Add labels and title
    plt.xlabel('Skill level')
    plt.ylabel('Percentage Change in Consumption')
    # plt.title('Percentage Change in Consumption for Each Decile')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}%', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}%', ha='center', va='top')

    # Adjust the y-axis limits to center the 0 value
    max_value = jnp.max(percentage_change)
    min_value = jnp.min(percentage_change)
    y_max = max(abs(max_value), abs(min_value)) * 1.1
    plt.ylim(-y_max, y_max)

    plt.tight_layout()

    plt.savefig(os.path.join("../bld", save_dir,
                'consumption_changes_skill.pdf'))
    plt.close()


def plot_consumption_changes_deciles(model, new_states, save_dir):

    # Create directories for saving the plots
    _create_directory(save_dir)

    model_dict = ep.parse(model)
    # compile the model
    mod = ep.load(model_dict)
    _ = mod.solve_stst(maxit=40)
    x0 = mod['stst'].copy()

    for key, value in new_states.items():
        model_dict['steady_state']['fixed_values'][key] = value

    mod_2 = ep.load(model_dict)
    _ = mod_2.solve_stst(maxit=40)
    y0 = mod_2['stst'].copy()

    c0 = mod['steady_state']['decisions']['c']
    dist0 = mod['steady_state']['distributions'][0]

    c1 = mod_2['steady_state']['decisions']['c']
    dist1 = mod_2['steady_state']['distributions'][0]

    consumption_dist0 = _calculate_consumption_percentiles(c0, dist0)
    consumption_dist1 = _calculate_consumption_percentiles(c1, dist1)

    deciles = jnp.arange(1, 11)

    percentage_change = (
        (consumption_dist1 - consumption_dist0) / consumption_dist0) * 100

    bar_width = 0.8  # Width of the bars
    zero_line_width = 2.0  # Width of the zero line

    fig, ax = plt.subplots()
    bars = ax.bar(deciles, percentage_change, width=bar_width, align='center')

    # Set the 0 value line
    ax.axhline(y=0, color='red', linewidth=zero_line_width)

    # Customize the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='dashed', alpha=0.5)
    ax.set_axisbelow(True)

    # Add labels and title
    plt.xlabel('Consumption Decile')
    plt.ylabel('Percentage Change in Consumption')
    # plt.title('Percentage Change in Consumption for Each Decile')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}%', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}%', ha='center', va='top')

    # Adjust the y-axis limits to center the 0 value
    max_value = jnp.max(percentage_change)
    min_value = jnp.min(percentage_change)
    y_max = max(abs(max_value), abs(min_value)) * 1.1
    plt.ylim(-y_max, y_max)

    plt.tight_layout()
    plt.savefig(os.path.join("../bld", save_dir,
                'consumption_changes_decile.pdf'))
    plt.close()


def _calculate_consumption_percentiles(C, dist):
    flattened_C = C.flatten()
    flattened_dist = dist.flatten()

    sorted_indices = jnp.argsort(flattened_C)
    sorted_C = flattened_C[sorted_indices]
    sorted_dist = flattened_dist[sorted_indices]

    cumulative_sum = jnp.cumsum(sorted_dist)
    total_sum = cumulative_sum[-1]
    cdf = cumulative_sum / total_sum

    percentiles = jnp.arange(0.1, 100, 10)
    consumption_percentiles = []

    for percentile in percentiles:
        indices_greater = jnp.where(cdf >= percentile / 100)
        if indices_greater[0].size == 0:
            # Handle the case where no elements satisfy the condition.
            consumption = jnp.nan
        else:
            index = indices_greater[0][0]
            consumption = sorted_C[index]
        consumption_percentiles.append(consumption)

    return jnp.array(consumption_percentiles)


def _calculate_average_consumption_percentiles(C, dist):
    flattened_C = C.flatten()
    flattened_dist = dist.flatten()

    sorted_indices = jnp.argsort(flattened_C)
    sorted_C = flattened_C[sorted_indices]
    sorted_dist = flattened_dist[sorted_indices]

    cumulative_sum = jnp.cumsum(sorted_dist)
    cdf = cumulative_sum / cumulative_sum[-1]

    # start from 0.01 to avoid problems
    percentiles = jnp.arange(0.01, 101, 10)
    average_consumption_percentiles = []

    for i in range(len(percentiles) - 1):
        start_percentile = percentiles[i] / 100
        end_percentile = percentiles[i + 1] / 100

        start_index = jnp.searchsorted(cdf, start_percentile, side='left')
        end_index = jnp.searchsorted(cdf, end_percentile, side='right')

        if start_index == end_index:
            # Handle case where start_index and end_index are the same
            # keep it as an array
            consumption_within_percentile = sorted_C[start_index:start_index+1]
        else:
            consumption_within_percentile = sorted_C[start_index:end_index]

        if consumption_within_percentile.size > 0:
            average_consumption = jnp.mean(consumption_within_percentile)
        else:
            average_consumption = jnp.nan

        average_consumption_percentiles.append(average_consumption)

    return jnp.array(average_consumption_percentiles)
