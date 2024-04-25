import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import seaborn as sns

def plot_counting_process(arrival_times):
    """
    Plot a counting process given a list of arrival times.

    Parameters:
    - arrival_times: List of arrival times.
    """
    # Sort arrival times
    arrival_times.sort()

    # Prepend a value of 0 at time 0
    arrival_times = [0] + arrival_times

    # Generate y-values for the counting process
    counting_process = np.arange(len(arrival_times))

    # Plot the counting process
    plt.step(arrival_times, counting_process, where='post')

    # Add labels and title
    plt.xlabel('Time $t$')
    plt.ylabel('$N_t$')
    #plt.title('Counting Process Plot')

    # Show the plot
    plt.show()

# Background functions and kernels

def exp_kernel(t, alpha=1, delta=2):
  if t<0:
    return 0
  return alpha*np.exp(-1*delta*t)

def exp_kernel_vectorised(t, alpha=1, delta=2):
    return np.where(t < 0, 0, alpha * np.exp(-delta * t))

def constant_background(t):
  return 1

def conditional_intensity(t, events_list, background_intensity, memory_kernel):
  sum = 0
  for T in events_list:
    if T < t:
      sum += memory_kernel(t-T)
  return sum + background_intensity(t)

def conditional_intensity_vectorised(t_values, events_list, background_intensity, memory_kernel):
    result = np.zeros_like(t_values)  # Initialize an array for the results

    for T in events_list:
        mask = T < t_values
        result += mask * memory_kernel(t_values - T)

    result += background_intensity(t_values)

    return result

def conditional_intensity_true_vectorised(t_vals, events_list, background_intensity, memory_kernel):

  t_vals = np.atleast_1d(t_vals)

  events_list = np.array(events_list)

  events_list = events_list[:, np.newaxis]
  t_vals = t_vals[np.newaxis, :]

  input_mat = events_list - t_vals

  input_mat = t_vals - events_list
  input_mat = np.where(input_mat >= 0, input_mat, np.nan)

  kernel_mat = memory_kernel(input_mat)

  summed_kernel_vals = np.nansum(kernel_mat, axis=0)

  background_vals = np.squeeze(background_intensity(t_vals))

  return summed_kernel_vals + background_vals

# WORK IN PROGRESS
# def plot_trace(draws_df, param_string, warmup, num_chains=2):
#   plt.figure()
#   if warmup is not None:
#     plt.axvspan(xmin=0, xmax=warmup, color='gray', alpha=0.3, label='warmup')
#   plt.title(f"Trace plot for {param_string}")
#   plot_input_data = np.array([draws_df[draws_df['chain__'] == float(i)][param_string].values for i in range(num_chains)])
#   plot_labels = ["chain {i}" for i in range(num_chains)]
#   plt.plot(plot_input_data, labels=plot_labels)
#   plt.legend()
#   plt.show()
#   return None

def get_axs_temp_2d(axs, i, j, n, m):
    if n == 1 and m == 1:
        return axs
    elif n == 1:
        return axs[j]
    elif m == 1:
        return axs[i]
    else:
        return axs[i, j]

def trace_plots(fits, params, warmup=None, chains=2, legend_height=-0.01):
    n = len(fits)
    m = len(params)
    
    max_y = np.zeros(m)
    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 3*n))
    for i in range(n):
        df = fits[i].draws_pd(inc_warmup=(warmup is not None))
        for j in range(m):
            for k in range(1, chains+1):
                axs_temp = get_axs_temp_2d(axs, i, j, n, m)
                axs_temp.plot(df[df['chain__']==k][params[j]].values, label=f"Chain {k}")
            if warmup:
                axs_temp.axvspan(xmin=0, xmax=warmup, color='gray', alpha=0.3, label='Warm-up')
            #axs[i, j].legend()
            max_y[j] = max(max(df[df['chain__']==k][params[j]].values), max_y[j])
            if i==0:
                axs_temp.set_title(params[j])
                # axs_temp.set_title(f"$\\{params[j]}$")

    for i in range(n):
        for j in range(m):
            axs_temp = get_axs_temp_2d(axs, i, j, n, m)
            axs_temp.set_ylim(0, max_y[j])
            axs_temp.set_xlabel('Iteration')
            axs_temp.set_ylabel('Value')
            axs_temp.set_xlim(0, 1000)

    an_axs = get_axs_temp_2d(axs, 0, 0, n, m)
    handles, labels = an_axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, legend_height), ncol=m)
    
    plt.tight_layout()
    plt.show()
    
def posterior_histograms(fits, params, prior_functions=None, xlims=None, legend_height=-0.01):
    n = len(fits)
    m = len(params)
    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 3*n))
    max_x = np.zeros(m)
    max_y = np.zeros(m)
    
    # Plot histograms and KDE plots for posterior distributions
    for i in range(n):
        df = fits[i].draws_pd()
        for j in range(m):
            axs_temp = get_axs_temp_2d(axs, i, j, n, m)
            data = df[params[j]].values
            hist, bins, _ = axs_temp.hist(data, density=True, alpha=0.5, bins=30, color='blue')  # Plot histogram
            
            # Suppress specific future warning
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                sns.kdeplot(data, color='blue', ax=axs_temp, label='Posterior KDE')  # Overlay KDE plot on histogram

            axs_temp.set_xlabel('Value')

            credible_interval = stats.mstats.mquantiles(data, [0.025, 0.975])
            axs_temp.axvspan(xmin=credible_interval[0], xmax=credible_interval[1], color='green', alpha=0.2, label='95% CI')

            max_x[j] = max(max_x[j], max(data))
            max_y[j] = max(max_y[j], max(hist))
            
            if i == 0:
                axs_temp.set_title(params[j])  # Set title for the first row of subplots

    if xlims:
        for j in range(m):
            if xlims[j]:
                max_x[j] = xlims[j]

    # Set the same limits for all subplots
    for i in range(n):
        for j in range(m):
            axs_temp = get_axs_temp_2d(axs, i, j, n, m)
            axs_temp.set_xlim(0, max_x[j])
            axs_temp.set_ylim(0, max_y[j])
            if prior_functions:
                x_values = np.linspace(0, max_x[j], 1000)  # Generate x values for prior function evaluation
                prior_values = prior_functions[j](x_values)  # Evaluate prior density function
                axs_temp.plot(x_values, prior_values, color='red', linestyle='--', label='Prior')
                
    handles, labels = get_axs_temp_2d(axs, 0, 0, n, m).get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, legend_height), ncol=m)
    
    plt.tight_layout()
    plt.show()
    
def get_axs_temp_1d(axs, j, m):
    if m == 1:
        return axs
    else:
        return axs[j]
    
def posterior_kdes_overlaid(fits, params, prior_functions=None, xlims=None, legend_height=-0.01):
    n = len(fits)
    m = len(params)
    fig, axs = plt.subplots(nrows=1, ncols=m, figsize=(5*m, 3))
    max_x = np.zeros(m)


    for i in range(n):
        df = fits[i].draws_pd()
        for j in range(m):
            data = df[params[j]].values
            axs_temp = get_axs_temp_1d(axs, j, m)
            
            # Suppress specific future warning
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                sns.kdeplot(data, ax=axs_temp, alpha=0.5)
                
            axs_temp.set_xlabel('Value')

            if i == 0:
                axs_temp.set_title(params[j])
            max_x[j] = max(max(data), max_x[j])

    if xlims:
        for j in range(m):
            if xlims[j]:
                max_x[j] = xlims[j]

    # Optionally plot prior density functions
    if prior_functions is not None:
        for j, prior_func in enumerate(prior_functions):
            axs_temp = get_axs_temp_1d(axs, j, m)
            axs_temp.set_xlim(0, max_x[j])
            x_values = np.linspace(0, max_x[j], 1000)  # Generate x values for prior function evaluation
            prior_values = prior_func(x_values)  # Evaluate prior density function
            axs_temp.plot(x_values, prior_values, color='red', linestyle='--', label='Prior')  # Overlay prior distribution

        handles, labels = get_axs_temp_1d(axs, 0, m).get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, legend_height), ncol=m)
    
    plt.tight_layout()
    plt.show()
    
def stacked_credible_intervals(fits, params, true_params, prior_cis=None, xlims=None, legend_height=-0.01):
    n = len(fits)
    m = len(params)
    fig, axs = plt.subplots(nrows=1, ncols=m, figsize=(5*m, 3))
    max_x = np.zeros(m)
    max_y = np.zeros(m)

    data = np.zeros((n, m, 1500))

    for i in range(n):
        df = fits[i].draws_pd()
        for j in range(m):
            data = df[params[j]].values
            axs_temp = get_axs_temp_1d(axs, j, m)

            mean = np.mean(data)
            credible_interval = stats.mstats.mquantiles(data, [0.025, 0.975])

            if credible_interval[0] <= true_params[j] <= credible_interval[1]:
                color_temp = 'green'
            else:
                color_temp = 'red'

            axs_temp.scatter(mean, i, color=color_temp)
            axs_temp.plot(credible_interval, [i, i], color=color_temp, linestyle='-', linewidth=2, marker='|')

    if xlims:
        for j in range(m):
            if xlims[j]:
                max_x[j] = xlims[j]

    for j in range(m):
        axs_temp = get_axs_temp_1d(axs, j, m)

        if prior_cis:
            axs_temp.plot(prior_cis[j], [n, n], color='blue', linestyle='-', linewidth=2, marker='|', label="Prior")

        
        axs_temp.axvline(true_params[j], color='black', linestyle='--', label='True parameter value')
        axs_temp.invert_yaxis()
        axs_temp.set_title(params[j])  # Set title for the first row of subplots
        axs_temp.set_xlabel('Values')
        if n == 1:
            axs_temp.set_yticks([])
        else:
            axs_temp.set_yticks(np.arange(n))
            axs_temp.set_ylabel('Realisation')
        if xlims:
            if xlims[j]:
                axs_temp.set_xlim(0, max_x[j])

    handles, labels = axs_temp.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, legend_height), ncol=m)
    
    plt.tight_layout()
    plt.show()