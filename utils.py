import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    plt.xlabel('Time')
    plt.ylabel('Counting Process')
    plt.title('Counting Process Plot')

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