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
  num_events = len(events_list)
  if isinstance(t_vals, float) or isinstance(t_vals, int):
     num_t_vals = 1
  else:
     num_t_vals = len(t_vals)
  events_list = np.array(events_list)
  events_list = events_list[np.newaxis, :] # Make events_list a row vector
  t_vals = t_vals[:, np.newaxis] # Make t_vals a column vector
  print(events_list)
  print(t_vals)

  mask = np.where(events_list<t_vals, 1, 0)
  print(mask)

  kernel_vals = memory_kernel(events_list)
  print(kernel_vals)

  kernel_mat = np.tile(kernel_vals, (num_t_vals, 1))
  print(kernel_mat)
  masked_kernel_mat = kernel_mat * mask
  print(masked_kernel_mat)
  summed_kernel_vals = np.sum(masked_kernel_mat, axis=1)
  print(summed_kernel_vals)

  background_vals = np.squeeze(background_intensity(t_vals))
  print(background_vals)

  return summed_kernel_vals + background_vals