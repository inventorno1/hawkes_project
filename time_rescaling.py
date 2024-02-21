import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import conditional_intensity_vectorised, constant_background, exp_kernel, exp_kernel_vectorised
from thinning import sample_hawkes_process_thinning

def integrate_conditional_intensity_vectorised(t_start, t_end, events_list, background_intensity, memory_kernel):
    # Generate a vector of time points within the specified range
    t_values = np.linspace(t_start, t_end, round((t_end-t_start)*100))

    # Evaluate the conditional intensity function for the given time points
    conditional_intensity_values = conditional_intensity_vectorised(t_values, events_list, background_intensity, memory_kernel)

    # Use np.trapz to perform numerical integration
    integral_value = np.trapz(conditional_intensity_values, t_values)

    return integral_value


hawkes_realisation = sample_hawkes_process_thinning(1000, constant_background, exp_kernel)
taus_list = []
hawkes_realisation_plus_zero = [0] + hawkes_realisation
for i in range(len(hawkes_realisation)):
  start = hawkes_realisation_plus_zero[i]
  end = hawkes_realisation_plus_zero[i+1]
  tau = integrate_conditional_intensity_vectorised(start, end, hawkes_realisation, constant_background, exp_kernel_vectorised)
  taus_list.append(tau)

z_list = [(1 - np.exp(-tau)) for tau in taus_list]

plt.hist(z_list, bins=10, density=True)