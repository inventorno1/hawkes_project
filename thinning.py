import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import conditional_intensity, conditional_intensity_vectorised, constant_background, exp_kernel, exp_kernel_vectorised

# Simulation by thinning

def sample_hawkes_process_thinning(T_max, background_intensity, memory_kernel):

  T = 0
  events_list = []

  while T < T_max:
    lambda_star = conditional_intensity(T, events_list, constant_background, exp_kernel)
    u = np.random.uniform()
    tau = -np.log(u)/lambda_star
    T += tau
    s = np.random.uniform()
    lambda_T = conditional_intensity(T, events_list, constant_background, exp_kernel)
    if s <= lambda_T/lambda_star:
      events_list.append(T)

  return events_list

def sample_hawkes_process_thinning_vectorised(T_max, background_intensity, memory_kernel):

  T = 0
  events_list = []

  while T < T_max:
    lambda_star = conditional_intensity_vectorised(T, events_list, constant_background, exp_kernel)
    u = np.random.uniform()
    tau = -np.log(u)/lambda_star
    T += tau
    s = np.random.uniform()
    lambda_T = conditional_intensity_vectorised(T, events_list, constant_background, exp_kernel)
    if s <= lambda_T/lambda_star:
      events_list.append(T)

  return events_list