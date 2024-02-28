import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import exp_kernel_vectorised, conditional_intensity_true_vectorised

# vectorised log_likelihood function for constant background

def log_likelihood(params, data):

  mu, alpha, delta = params
  events_list, max_T = data
  
  # first term

  events_list = np.array(events_list)
  events_list = events_list[np.newaxis, :]

  differences_from_max_T = max_T - events_list
  summands = np.exp(-delta*differences_from_max_T) - 1

  within_max_T_mask = differences_from_max_T > 0
  summands = summands * within_max_T_mask

  first = mu*max_T - (alpha/delta) * np.sum(summands)
  
  # second term

  differences_mat = np.tril(events_list.T - events_list)

  inner_sum_mat = np.exp(-delta*differences_mat)
  inner_sum_mat = np.tril(inner_sum_mat, k=-1)
  # IMPORTANT - should k=0 or -1 here?
  # Think -1 otherwise you include differences of each arrival between itself

  term_inside_log = mu + alpha*np.sum(inner_sum_mat, axis=1)

  second_sum_terms = np.log(term_inside_log)

  second = np.sum(second_sum_terms)

  return -first + second

# OLD likelihood function
def old_log_likelihood(params, data):

  mu, alpha, delta = params
  T_list, T = data

  temp = 0
  for T_i in T_list:
    temp += (np.exp(delta*T_i)*(np.exp(-delta*T) - 1))
  first = mu * T - (alpha/delta) * temp

  second = 0
  for T_i in T_list:
    second += np.log(conditional_intensity_true_vectorised(T_i, T_list, lambda x: mu*np.ones_like(x), lambda x: exp_kernel_vectorised(x, alpha, delta)))

  return -first + second