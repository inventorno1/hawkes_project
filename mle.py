import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import exp_kernel_vectorised

def log_likelihood(mu, alpha, delta, events_list, max_T):
  
  # first term

  events_list = np.array(events_list)
  events_list = events_list[:, np.newaxis]

  mask = events_list < max_T
  integral_sum_terms = np.exp(delta*events_list)
  integral_sum_terms = np.where(mask, integral_sum_terms, 0)

  first = mu*max_T - (alpha/delta) * (np.exp(-delta*max_T) - 1) \
    * np.sum(integral_sum_terms)
  
  # second term

  differences_mat = np.tril(events_list - events_list.T)



def log_likelihood(params, data):

  mu, alpha, delta = params
  T_list, T = data

  temp = 0
  for T_i in T_list:
    temp += (np.exp(delta*T_i)*(np.exp(-delta*T - 1)))
  first = mu * T - (alpha/delta) * temp

  second = 0
  for T_i in T_list:
    second += np.log(exp_kernel_vectorised(T_i, T_list, mu, alpha, delta))

  return -first + second