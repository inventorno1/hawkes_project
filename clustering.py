import numpy as np
import matplotlib.pyplot as plt
import scipy

from utils import exp_kernel, exp_kernel_vectorised

# Simulate homogeneous Poisson process - interarrival times are Exponential
def simulate_homogeneous_poisson_process(mu, max_T):
    T_list = []
    rng = np.random.default_rng()
    latest_T = 0
    while latest_T < max_T:
        interarrival_time = rng.exponential(1/mu)
        latest_T += interarrival_time
        if latest_T < max_T:
            T_list.append(latest_T)
    return T_list


# Simulate inhomogeneous Poisson process - need to use accept-reject method
def simulate_inhomogeneous_poisson_process(rate, max_T, s):
    T_list = []
    
    rng = np.random.default_rng()
    lambda_bar = rate(s)
    while s < max_T:
        u1 = rng.uniform()
        w = -np.log(u1)/lambda_bar
        s += w
        D = rng.uniform() 
        if D <= rate(s)/lambda_bar:
            latest_T = s
            if latest_T < max_T:
                T_list.append(latest_T)

    return T_list


def simulate_immigrants(max_T):
  T_list = []
  mu = 1
  rng = np.random.default_rng()
  latest_T = 0
  while latest_T < max_T:
    interarrival_time = rng.exponential(mu)
    latest_T += interarrival_time
    if latest_T < max_T:
      T_list.append(latest_T)
  return T_list

def simulate_offspring(event_time, max_T):
  T_list = []
  rate = lambda t: exp_kernel_vectorised(t-event_time)
  rng = np.random.default_rng()
  s = event_time
  lambda_bar = rate(event_time)
  while s < max_T:
    u1 = rng.uniform()
    w = -np.log(u1)/lambda_bar
    s += w
    D = rng.uniform()
    if D <= rate(s)/lambda_bar:
      latest_T = s
      if latest_T < max_T:
        T_list.append(latest_T)
  return T_list

def produce_cluster(immigrant, max_T):
  cluster = {}
  generation = 0
  cluster[generation] = [immigrant]
  offspring = simulate_offspring(immigrant, max_T)
  if (len(offspring) == 0) or (min(offspring) > max_T):
    return cluster
  generation += 1
  cluster[generation] = offspring
  continue_cluster=True
  while continue_cluster:
    new_offspring = []
    for event in offspring:
      new_offspring_part = simulate_offspring(event, max_T)
      new_offspring = new_offspring + new_offspring_part
    if (len(new_offspring) == 0) or (min(new_offspring) > max_T):
      continue_cluster = False
      return cluster
    generation += 1
    offspring = new_offspring
    cluster[generation] = offspring
  return cluster

def sample_hawkes_process_clustering(max_T):
  immigrants = simulate_immigrants(max_T)
  clusters = {}
  for i, immigrant in enumerate(immigrants):
    cluster = produce_cluster(immigrant, max_T)
    clusters[i] = cluster
  return clusters

def cluster_plot(clusters):
  # Generate unique colors for each immigrant
  colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))

  counter = 0

  for i in range(len(clusters)):
    cluster = clusters[i]
    color = colors[i]
    plt.scatter(cluster[0], 0, color=color)
    for generation, events in cluster.items():
      for event in events:
        counter += 1 # Should only be incrementing counter here!
        plt.scatter(event, generation, color=color)

  print(counter)