import os
from cmdstanpy import CmdStanModel
import numpy as np
import matplotlib.pyplot as plt

from thinning import sample_hawkes_process_thinning_true_vectorised
from utils import exp_kernel_vectorised, constant_background, plot_trace

stan_file = os.path.join('.', 'simple_exp_hawkes.stan')
model_compiled = CmdStanModel(stan_file=stan_file)

max_T = 100
hawkes_realisation = sample_hawkes_process_thinning_true_vectorised(max_T, constant_background, exp_kernel_vectorised)
N = len(hawkes_realisation)

data = {
    "N" : N,
    "events_list" : hawkes_realisation,
    "max_T" : max_T
}

inits = {"mu": 0.5, "alpha": 0.5, "delta": 0.5}

warmup=1000

fit = model_compiled.sample(data=data,
                            seed=123,
                            chains=2,
                            iter_warmup=warmup,
                            iter_sampling=3*warmup,
                            inits=inits,
                            refresh=500,
                            show_console=True,
                            save_warmup=True)

df = fit.draws_pd(inc_warmup=True)

# for param in ['mu', 'alpha', 'delta']:
#     plot_trace(df, param, warmup)

df_chain1 = df[df['chain__'] == 1.0]
df_chain2 = df[df['chain__'] == 2.0]

plt.figure()
plt.plot(df_chain1['mu'].values, label=1)
plt.plot(df_chain2['mu'].values, label=2)
plt.axvspan(xmin=0, xmax=warmup, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('mu')
plt.show()

plt.figure()
plt.plot(df_chain1['alpha'].values, label=1)
plt.plot(df_chain2['alpha'].values, label=2)
plt.axvspan(xmin=0, xmax=warmup, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('alpha')
plt.show()

plt.figure()
plt.plot(df_chain1['delta'].values, label=1)
plt.plot(df_chain2['delta'].values, label=2)
plt.axvspan(xmin=0, xmax=warmup, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('delta')
plt.show()

print(fit.summary())

print(fit.diagnose())