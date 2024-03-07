# TO SET UP/INSTALL Cmdstan

# import cmdstanpy
# # cmdstanpy.install_cmdstan()
# cmdstanpy.install_cmdstan(compiler=True)  # only valid on Windows

import os
from cmdstanpy import CmdStanModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# THIS IS THE TRUE FILE LOCATION FOR THE EXAMPLES!!
# stan_file = os.path.join('examples', 'bernoulli', 'bernoulli.stan')

# print(stan_file)

# model = CmdStanModel(stan_file=stan_file)

# from cmdstanpy import cmdstan_path

# print(cmdstan_path())

# Use following to load from CmdStanModel output csv file
# df = pd.read_csv(path, comment='#')

### AIMS Rwanda workshop - 4 Hello world

# stan_file = os.path.join('.', 'model1.stan')
# model1_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

# rng = np.random.default_rng()

# N = 100
# y = rng.normal(loc=0, scale=1, size=N)

# model1_data = {
#     "N" : N,
#     "y" : y
# }

# model1_inits = [{"mu": 1, "sigma": 2}, {"mu": -1, "sigma": 0.5}]

# # pass as output_dir below
# # output_folder = r'C:\Users\ethan\OneDrive - University of Bristol\Maths\Year 4\Project\hawkes_project\stan_test_output'

# fit = model1_compiled.sample(data=model1_data,
#                        seed=123,
#                        chains=2,
#                        iter_warmup=1000,
#                        iter_sampling=3000,
#                        inits=model1_inits,
#                        refresh=500,
#                        show_console=True,
#                        save_warmup=True)

# df = fit.draws_pd(inc_warmup=True)

# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# plt.plot(df_chain1['mu'].values, label=1)
# plt.plot(df_chain2['mu'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('mu')
# plt.show()

# plt.figure()
# plt.plot(df_chain1['sigma'].values, label=1)
# plt.plot(df_chain2['sigma'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('sigma')
# plt.show()

# plt.figure()
# sns.pairplot(df_chain1[['lp__', 'mu', 'sigma']])
# plt.show()

# plt.figure()
# sns.pairplot(df_chain2[['lp__', 'mu', 'sigma']])
# plt.show()

### AIMS Rwanda workshop - 5 Flat priors example

# y = np.array([-1,1])
# N = len(y)

# model2_data = {
#     "N" : N,
#     "y" : y
# }

# stan_file = os.path.join('.', 'model2.stan')
# model2_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

# model2_inits = [{"mu": 0, "sigma": 1}, {"mu": 0, "sigma": 1}]

# fit = model2_compiled.sample(data=model2_data,
#                        seed=123,
#                        chains=2,
#                        iter_warmup=1000,
#                        iter_sampling=3000,
#                        inits=model2_inits,
#                        refresh=500,
#                        show_console=True,
#                        save_warmup=True)

# df = fit.draws_pd(inc_warmup=True)

# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# plt.plot(df_chain1['mu'].values, label=1)
# plt.plot(df_chain2['mu'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('mu')
# plt.show()

# plt.figure()
# plt.plot(df_chain1['sigma'].values, label=1)
# plt.plot(df_chain2['sigma'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('sigma')
# plt.show()

## Weakly informative priors

# y = np.array([-1,1])
# N = len(y)

# model3_data = {
#     "N" : N,
#     "y" : y
# }

# stan_file = os.path.join('.', 'model3.stan')
# model3_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

# model3_inits = [{"mu": 0, "sigma": 1}, {"mu": 0, "sigma": 1}]

# fit = model3_compiled.sample(data=model3_data,
#                        seed=123,
#                        chains=2,
#                        iter_warmup=1000,
#                        iter_sampling=3000,
#                        inits=model3_inits,
#                        refresh=500,
#                        show_console=True,
#                        save_warmup=True)

# df = fit.draws_pd(inc_warmup=True)

# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# plt.plot(df_chain1['mu'].values, label=1)
# plt.plot(df_chain2['mu'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('mu')
# plt.show()

# plt.figure()
# plt.plot(df_chain1['sigma'].values, label=1)
# plt.plot(df_chain2['sigma'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('sigma')
# plt.show()

### AIMS Rwanda workshop - 6 Unidentifiable parameters example

# y = np.array([-1,1])
# N = len(y)

# model4_data = {
#     "N" : N,
#     "y" : y
# }

# stan_file = os.path.join('.', 'model4.stan')
# model4_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

# model4_inits = [{"alpha1": 0, "alpha2": 0,"sigma": 1}, {"alpha1": 0, "alpha2": 0,"sigma": 1}]

# fit = model4_compiled.sample(data=model4_data,
#                        seed=123,
#                        chains=2,
#                        iter_warmup=1000,
#                        iter_sampling=3000,
#                        inits=model4_inits,
#                        refresh=500,
#                        show_console=True,
#                        save_warmup=True)

# df = fit.draws_pd(inc_warmup=True)

# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# plt.plot(df_chain1['mu'].values, label=1)
# plt.plot(df_chain2['mu'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('mu')
# plt.show()

# plt.figure()
# plt.plot(df_chain1['sigma'].values, label=1)
# plt.plot(df_chain2['sigma'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('sigma')
# plt.show()

## With weakly informative priors

# y = np.array([-1,1])
# N = len(y)

# model5_data = {
#     "N" : N,
#     "y" : y
# }

# stan_file = os.path.join('.', 'model5.stan')
# model4_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

# model5_inits = [{"alpha1": 0, "alpha2": 0,"sigma": 1}, {"alpha1": 0, "alpha2": 0,"sigma": 1}]

# fit = model4_compiled.sample(data=model5_data,
#                        seed=123,
#                        chains=2,
#                        iter_warmup=1000,
#                        iter_sampling=3000,
#                        inits=model5_inits,
#                        refresh=500,
#                        show_console=True,
#                        save_warmup=True)

# df = fit.draws_pd(inc_warmup=True)

# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# plt.plot(df_chain1['mu'].values, label=1)
# plt.plot(df_chain2['mu'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('mu')
# plt.show()

# plt.figure()
# plt.plot(df_chain1['sigma'].values, label=1)
# plt.plot(df_chain2['sigma'].values, label=2)
# plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
# plt.legend()
# plt.title('sigma')
# plt.show()

# df = fit.draws_pd()
# df_chain1 = df[df['chain__'] == 1.0]
# df_chain2 = df[df['chain__'] == 2.0]

# plt.figure()
# sns.pairplot(df_chain1[['alpha1', 'alpha2', 'sigma', 'mu', 'lp__']])
# plt.show()

# plt.figure()
# sns.pairplot(df_chain2[['alpha1', 'alpha2', 'sigma', 'mu', 'lp__']])
# plt.show()

### AIMS Rwanda - 7 Redo with CmdStan

stan_file = os.path.join('.', 'model1.stan')
model1_compiled = CmdStanModel(stan_file=stan_file) # Compile and load model

rng = np.random.default_rng()

N = 100
y = rng.normal(loc=0, scale=1, size=N)

model1_data = {
    "N" : N,
    "y" : y
}

model1_inits = [{"mu": 1, "sigma": 2}, {"mu": -1, "sigma": 0.5}]

fit = model1_compiled.sample(data=model1_data,
                       seed=123,
                       chains=2,
                       iter_warmup=1000,
                       iter_sampling=3000,
                       inits=model1_inits,
                       refresh=500,
                       show_console=True,
                       save_warmup=True)


print(fit.summary(percentiles=[2.75, 97.5]))

df = fit.draws_pd(inc_warmup=True)

df_chain1 = df[df['chain__'] == 1.0]
df_chain2 = df[df['chain__'] == 2.0]

plt.figure()
plt.plot(df_chain1['mu'].values, label=1)
plt.plot(df_chain2['mu'].values, label=2)
plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('mu')
plt.show()

plt.figure()
plt.plot(df_chain1['sigma'].values, label=1)
plt.plot(df_chain2['sigma'].values, label=2)
plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('sigma')
plt.show()

plt.figure()
plt.plot(df_chain1['lp__'].values, label=1)
plt.plot(df_chain2['lp__'].values, label=2)
plt.axvspan(xmin=0, xmax=1000, color='gray', alpha=0.3, label='warmup')
plt.legend()
plt.title('lp__')
plt.show()

df = fit.draws_pd()
df_chain1 = df[df['chain__'] == 1.0]
df_chain2 = df[df['chain__'] == 2.0]

plt.figure()
sns.pairplot(df_chain1[['mu', 'sigma', 'lp__']])
plt.show()

plt.figure()
sns.pairplot(df_chain2[['mu', 'sigma', 'lp__']])
plt.show()

## From cmdstanpy Hello World

# print(fit.stan_variable('mu'))

# print(fit.draws_pd('mu')[:5])

# print(fit.draws_xr('mu'))

# for k, v in fit.stan_variables().items():
#     print(f'{k}\t{v.shape}')

# for k, v in fit.method_variables().items():
#     print(f'{k}\t{v.shape}')

# print(f'numpy.ndarray of draws: {fit.draws().shape}')

# print(fit.draws_pd())

# print(fit.metric_type)

# print(fit.metric)

# print(fit.step_size)

# print(fit.metadata.cmdstan_config['model'])

# print(fit.metadata.cmdstan_config['seed'])

# print(fit.summary())

# Next line takes 20ish seconds
# print(fit.diagnose())

