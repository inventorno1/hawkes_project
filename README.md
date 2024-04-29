# hawkes_project - Please read

This Git repository contains the code developed for the project 'Hawkes Processes for Infectious Disease Modelling', alongside saved results of particular experiments that were run in this project.

The code is split into two main categories. The `.py` files are where all functions have been developed, whereas `.ipynb` are Python notebooks in which experiments have been run using those functions.

We briefly explain what is in each file and folder.

## `.py` files
These files contain functions that do the following.
* `clustering.py` - implement clustering algorithm for simulating Hawkes process.
* `mle.py` - compute the log-likelihood function and perform maximum likelihood estimation.
* `thinning.py` - implement thinning algorithm for simulating Hawkes processes.
* `time-rescaling.py` - implement the Time-Rescaling Theorem for the verification of correctness of simulations.
* `utils.py`
  * compute the conditional intensity function
  * custom plotting functions, particularly for Bayesian inference
  * processes real-world data such as from Ebola virus disease outbreak

## `.ipynb` files
These correspond to different sets of experiments in the project report.

1. Data simulation and verification (Sections 5.1, 6.1) - `basic2.ipynb`
2. Maximum likelihood estimation (Sections 5.2, 6.2) - `mle3.ipynb`
3. Bayesian inference (Sections 5.3, 6.3) - inside the `bayesian_clean` folder
   * Initial test run of Stan - `initial.ipynb`
   * Increasing period of observation - `increasing_observation_window.ipynb`
   * Investigating missingness - `missingness.ipynb`
   * Stan models - `models` folder
4. Fitting to real data (Sections 5.4, 6.4) - `data_fitting.ipynb`

The reproducibility of our code was tested in `reproducible.ipynb`. Vectorised versus naive implementations of the thinning simulator were tested and timed in `vectorise.ipynb`.

## Saved results

Some experiments took considerably long times to run, in which case their code cells have been commented out and the output has been saved. Code cells were then added to the Python notebooks for the proper reloading of the saved results.
* The saved results of MLE can be found in `param_estimates_march11.npy`, `missing_data_param_estimates_march11.npy`, `missingness_fixed_p_varying_end_march12.npy` and `missingness_fixed_p_varying_start_march12.npy`.
* The saved results of Bayesian inference can be found in `bayesian_clean/saved_fits`.
* The saved results of fitting to real data can be found in `data_fitting/saved_fits`.

## Data for 'Fitting to real data'

There is an Excel file in `data_fitting/data/` with the downloaded and extracted data for the 'Fitting to real data' experiment, called `ebola_data_with_first_800.xlsx`. The data for each country was then further extracted into separate `.csv` files in the same folder.
