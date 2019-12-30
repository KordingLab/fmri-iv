# fmri-iv

An attempt to examine fMRI connectivity using instrumental variables.

This began as Ethan Blackwood's rotation project in the Kording Lab (Fall 2019). For a summary of the main results obtained during this period, see [Blackwood_writeup.docx](Blackwood_writeup.docx).

For background on instrumental variables, see:

* Chapter 4 of *Mostly Harmless Econometrics* by J. Angrist and J. Pischke.
* ["Inferring causal connectivity from pairwise recordings and optogenetics"](http://biorxiv.org/lookup/doi/10.1101/463760), M. Lepperød, T. Stöber, T. Hafting, M. Fyhn, and K. Kording.

## Motivation (written by Konrad)

Neuroscience is supremely interested in identifying the causal influences that brain areas have upon one another (effective connectivity) from fMRI recordings. Such causal inferences are only possible under a set of well understood assumptions. Here we derive how causality could be inferred under a specific assumption about activity in brain areas: that they undergo periods of hyperpolarization that are not affected by network activity and can be detected in MRI.  We can then use so-called instrumental variables to identify causality. Here we derive the relevant estimators, apply them to an fMRI dataset but ultimately show that the necessary assumptions are not satisfied. We thus highlight a clean cautionary tale of how strong assumptions are needed for causality and are often not satisfied.

## Data summary

The analyses done during the rotation project mainly consisted of simulations, followed by comparing estimates with the expected connectivity (as described in [Blackwood_writeup.docx](Blackwood_writeup.docx). However, I did find a dataset of real fMRI data that seems best suited to further analyses, located [here](https://neurodata.io/mri/) under "BNU2." The plan is to use just the "ROI Timeseries" data. A minimal test of this is located in `test_fmri_analysis.ipynb`.

## Code summary

### Python files:

* `iv_analysis.py`: Implements IV analysis by 2-stage least squares (`iv_betas`), as well as some other IV variations (including the "pseudo-IV" that uses past values of X to produce an instrument) and other connectivity estimation methods. These can be systematically applied to a given dataset with IVs using the `methods` dictionary.

* `network_sim.py`: Tools for simulating a first-order multivariate autoregressive process, stochastically perturbed by an instrumental variable with customizable probability and effect, which may stand in for regional activations as measured by fMRI. The overall flow is to use `gen_con_mat` to produce a random connectivity matrix of a given size, etc., and then pass that into the `sim_network` function to produce activation and iv timeseries.
  * If provided, the `snr` argument to `sim_network` rescales the input matrix so that the expected ratio of carried-over to total variance at each step is close to the given value. However, note that the complexity of estimating SNR for a given matrix is about O(n^5) in the number of variables, so use with caution.

* `smallworld.py`: Code from the internet to generate a small-world network, which is an option in `network_sim.gen_con_mat`.

* `stats_util.py`: Contains the function `get_medoid_ind`, which is used in a simulation with many trials to select the one whose scores on a set of statistics are closest to the multidimensional median.

* `get_data.py`: Currently only downloads the AAL atlas labels from the neurodata.io website. I haven't yet found a systematic way to download the datasets themselves, if we want to combine many of them into an analysis.

### Jupyter notebooks:

* `test_iv_analysis.ipynb`: Preliminary tests of `iv_analysis` and `network_sim` functions, without doing many trials. Grew organically as I added more functionality to these files.

* `multi_network_sim_tests.ipynb`: This contains the "parameter exploration" described in the first paragraph of the results section of the writeup. Produces the data file `data_sim/iv_rs_w_seed.npz`. Tests the effects of changing gain, SNR, and state transitions per timepoint on estimate accuracies.

* `sparse_networks_sim_tests.ipynb`: This analyzes the effects of making the simulated network sparsely observable (shown in Figure 1 of the rotation writeup). Produces the data file `data_sim/est_corr_sparse.npz`.

* `test_fmri_analysis.ipynb`: Preliminary test of using IV method on real fMRI data (ROI timeseries).
