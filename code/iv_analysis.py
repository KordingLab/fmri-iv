import numpy as np


def iv_betas(activations, instruments):
    """
    Estimate causal connection strengths using instrumental variable method.
    :param activations: NxM time series of activations for N neurons/regions
    :param instruments: NxM time series N instruments each of which corresponds to 1 region
    only directly affects its corresponding neuron (i.e. satisfies the IV criteria for
    that node of the causal graph).

    Typically, the instrument might be binary and represent whether its neuron is locally
    inhibited from firing, but any linear relationship between instrument and neuron should work.
    Instruments should satisfy the IV criteria for their corresponding neurons/regions, i.e.
    have no direct causal influence on other neurons.

    :return: a matrix of "beta" coefficients representing the extent of directed causal influence
    between each neuron and every other neuron at a later time step, using the IV method.
    beta[i, j] = estimated effect of region j on region i.
    """

    assert instruments.shape == activations.shape, "Activations and instruments must be the same shape"

    n_regions, n_times = activations.shape

    assert n_times >= 2, "Must have at least 2 timepoints for IV analysis"

    # define z (instrument), x (timepoint 1), and y (timepoint 2)
    z = instruments[:, :-1]
    x = activations[:, :-1]
    y = activations[:, 1:]

    # do 2SLS for each region
    beta = np.zeros((n_regions, n_regions))

    for kR1 in range(n_regions):
        w, b = np.linalg.lstsq(np.vstack([z[kR1], np.ones(n_times - 1)]).T, x[kR1], rcond=None)[0]
        x_pred = z[kR1] * w + b

        for kR2 in range(n_regions):
            beta[kR2, kR1], _ = np.linalg.lstsq(np.vstack([x_pred, np.ones(n_times - 1)]).T, y[kR2], rcond=None)[0]

    return beta


def delayed_iv_betas(activations, instruments):
    """
    Same as iv_betas, but use the instrument value from the *previous* rather than current timestep as the IV.
    This introduces a violation of the second IV condition, i.e. the instrument value from the previous
    timestep could affect y through paths that don't go through x (at the current timestep). However,
    it violates the IV conditions less than just using a previous value of x as the instrument,
    i.e. the pseudo-IV method below.

    :param activations: see iv_betas
    :param instruments: see iv_betas
    :return: A matrix of "beta" coefficients - beta[i, j] = estimated effect of region j on region i.
    """

    return iv_betas(activations[:, 1:], instruments[:, :-1])


def pseudo_iv_betas(activations, sd_threshold=2, log_transform_input=False):
    """
    Try to estimate causal connection strengths, using past activity of each neuron as its own
    "instrumental variable." This is unlikely to actually satisfy the IV criteria since the
    instrument will have the same outgoing influences as the variable itself, meaning it
    does probably affect other neurons in the network. Similarly, it is going to be influenced
    by other neurons in the network.

    :param activations: NxM time series of activations for N neurons/regions
    :param sd_threshold: instrument is "on" when past activity of each neuron is below
    its mean activation minus this number times the standard deviation of its activation.
    :param log_transform_input: if true, take the natural log of the activations before calculating the pseudo-IV.

    :return: a matrix of "beta" coefficients representing the extent of directed causal influence
    between each neuron and every other neuron at a later time step, using the IV method.
    beta[i, j] = estimated effect of region j on region i.
    """
    n_times, n_regions = activations.shape

    assert n_times >= 3, "Must have at least 3 timepoints for pseudo-IV analysis"

    # define instrument, x, and y
    act_t0 = activations[:, :-1]
    if log_transform_input:
        act_t0 = np.log(act_t0)

    threshold = np.mean(activations, axis=1, keepdims=True) - sd_threshold * np.std(activations, axis=1, keepdims=True)
    z = np.array(act_t0 < threshold, dtype=np.float64)

    return iv_betas(activations[:, 1:], z)


def norm_lagged_corr(activations):
    """
    Estimate connectivity matrix A from autocorrelation and covariance matrices
    :param activations: NxM time series of activations for N neurons/regions
    :return: estimation of A
    """

    _, n_times = activations.shape

    cov_mat = np.cov(activations)
    autocorr_mat = activations[:, 1:] @ activations[:, :-1].T / (n_times - 1)

    return autocorr_mat @ np.linalg.pinv(cov_mat)


# Also export dictionary of methods for convenience and consistency
methods = {
    'real': {
        'name': 'Real IV',
        'fn': iv_betas
    },
    'pseudo': {
        'name': 'Pseudo-IV',
        'fn': lambda log_act, ivs: pseudo_iv_betas(log_act)
    },
    'delayed': {
        'name': 'Delayed IV',
        'fn': delayed_iv_betas
    },
    'corr': {
        'name': 'Correlations',
        'fn': lambda log_act, ivs: np.corrcoef(log_act)
    },
    'lagged_corr': {
        'name': 'Normalized lag-1 correlations',
        'fn': lambda log_act, ivs: norm_lagged_corr(log_act)
    }
}


def all_methods_estimate(activations, instruments, observable=None, rois=None):
    """
    Estimate connectivity using all candidate methods (in dictionary above).
    Optionally restrict which variables are observable and which are returned in the output.

    :param activations: NxM timeseries of (log of) regional activations
    :param instruments: NxM timeseries of artificial (binary) instruments
    :param observable:  If not None, should be binary or numerical 1-D indices. Only use these variables in analysis.
    :param rois:        If not None, should be binary or numerical 1-D indices and this must correspond to a
                        subset of `observable` indices. Only return these variables.

    :return: Dictionary of connectivity estimates (keys match the keys of `methods` above)
    """
    instruments = np.array(instruments, dtype=np.float64)

    n_vars = activations.shape[0]
    all_vars = np.arange(n_vars)

    if observable is None:
        observable = all_vars
    else:
        observable = all_vars[observable]

    if rois is None:
        rois = all_vars
    else:
        rois = all_vars[rois]

    # ensure that rois is a subset of observable
    assert np.isin(rois, observable).all(), 'rois must be a subset of observable variables'

    # get which *of the observable* variables are of interest
    rois_rel = np.isin(observable, rois)

    return {key: method['fn'](activations[observable, :], instruments[observable, :])[np.ix_(rois_rel, rois_rel)]
            for key, method in methods.items()}
