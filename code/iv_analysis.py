import numpy as np


def iv_betas(activations, instruments):
    """
    Estimate causal connection strengths using instrumental variable method.
    :param activations: M x N time series of activations for N neurons/regions
    :param instruments: M x N time series N instruments each of which corresponds to 1 region
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

    n_times, n_regions = activations.shape

    assert n_times >= 2, "Must have at least 2 timepoints for IV analysis"

    # define z (instrument), x (timepoint 1), and y (timepoint 2)
    z = instruments[:-1, :]
    x = activations[:-1, :]
    y = activations[1:, :]

    # demean for regression and split along columns
    zs = np.hsplit(z - np.mean(z, axis=0), n_regions)
    xs = np.hsplit(x - np.mean(x, axis=0), n_regions)
    ys = np.hsplit(y - np.mean(y, axis=0), n_regions)

    # do 2SLS for each region
    beta = np.zeros((n_regions, n_regions))

    for kR1 in range(n_regions):
        w = np.linalg.lstsq(zs[kR1], xs[kR1], rcond=None)[0]
        x_pred = zs[kR1] * w

        for kR2 in range(n_regions):
            beta[kR2, kR1] = np.linalg.lstsq(x_pred, ys[kR2], rcond=None)[0]

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

    return iv_betas(activations[1:, :], instruments[:-1, :])


def pseudo_iv_betas(activations, sd_threshold=2, log_transform_input=False):
    """
    Try to estimate causal connection strengths, using past activity of each neuron as its own
    "instrumental variable." This is unlikely to actually satisfy the IV criteria since the
    instrument will have the same outgoing influences as the variable itself, meaning it
    does probably affect other neurons in the network. Similarly, it is going to be influenced
    by other neurons in the network.

    :param activations: M x N time series of activations for N neurons/regions
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
    act_t0 = activations[:-1, :]
    if log_transform_input:
        act_t0 = np.log(act_t0)

    z = np.array(act_t0 < np.mean(activations, axis=0) - sd_threshold * np.std(activations, axis=0),
                 dtype=np.float64)

    return iv_betas(activations[1:, :], z)


def davids_method(activations):
    """
    Estimate connectivity matrix A from autocorrelation and covariance matrices
    :param activations: M x N time series of activations for N neurons/regions
    :return: estimation of A
    """

    n_times, _ = activations.shape

    cov_mat = activations.T @ activations / n_times
    autocorr_mat = activations[1:].T @ activations[:-1] / (n_times - 1)

    return autocorr_mat @ np.linalg.pinv(cov_mat)
