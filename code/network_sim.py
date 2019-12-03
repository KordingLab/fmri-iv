import numpy as np
from typing import Union


def sim_network(con_mat: Union[list, np.ndarray],
                n_time: int = 10000,
                iv_prob: Union[float, list, np.ndarray] = 0.025,
                iv_gain: Union[float, list, np.ndarray] = 0.1,
                snr: Union[float, list, np.ndarray] = 10.):
    """
    Simulate a general log-linear recurrent network of M neurons or regions.
    The log of activations at each timestep is equal to the
    sum of a linear transformation of the previous log-activations, Gaussian
    noise, and possibly a negative deflection due to artificial intervention.
    Thus, the activations are expected to have a log-normal distribution overall.

    An artificial intervention will occur on each variable (independently)
    at each timestep with a probability of iv_prob. When this happens, the
    activation at that point is iv_gain times what it would otherwise be
    (i.e. ln(iv_gain) is added to the log-activation). This can then be used
    to perform an IV analysis and recover the connectivity matrix.

    :param con_mat: MxM array-like of connection strengths.
                    Note that the largest eigenvalue should be less than 1 in magnitude
                    to avoid instability and unit roots.
    :param n_time:  Number of timepoints (N)
    :param iv_prob: Probability of IV event at each time for each variable
                    (can be scalar or an array of length M)
    :param iv_gain: Factor to scale activation when IV event occurs
                    (can be scalar or an array of length M)
    :param snr:     Signal-to-noise ratio of each variable (scalar or array of length M)

    :return: tuple of:
        NxM double ndarray of ln(activations) over time
        NxM boolean ndarray of IV events
    """

    # Argument checks
    con_mat = np.array(con_mat)
    assert con_mat.ndim == 2, 'Connectivity matrix must be a matrix'
    n_vars, n_cols = con_mat.shape
    assert n_vars >= 2, 'Must have at least 2 regions (2x2 connectivity)'
    assert n_vars == n_cols, 'Connectivity matrix must be square'
    assert np.isreal(con_mat).all(), 'Connectivity matrix must be real'
    assert max(abs(np.linalg.eig(con_mat)[0])) < 1.0,\
        'System is nonstationary - make sure all eigenvalues are < 1 in absolute value'

    assert isinstance(n_time, int) and n_time >= 1, 'N times must be an int and at least 1'

    assert np.all(np.isreal(iv_prob) & (0 <= iv_prob) & (iv_prob <= 1)),\
        'IV probability must be real and between 0 and 1'
    if not np.isscalar(iv_prob):
        iv_prob = np.array(iv_prob)
        assert iv_prob.shape == (n_vars,),\
            'IV probability must be a scalar or a vector of length M (number of regions)'

    assert np.all(np.isreal(iv_gain)), 'IV gain must be real'
    if not np.isscalar(iv_gain):
        iv_gain = np.array(iv_gain)
        assert iv_gain.shape == (n_vars,),\
            'IV gain must be a scalar or a vector of length M (number of regions)'

    assert np.all(np.isreal(snr)), 'SNR must be real'
    if np.isscalar(snr):
        snr = np.repeat(snr, n_vars)
    else:
        snr = np.array(snr)
        assert snr.shape == (n_vars,),\
            'SNR must be a scalar or a vector of length M (number of regions)'

    # Generate IVs with given probability at each timepoint
    ivs = np.random.rand(n_time, n_vars) < iv_prob

    # Initialize log of network activations as Gaussian
    log_act = np.zeros((n_time, n_vars))
    log_act[0, :] = np.random.randn(n_vars) + ivs[0, :] * np.log(iv_gain)

    # generate additive noise
    cov = np.diag(1 / snr)
    noise = np.random.multivariate_normal(np.zeros(n_vars), cov, size=n_time)

    # run the network
    for kT in range(1, n_time):
        log_act[kT, :] = con_mat @ log_act[kT - 1, :] + noise[kT, :] + ivs[kT, :] * np.log(iv_gain)

    return log_act, ivs


def gen_con_mat(n_vars: int, max_eig: float = 0.9):
    """
    Generate a random stable connectivity matrix. Coefficients are chosen uniformly at
    random with a mean of 0 and then scaled to have the requested maximum eigenvalue.

    :param n_vars:  Number of variables N
    :param max_eig: Absolute value of max eigenvalue
    :return: N x N connectivity matrix
    """

    assert n_vars >= 1, 'Number of variables must be at least 1'
    assert 0 < max_eig < 1, 'Max eigenvalue should be between 0 and 1'

    mat = np.random.rand(n_vars, n_vars) - 0.5
    curr_max_eig = max(abs(np.linalg.eig(mat)[0]))

    return mat * max_eig / curr_max_eig
