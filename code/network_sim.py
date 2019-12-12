import numpy as np
from typing import Union


def activation_cov(con_mat, noise_cov):
    """
    Compute the expected steady-state covariance matrix of activations in the network with
    given connectivity matrix and noise covariance.

    :param con_mat: Connectivity matrix (A)
    :param noise_cov: Covariance matrix of the additive noise
    :return: Covariance matrix of (log-)activations
    """

    nvars, ncols = con_mat.shape
    assert nvars == ncols, 'Connectivity matrix must be square'

    # set up the tensor equation noise_cov = D @ activation_cov
    # this is based on the relation: A @ activation_cov @ A.T + noise_cov = activation_cov
    d = np.eye(nvars * nvars)
    d = np.reshape(d, (nvars,) * 4)

    d = d - np.einsum('ik,jl->ijkl', con_mat, con_mat)

    # get activation covariance by solving
    return np.linalg.tensorsolve(d, noise_cov)


def steady_snr(con_mat, noise_cov):
    """
    Estimate the expected steady-state SNR given connectivity matrix and noise covariance

    :param con_mat: Connectivity matrix (A)
    :param noise_cov: Covariance matrix of the additive noise
    :return: Vector of SNR for each variable
    """

    return np.diag(activation_cov(con_mat, noise_cov)) / np.diag(noise_cov) - 1


def steady_sxr(con_mat, noise_cov):
    """
    Like signal-to-noise ratio, except compare the signal to signal plus noise rather
    than just the noise. Thus it is between 0 and 1. Has a more straightforward relationship
    to the scale/spectral radius of con_mat.

    :param con_mat: Connectivity matrix (A)
    :param noise_cov: Covariance matrix of the additive noise
    :return: Vector of signal over x for each variable
    """
    return 1 - np.diag(noise_cov) / np.diag(activation_cov(con_mat, noise_cov))


def rescale_to_snr(con_mat, noise_cov, snr, max_iters=10, stop_error=0.001, print_info=False):
    """
    Try to rescale the given connectivity matrix so that the maximum SNR of all variables
    matches the input snr. Uses an iterative process; the number of iterations and stopping
    criterion can be controlled.

    :param con_mat: Connectivity matrix (A)
    :param noise_cov: Covariance matrix of the additive noise
    :param snr: Target signal-to-noise ratio
    :param max_iters: Maximum iterations of the matrix-rescaling process
    :param stop_error: Process stops when the signal-to-x ratio (snr/(snr+1)) is within this amount of the target.
    :param print_info: Print out debug info about iterations and error

    :return: Rescaled connectivity matrix
    """

    # corresponding signal-to-x ratio:
    sxr_target = snr / (snr + 1)

    radius = max(abs(np.linalg.eig(con_mat)[0]))

    con_mat = con_mat / radius * 0.5
    radius = 0.5

    # iteratively try to find magnitude of con_mat that gives the desired snr
    err = 1.
    for it in range(max_iters):
        sxr = max(steady_sxr(con_mat, noise_cov))
        err = abs(sxr - sxr_target)
        if err < stop_error:
            if print_info:
                print('Matrix rescaling info:')
                print('Iterations:', it)
                print('SXR error:', err)
            break

        # sxr approximately scales as radius squared
        new_radius = min(0.999, radius * np.sqrt(sxr_target / sxr))
        con_mat = con_mat * new_radius / radius
        radius = new_radius

    else:
        if print_info:
            print('Matrix rescaling info:')
            print('Iterations:', max_iters, '(max)')
            print('SXR error:', err)

    return con_mat


def sim_network(con_mat: Union[list, np.ndarray],
                n_time: int = 10000,
                iv_prob: Union[float, list, np.ndarray] = 0.025,
                iv_gain: Union[float, list, np.ndarray] = 0.1,
                noise_cov: Union[float, list, np.ndarray] = 1.,
                snr: float = None):
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

    :param con_mat:   MxM array-like of connection strengths. If var_kept is not None, it will be
                      scaled to produce the requested SNR (ratio of non-noise to noise power
                      at each timestep). Otherwise, the largest eigenvalue should be
                      less than 1 in magnitude to avoid instability and unit roots.
    :param n_time:    Number of timepoints (N)
    :param iv_prob:   Probability of IV event at each time for each variable
                      (can be scalar or an array of length M)
    :param iv_gain:   Factor to scale activation when IV event occurs
                      (can be scalar or an array of length M)
    :param noise_cov: Covariance of additive noise - either a scalar, vector, or matrix.
                      Scalar = independent noise with same variance
                      Vector = independent noise with different variances
                      Matrix = noise with dependencies between variables
    :param snr:       Maximum ratio of variance kept from the previous timestep to noise added over all variables.
                      If None, do not scale con_mat.

    :return: tuple of:
        MxN double ndarray of ln(activations) over time
        MxN boolean ndarray of IV events
        MxM connectivity matrix, rescaled if var_kept was not None.
    """

    # Argument checks
    con_mat = np.array(con_mat)
    assert con_mat.ndim == 2, 'Connectivity matrix must be a matrix'
    n_vars, n_cols = con_mat.shape
    assert n_vars >= 2, 'Must have at least 2 regions (2x2 connectivity)'
    assert n_vars == n_cols, 'Connectivity matrix must be square'
    assert np.isreal(con_mat).all(), 'Connectivity matrix must be real'

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

    assert np.all(np.isreal(noise_cov)), 'Noise covariance must be real'
    if np.isscalar(noise_cov):
        noise_cov = noise_cov * np.eye(n_vars)
    else:
        noise_cov = np.array(noise_cov)
        if np.ndim(noise_cov) == 1:
            noise_cov = np.diag(noise_cov)
    assert noise_cov.shape == (n_vars, n_vars), "Invalid shape for noise covariance"

    radius = max(abs(np.linalg.eig(con_mat)[0]))

    if snr is None:
        assert radius < 1.0,\
            'System is nonstationary - make sure all eigenvalues are < 1 in absolute value'
    else:
        assert np.isscalar(snr) and np.isreal(snr), 'SNR must be a real scalar'
        assert snr >= 0, 'SNR cannot be negative'

        con_mat = rescale_to_snr(con_mat, noise_cov, snr)

    # Generate IVs with given probability at each timepoint
    ivs = np.random.rand(n_vars, n_time) < iv_prob

    # Initialize log of network activations as Gaussian
    log_act = np.zeros((n_vars, n_time))
    log_act[:, 0] = np.random.randn(n_vars) + ivs[:, 0] * np.log(iv_gain)

    # generate additive noise
    noise = np.random.multivariate_normal(np.zeros(n_vars), noise_cov, size=n_time).T

    # run the network
    for kT in range(1, n_time):
        log_act[:, kT] = con_mat @ log_act[:, kT - 1] + noise[:, kT] + ivs[:, kT] * np.log(iv_gain)

    return log_act, ivs, con_mat


def gen_con_mat(n_vars: int, max_eig: float = 0.9):
    """
    Generate a random stable connectivity matrix. Coefficients are chosen uniformly at
    random with a mean of 0 and then scaled to have the requested maximum eigenvalue.

    :param n_vars:  Number of variables N
    :param max_eig: Absolute value of max eigenvalue
    :return: N x N connectivity matrix
    """

    assert n_vars >= 1, 'Number of variables must be at least 1'
    assert 0 < max_eig <= 1, 'Max eigenvalue should be between 0 and 1'

    mat = np.random.rand(n_vars, n_vars) - 0.5
    curr_max_eig = max(abs(np.linalg.eig(mat)[0]))

    return mat * max_eig / curr_max_eig
