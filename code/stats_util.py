import numpy as np


def calc_dist_matrix(multidim_data, norm_ord=2):
    """
    Find the pairwise distances between multidimensional points.
    Input multidim_data should be N x D (N observations, D dimensions)
    Distance is defined by an L2 norm by default, or specify norm_ord parameter as in np.norm
    """

    # rotate so variables are in 3rd dimension
    data1 = np.transpose(multidim_data[:, :, np.newaxis], (0, 2, 1))
    data2 = np.transpose(multidim_data[:, :, np.newaxis], (2, 0, 1))

    diff = data1 - data2  # now N x N x D
    return np.linalg.norm(diff, ord=norm_ord, axis=2)


def get_medoid_ind(multidim_data, norm_ord=2):
    """
    Return the index of the mediod of multidim_data, which is the point with minimal total
    distance to all other points, as defined by the norm of choice.
    """

    dist = calc_dist_matrix(multidim_data, norm_ord=norm_ord)
    return np.argmin(np.sum(dist, axis=0))


if __name__ == '__main__':
    # test functions
    np.set_printoptions(precision=3)

    partial_cube_dist = calc_dist_matrix(np.array([
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 1., 1.],
        [1., 1., 1.]
    ]))

    expected_dist = np.array(
        [[0, 1, np.sqrt(2), np.sqrt(3)],
         [1, 0, 1, np.sqrt(2)],
         [np.sqrt(2), 1, 0, 1],
         [np.sqrt(3), np.sqrt(2), 1, 0]])

    assert np.all(partial_cube_dist == expected_dist),\
        f'Incorrect result from calc_dist_matrix. Error:\n{partial_cube_dist - expected_dist}'

    octo = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [0., 0., 0.],
    ])

    octo_medoid = get_medoid_ind(octo)
    expected_medoid = 6

    assert octo_medoid == expected_medoid,\
        f'Incorrect result from get_medoid_ind. Got {octo_medoid}, expected {expected_medoid}.'

    print('All tests passed!')
