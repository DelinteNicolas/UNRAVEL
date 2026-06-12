# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:23:26 2023

@author: DELINTE Nicolas
"""

from scipy.sparse.csgraph import shortest_path
import numpy as np
from tqdm import tqdm
from itertools import combinations
from unravel.core import get_microstructure_map, get_weighted_mean


def get_metric_along_trajectory(fixel_weights, metric_maps, roi_sections,
                                weighting: str = 'tsl'):
    '''


    Parameters
    ----------
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    metric_maps : 4D array of shape (x,y,z,k)
        List of K 4D arrays of shape (x,y,z) containing metric estimations.
    roi_sections : 3D array of size (x,y,z)
        Labeled array containing the volumes of the section of the tract.
    weighting : str, optional
        Weighting used for the mean. The default is 'tsl'.

    Returns
    -------
    m_array : 1D array of size (n)
        DESCRIPTION.
    std_array : 1D array of size (n)
        DESCRIPTION.

    '''

    m_array = np.zeros(np.max(roi_sections)+1)
    std_array = np.zeros(np.max(roi_sections)+1)

    if len(metric_maps.shape) <= 3:
        fixel_weights = fixel_weights[..., np.newaxis]

    micro_map = get_microstructure_map(fixel_weights, metric_maps)

    for i in range(np.max(roi_sections)+1):

        if i == 0:
            continue

        roi = np.where(roi_sections == i, 1, 0)
        fixel_weights_roi = fixel_weights * roi[..., np.newaxis]

        mean, std = get_weighted_mean(micro_map, fixel_weights_roi,
                                      weighting=weighting)
        if mean == 0:
            mean = m_array[i-1]
            std = std_array[i-1]
        m_array[i], std_array[i] = mean, std

    return m_array, std_array


def connectivity_matrix(streamlines, label_volume, inclusive: bool = True,
                        weights=None):
    '''
    Returns the symetric connectivity matrix of the streamlines. Usage of
    trk.to_vox(), trk.to_corner() beforehand is highly recommended. This
    fonction is at least x6 times faster than the implementation in Dipy
    when inclusive is set to True.

    Parameters
    ----------
    streamlines : streamline object
        DESCRIPTION.
    label_volume : 3D array of size (x,y,z)
        Array containing the labels as int.
    inclusive: bool, optional
        Whether to analyze the entire streamline, as opposed to just the
        endpoints.
    weights: 1D array of size (n), optional
        Array containing the weights for the n streamlines in the tract. For
        example the weights computed with SIFT2. Default: the weight of each
        streamline is set to 1.

    Returns
    -------
    matrix : 2D array
        Connectivity matrix.

    '''

    label_volume = label_volume.astype(int)
    matrix = np.zeros((np.max(label_volume)+1, np.max(label_volume)+1))

    if not inclusive:
        streamlines = [sl[0::len(sl)-1] for sl in streamlines]

    if weights is None:
        weights = np.ones(len(streamlines))

    for i, sl in enumerate(tqdm(streamlines, desc='Computing connectivity matrix')):

        x, y, z = np.floor(sl.T).astype(int)
        crossed_labels = np.unique(label_volume[x, y, z])

        for comb in combinations(crossed_labels, 2):
            matrix[comb] += weights[i]

    matrix = matrix+matrix.T

    return matrix


def degree(A, weighted=True):
    """
    Compute node degree.

    Parameters
    ----------
    connectivity_matrix : ndarray (N,N)
        Weighted adjacency matrix.
    weighted : bool
        If True, returns node strength (sum of weights).
        If False, returns binary degree.

    Returns
    -------
    deg : ndarray (N,)
    """
    A = np.asarray(A)

    if weighted:
        return np.sum(A, axis=1)

    return np.sum(A > 0, axis=1)


def clustering_coefficient(A, weighted=True):
    """
    Clustering coefficient.

    Parameters
    ----------
    A : (N,N) ndarray
        Connectivity matrix.
    weighted : bool

    Returns
    -------
    C : (N,) ndarray
    """

    A = np.asarray(A, dtype=float)

    if not weighted:
        B = (A > 0).astype(float)

        k = np.sum(B, axis=1)

        triangles = np.diag(B @ B @ B) / 2

        C = np.zeros(len(k))

        mask = k > 1
        C[mask] = (
            2 * triangles[mask]
            / (k[mask] * (k[mask] - 1))
        )

        return C

    # Onnela et al. weighted clustering coefficient

    if A.max() > 0:
        W = A / A.max()
    else:
        W = A.copy()

    K = np.sum(A > 0, axis=1)

    C = np.zeros(len(A))

    W13 = np.power(W, 1 / 3)

    cyc3 = np.diag(W13 @ W13 @ W13)

    mask = K > 1

    C[mask] = cyc3[mask] / (K[mask] * (K[mask] - 1))

    return C


def global_efficiency(A, weighted=True):
    """
    Global efficiency.

    Parameters
    ----------
    A : (N,N) ndarray
        Connectivity matrix.
    weighted : bool

    Returns
    -------
    Eglob : float
    """

    A = np.asarray(A, dtype=float)

    if weighted:

        Dmat = np.zeros_like(A)

        mask = A > 0
        Dmat[mask] = 1.0 / A[mask]

        D = shortest_path(
            Dmat,
            directed=False,
            unweighted=False
        )

    else:

        D = shortest_path(
            (A > 0).astype(float),
            directed=False,
            unweighted=True
        )

    with np.errstate(divide='ignore'):
        invD = 1.0 / D

    np.fill_diagonal(invD, 0)

    N = len(A)

    return invD.sum() / (N * (N - 1))


def local_efficiency(A, weighted=True):
    """
    Local efficiency for each node.

    Parameters
    ----------
    A : (N,N) ndarray
        Connectivity matrix.
    weighted : bool

    Returns
    -------
    Eloc : (N,) ndarray
    """

    A = np.asarray(A, dtype=float)

    N = len(A)

    Eloc = np.zeros(N)

    for i in range(N):

        neighbors = np.where(A[i] > 0)[0]

        m = len(neighbors)

        if m < 2:
            continue

        subA = A[np.ix_(neighbors, neighbors)]

        if weighted:

            Dmat = np.zeros_like(subA)

            mask = subA > 0
            Dmat[mask] = 1.0 / subA[mask]

            D = shortest_path(
                Dmat,
                directed=False,
                unweighted=False
            )

        else:

            D = shortest_path(
                (subA > 0).astype(float),
                directed=False,
                unweighted=True
            )

        with np.errstate(divide='ignore'):
            invD = 1.0 / D

        np.fill_diagonal(invD, 0)

        Eloc[i] = invD.sum() / (m * (m - 1))

    return Eloc
