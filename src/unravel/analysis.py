# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:23:26 2023

@author: DELINTE Nicolas
"""

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
        weights=np.ones(len(streamlines))

    for i,sl in enumerate(tqdm(streamlines, desc='Computing connectivity matrix')):

        x, y, z = np.floor(sl.T).astype(int)
        crossed_labels = np.unique(label_volume[x, y, z])

        for comb in combinations(crossed_labels, 2):
            matrix[comb] += weights[i]

    matrix = matrix+matrix.T

    return matrix
