# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:23:26 2023

@author: DELINTE Nicolas
"""

import numpy as np
from unravel.core import get_microstructure_map, get_weighted_mean


def get_tract_metric_along_trajectory(fixel_weights, metric_maps: list,
                                      roi_sections):
    '''


    Parameters
    ----------
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    metric_maps : list
        List of K 3-D arrays of shape (x,y,z) containing metric estimations.
    roi_sections : 3D array of size (x,y,z)
        Labeled array containing the volumes of the section of the tract.

    Returns
    -------
    m_array : 1D array of size (n)
        DESCRIPTION.
    std_array : 1D array of size (n)
        DESCRIPTION.

    '''

    m_array = np.zeros(np.max(roi_sections))
    std_array = np.zeros(np.max(roi_sections))

    if len(metric_maps) == 1:
        fixel_weights = fixel_weights[..., np.newaxis]

    micro_map = get_microstructure_map(fixel_weights, metric_maps)

    for i in range(np.max(roi_sections)):

        if i == 0:
            continue

        roi = np.where(roi_sections == i, 1, 0)
        fixel_weights_roi = fixel_weights * np.repeat(roi[..., np.newaxis],
                                                      1, axis=-1)

        mean, std = get_weighted_mean(micro_map, fixel_weights_roi)
        if mean == 0:
            mean = m_array[i-1]
            std = std_array[i-1]
        m_array[i], std_array[i] = mean, std

    return m_array, std_array
