# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:17:13 2022

@author: DELINTE Nicolas
"""

import numpy as np
from dipy.io.streamline import load_tractogram


def tract_to_ROI(trk_file: str):
    '''


    Parameters
    ----------
    trk_file : str
        Path to tractography file (.trk)

    Returns
    -------
    ROI : 3-D array of shape (x,y,z)
        Binary array containing the ROI associated to the tract in trk_file.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    streams_data = trk.streamlines.get_data()

    b = np.float64(streams_data)
    ROI = np.zeros(trk._dimensions)

    for i in range(b.shape[0]):

        ROI[(int(b[i, 0]), int(b[i, 1]), int(b[i, 2]))] = 1

    return ROI


def peaks_to_RGB(peaksList: list, fracList: list = None, fvfList: list = None):
    '''
    Returns a RGB map of shape (x,y,z,3) representing the main direction of
    of the peaks. Optionnaly scaled by fraction and/or fiber volume fraction.

    Parameters
    ----------
    peaksList : list of 4-D arrays
        List of arrays containing the peaks of shape (x,y,z,3)
    fracList : list of 3-D arrays, optional
        List of arrays of shape (x,y,z) containing the fraction of each fixel.
        The default is None.
    fvfList : list of 3-D arrays, optional
        List of arrays of shape (x,y,z) containing the fiber volume fraction of
        each fixel. The default is None.

    Returns
    -------
    rgb : 4-D array of shape (x,y,z,3)
        RGB map of shape (x,y,z,3) representing the main direction of
        of the peaks.

    '''

    # Safety net
    if type(peaksList) != list:
        if len(peaksList.shape) == 5:
            peaksArray = peaksList.copy()
            peaksList = []
            for k in range(peaksArray.shape[4]):
                peaksList.append(peaksArray[:, :, :, :, k])
        else:
            peaksList = [peaksList]

    K = len(peaksList)

    if fracList is None:
        fracList = []
        for k in range(K):
            fracList.append(np.ones(peaksList[0].shape[:3]))

    if fvfList is None:
        fvfList = []
        for k in range(K):
            fvfList.append(np.ones(peaksList[0].shape[:3]))

    rgb = np.zeros(peaksList[0].shape)

    for xyz in np.ndindex(peaksList[0].shape[:3]):
        for k in range(K):
            rgb[xyz] += abs(peaksList[k][xyz])*fracList[k][xyz]*fvfList[k][xyz]

    return rgb
