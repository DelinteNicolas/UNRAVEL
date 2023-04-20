# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:17:13 2022

@author: DELINTE Nicolas
"""

import numpy as np
from dipy.io.streamline import load_tractogram


def tract_to_ROI(trk_file: str):
    '''
    Returns a binary mask of each voxel containing a tractography node.The
    voxels containing streamlines segments but no nodes will not be selected.

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
        of the peaks. With type float64 [0,1].

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
    dim = len(peaksList[0].shape[:-1])

    len_ratio = np.ones(peaksList[0].shape[:-1])

    for k in range(K):
        peaksList[k] = np.nan_to_num(peaksList[k])
        len_ratio += np.where(np.sum(peaksList[k], axis=dim) == 0, 1, 0)

    if fracList is None:
        fracList = []
        for k in range(K):
            fracList.append(np.ones(peaksList[0].shape[:-1]))

    if fvfList is None:
        fvfList = []
        for k in range(K):
            fvfList.append(np.ones(peaksList[0].shape[:-1]))

    rgb = np.zeros(peaksList[0].shape)

    for xyz in np.ndindex(peaksList[0].shape[:-1]):
        for k in range(K):
            rgb[xyz] += abs(peaksList[k][xyz])*fracList[k][xyz]*fvfList[k][xyz]

    # Normalize between [0,1] and by number of peaks per voxel
    rgb *= np.repeat(1+len_ratio[(slice(None),) *
                     dim + (np.newaxis,)]/K, 3, axis=dim)
    rgb /= np.max(rgb)

    return rgb


def peaks_to_peak(peaksList: list, fixel_weights, fracList: list = None,
                  fvfList: list = None):
    '''
    Fuse peaks into a single peak based on fixel weight and fvf, intensity
    is then weighted with frac. Mostly used for visualization purposes.

    Parameters
    ----------
    peaksList : list of 4-D arrays
        List of arrays containing the peaks of shape (x,y,z,3)
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    peak : 3-D array of shape (x,y,z,3)

    '''

    K = len(peaksList)

    for k in range(K):
        peaksList[k] = np.nan_to_num(peaksList[k])

    peak = np.zeros(peaksList[0].shape)

    if fracList is None:
        fracList = []
        for k in range(K):
            fracList.append(np.ones(peaksList[0].shape[:-1])/(k+1))

    if fvfList is None:
        fvfList = []
        for k in range(K):
            fvfList.append(np.ones(peaksList[0].shape[:-1]))

    fracTot = np.zeros(peaksList[0].shape[:-1])

    for xyz in np.ndindex(peaksList[0].shape[:-1]):
        for k in range(K):
            peak[xyz] += (abs(peaksList[k][xyz])*fixel_weights[xyz+(k,)]
                          / np.sum(fixel_weights[xyz])*fvfList[k][xyz])

    for k in range(K):
        fracTot += fracList[k]

    peak[..., 0] *= fracTot
    peak[..., 1] *= fracTot
    peak[..., 2] *= fracTot

    # peak = np.where(np.isnan(peak), None, peak)

    return peak


def tensor_to_DTI(t):
    '''
    Creates fractional anisotropy (FA), axial diffusivity (AD), radial
    diffusivity (RD) and mean diffusivity (MD) maps from a tensor array.

    Parameters
    ----------
    t : 5-D array
        Tensor array of shape (x,y,z,1,6).

    Returns
    -------
    FA : 3-D array
        FA array of shape (x,y,z).
    AD : 3-D array
        AD array of shape (x,y,z).
    RD : 3-D array
        RD array of shape (x,y,z).
    MD : 3-D array
        MD array of shape (x,y,z).

    '''

    np.seterr(divide='ignore', invalid='ignore')

    if len(t.shape) == 4:
        t = t[..., np.newaxis]
        t = np.transpose(t, (1, 2, 3, 4, 0))

        mt = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 2]],
                      [t[:, :, :, 0, 1], t[:, :, :, 0, 3], t[:, :, :, 0, 4]],
                      [t[:, :, :, 0, 2], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]])

    else:

        mt = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 3]],
                      [t[:, :, :, 0, 1], t[:, :, :, 0, 2], t[:, :, :, 0, 4]],
                      [t[:, :, :, 0, 3], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]])

    mt = np.transpose(mt, (2, 3, 4, 0, 1))
    val, vec = np.linalg.eig(mt)

    # Sorting to make sure that first dimension contains lambda_max
    val = -np.sort(-val, axis=-1)

    FA = np.sqrt((np.power((val[:, :, :, 0]-val[:, :, :, 1]), 2) +
                  np.power((val[:, :, :, 1]-val[:, :, :, 2]), 2) +
                  np.power((val[:, :, :, 2]-val[:, :, :, 0]), 2))
                 / (2*(np.power(val[:, :, :, 0], 2) +
                       np.power(val[:, :, :, 1], 2) +
                       np.power(val[:, :, :, 2], 2))))

    MD = (val[:, :, :, 0]+val[:, :, :, 1]+val[:, :, :, 2])/3

    AD = val[:, :, :, 0]

    RD = (val[:, :, :, 1]+val[:, :, :, 2])/2

    for m in [FA, MD, RD, AD]:
        m[np.isnan(m)] = 0

    FA = FA.real.astype('float64')
    AD = AD.real.astype('float64')
    RD = RD.real.astype('float64')
    MD = MD.real.astype('float64')

    np.seterr(divide='warn', invalid='warn')

    return FA, AD, RD, MD


def get_streamline_count(trk) -> int:
    '''
    Returns the number of streamlines in a tractogram.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file

    Returns
    -------
    count : int
        Number of streamlines in tractogram.

    '''

    # Safety net
    if type(trk) == str:
        trk = load_tractogram(trk, 'same')

    count = len(trk.streamlines._offsets)

    return count


def get_streamline_angle(trk, resolution_increase: int = 1):
    '''
    Get the fixel weights from a tract specified in trk_file.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    resolution_increase : int, optional
        Factor multiplying the resolution/dimensions of output array. The
        default is 1.

    Returns
    -------
    density : 3-D array of shape (x,y,z)
        Array containing the mean angle of streamline segments in each voxel.
    '''

    from TIME.core import (tract_to_streamlines, compute_subsegments,
                           angle_difference)
    from tqdm import tqdm

    num = np.zeros(trk._dimensions*resolution_increase)
    angle = np.zeros(trk._dimensions*resolution_increase)

    sList = tract_to_streamlines(trk)

    for streamline in tqdm(sList):

        previous_point = streamline[1, :]*resolution_increase
        previous_dir = (previous_point-streamline[0, :]*resolution_increase)

        for i in range(2, streamline.shape[0]):

            point = streamline[i, :]*resolution_increase

            voxList = compute_subsegments(previous_point, point)
            vs = (point-previous_point)
            ang = angle_difference(vs, previous_dir)

            for x, y, z in voxList:

                x, y, z = (int(x), int(y), int(z))

                num[x, y, z] += 1
                angle[x, y, z] += ang

            previous_point = point
            previous_dir = vs

    angle[num != 0] /= num[num != 0]

    return angle


def get_streamline_density(trk, resolution_increase: int = 1,
                           color: bool = False):
    '''
    Get the total segment length from a tract specified in trk.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    resolution_increase : int, optional
        Factor multiplying the resolution/dimensions of output array. The
        default is 1.
    color : bool, optional
        If True, output a RGB volume with colors corresponding to the
        directions of the streamlines, modulated by streamline density.
        The default is False.

    Returns
    -------
    density : 3-D array of shape (x,y,z)
        Array containing the streamline density in each voxel.
    '''

    from TIME.core import tract_to_streamlines, compute_subsegments
    from tqdm import tqdm

    density = np.zeros(trk._dimensions*resolution_increase, dtype=np.float32)
    rgb = np.zeros(tuple(trk._dimensions*resolution_increase)+(3,),
                   dtype=np.float32)

    sList = tract_to_streamlines(trk)

    for streamline in tqdm(sList):

        previous_point = streamline[0, :]*resolution_increase

        for i in range(1, streamline.shape[0]):

            point = streamline[i, :]*resolution_increase

            voxList = compute_subsegments(previous_point, point)
            vs = (point-previous_point)

            for x, y, z in voxList:

                x, y, z = (int(x), int(y), int(z))

                density[x, y, z] += voxList[(x, y, z)]
                if color:
                    rgb[x, y, z] += abs(vs)

            previous_point = point

    if color:
        return rgb

    return density


def normalize_color(rgb, norm_all_voxels: bool = False):
    '''
    Sets values in RGB array (x,y,z,3) to be within [0,1].

    TODO: increase speed when norm_all_voxels is set to True.

    Parameters
    ----------
    rgb : 3-D array of shape (x,y,z,3)
        RGB volume.
    norm_all_voxels : bool, optional
        If True, all voxel display maximum intensity. The default is False.

    Returns
    -------
    norm : 3-D array of shape (x,y,z,3)
        RGB volume.

    '''

    if norm_all_voxels:
        norm = np.zeros(rgb.shape)

        for xyz in np.ndindex(rgb.shape[:-1]):
            if np.sum(rgb[xyz]) != 0:
                norm[xyz] = rgb[xyz] / np.linalg.norm(rgb[xyz])
    else:
        norm = rgb/np.max(rgb)

    return norm


def plot_streamline_trajectory(trk, resolution_increase: int = 1,
                               streamline_number: int = 0, axis: int = 1,
                               color: bool = False,
                               norm_all_voxels: bool = False):
    '''
    Produces a grpah of the streamline density of tract 'trk', the streamline
    specified with 'streamline_number' is highlighted along 'axis'.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file.
    resolution_increase : int, optional
        The dimensions of the volume are multiplied by this value to increase
        resolution. The default is 1.
    streamline_number : int, optional
        Number of the streamline to be highlighted. The default is 0.
    axis : int, optional
        Axis of inspection, [0:2] in 3D volumes. The default is 1.
    color : bool, optional
        If True, output a RGB volume with colors corresponding to the
        directions of the streamlines, modulated by streamline density.
        The default is False.
    norm_all_voxels : bool, optional
        If True, all RGB voxels display maximum intensity. Increases computation
        time. The default is False.

    Returns
    -------
    None.

    '''

    import matplotlib.pyplot as plt
    from TIME.core import tract_to_streamlines

    density = get_streamline_density(trk, color=color,
                                     resolution_increase=resolution_increase)

    if color:
        density = normalize_color(density, norm_all_voxels=norm_all_voxels)
        density = (density*255).astype('uint8')

    sList = tract_to_streamlines(trk)

    streamline = sList[streamline_number]

    x = []
    y = []
    z = []

    for i in range(0, streamline.shape[0]):

        point = streamline[i, :]*resolution_increase

        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    transpose = [1, 0]
    c = '#e69402ff'
    if color:
        transpose.append(2)
        c = '#ffffffff'

    plt.figure()
    if axis == 0:
        plt.imshow(np.transpose(density[int(sum(x)/len(x)), :, :], transpose),
                   origin='lower', cmap='gray')
        plt.plot(y, z, '.-', c=c)
    elif axis == 1:
        plt.imshow(np.transpose(density[:, int(sum(y)/len(y)), :], transpose),
                   origin='lower', cmap='gray')
        plt.plot(x, z, '.-', c=c)
    else:
        plt.imshow(np.transpose(density[:, :, int(sum(z)/len(z))], transpose),
                   origin='lower', cmap='gray')
        plt.plot(x, y, '.-', c=c)
    plt.title('Streamline trajectory')
