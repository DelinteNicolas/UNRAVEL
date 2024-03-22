# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:17:13 2022

@author: DELINTE Nicolas
"""

import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram


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


def peaks_to_RGB(peaks, frac=None, fvf=None, order: str = 'rgb'):
    '''
    Returns a RGB map of shape (x,y,z,3) representing the main direction of
    of the peaks. Optionaly scaled by fraction and/or fiber volume fraction.

    Parameters
    ----------
    peaks : 5D array of shape (x,y,z,3,k)
        List of arrays containing the peaks of shape (x,y,z,3)
    frac : 4D array of shape (x,y,z,k), optional
        List of arrays of shape (x,y,z) containing the fraction of each fixel.
        The default is None.
    fvf : 4D array of shape (x,y,z,k), optional
        List of arrays of shape (x,y,z) containing the fiber volume fraction of
        each fixel. The default is None.
    order : str, optional
        Order of colors, either 'rgb', 'gbr' or 'brg'. The default is 'rgb'.

    Returns
    -------
    rgb : 4-D array of shape (x,y,z,3)
        RGB map of shape (x,y,z,3) representing the main direction of
        of the peaks. With type float64 [0,1].

    '''

    peaks = np.nan_to_num(peaks)
    if len(peaks.shape) <= 4:
        peaks = peaks[..., np.newaxis]

    K = peaks.shape[-1]

    if frac is None:
        frac = np.ones(peaks.shape[:3]+(K,))/K
    elif len(frac.shape) <= 3:
        frac = frac[..., np.newaxis]
    frac = np.stack((frac,)*3, axis=3)

    if fvf is None:
        fvf = np.ones(peaks.shape[:3]+(K,))
    elif len(fvf.shape) <= 3:
        fvf = fvf[..., np.newaxis]
    fvf = np.stack((fvf,)*3, axis=3)

    rgb = np.sum(abs(peaks)*frac*fvf, axis=-1)

    # Normalize
    if frac is None and fvf is None:
        rgb = normalize_color(rgb, norm_all_voxels=True)

    # Color order
    if order == 'brg':
        rgb = rgb[(slice(None),) * 3 + ([2, 0, 1],)]
    elif order == 'gbr':
        rgb = rgb[(slice(None),) * 3 + ([1, 2, 0],)]

    return rgb


def peaks_to_peak(peaks, fixel_weights, frac=None, fvf=None):
    '''
    Fuse peaks into a single peak based on fixel weight and fvf, intensity
    is then weighted with frac. Mostly used for visualization purposes.

    Parameters
    ----------
    peaks : 4-D array of shape (x,y,z,3,k)
        List of 4-D arrays of shape (x,y,z,3) containing peak information.
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    peak : 3-D array of shape (x,y,z,3)

    '''

    K = peaks.shape[4]

    peaks = np.nan_to_num(peaks)

    if frac is None:
        frac = np.ones(peaks.shape[:-2]+(K,))/K
    frac = np.stack((frac,)*3, axis=3)

    if fvf is None:
        fvf = np.ones(peaks.shape[:-2]+(K,))
    fvf = np.stack((fvf,)*3, axis=3)

    fixel_weights = np.stack((fixel_weights,)*3, axis=3)
    weight_sum = np.sum(fixel_weights, axis=4)
    weight_sum = np.stack((weight_sum,)*K, axis=4)

    peak = np.abs(peaks) * fixel_weights * fvf*frac
    peak = np.divide(peak, weight_sum, where=weight_sum != 0)
    peak = np.sum(peak, axis=-1)

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

    from unravel.core import (tract_to_streamlines, compute_subsegments,
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
                           color: bool = False, subsegment: int = 10,
                           norm_all_voxels: bool = True):
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
    subsegment : int, optional
        Divides the streamline segment into n subsegments. Increases spatial
        resolution of streamline segments and computation time. The default
        is 10.
    norm_all_voxels : bool, optional
        If True, sets all color voxel with a maximum intensity. Else, the
        intensity is weighted by the number of fibers. The default is True.

    Returns
    -------
    density : 3-D array of shape (x,y,z)
        Array containing the streamline density in each voxel.
    '''

    streams = trk.streamlines
    point = streams.get_data().astype(np.float32)*resolution_increase

    # Creating subpoints
    subpoint = np.linspace(point, np.roll(point, -1, axis=0),
                           subsegment+1, axis=1)
    point = subpoint[:, :-1, :].reshape(point.shape[0]*subsegment, 3)
    del subpoint

    # Getting fixel vectors
    x, y, z = point.astype(np.int32).T

    # Removing streamline end points
    ends = (streams._offsets+streams._lengths-1)*subsegment
    idx = np.linspace(0, subsegment-1, subsegment, dtype=np.int32)
    ends = ends[:, np.newaxis] + idx
    del idx
    ends = ends.flatten()

    if color:
        # Computing streamline segment vectors
        next_point = np.roll(point, -1, axis=0)
        vs = next_point-point
        del point, next_point

        vs[ends, :] = [0, 0, 0]
        del ends

        rgb = np.zeros(tuple(trk._dimensions*resolution_increase)+(3,),
                       dtype=np.float32)
        np.add.at(rgb, (x, y, z), abs(vs))

        return normalize_color(rgb, norm_all_voxels=norm_all_voxels)

    else:
        coef = np.ones(x.shape, dtype=np.float32)

        coef[ends] = 0

        density = np.zeros(trk._dimensions*resolution_increase,
                           dtype=np.float32)
        np.add.at(density, (x, y, z), coef)

        return density


def normalize_color(rgb, norm_all_voxels: bool = False):
    '''
    Sets values in RGB array (x,y,z,3) to be within [0,1].

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
        norm = np.linalg.norm(rgb, axis=3)
        norm = np.stack((norm,)*3, axis=3, dtype=np.float32)
        norm = np.divide(rgb, norm, dtype=np.float64,
                         where=np.sum(rgb, axis=3, keepdims=True) != 0)
    else:
        norm = (rgb/np.max(rgb)).astype(np.float32)

    return norm


def plot_streamline_trajectory(trk, resolution_increase: int = 1,
                               streamline_number: int = 0, axis: int = 1,
                               color: bool = False, subsegment: int = 10,
                               norm_all_voxels: bool = False):
    '''
    Produces a graph of the streamline density of tract 'trk', the streamline
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
    from unravel.core import tract_to_streamlines

    density = get_streamline_density(trk, color=color, subsegment=subsegment,
                                     resolution_increase=resolution_increase)

    if color:
        # density = normalize_color(density, norm_all_voxels=norm_all_voxels)
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


def xyz_to_spherical(xyz):
    '''
    X,y,z coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : array of size (n,3)
        X,y,z coordinates of n points

    Returns
    -------
    r : array of size (n)
        DESCRIPTION.
    theta : array of size (n)
        DESCRIPTION.
    phi : array of size (n)
        DESCRIPTION.

    '''
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return r, theta, phi


def spherical_to_xyz(theta, phi):

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z


def fuse_trk(trk_file_1: str, trk_file_2: str, output_file: str):
    '''
    Creates a new .trk file with all streamlines contained in the two input .trk
    files. The input files must be in the same space.

    Parameters
    ----------
    trk_file_1 : str
        Filename of first input .trk file.
    trk_file_2 : str
        Filename of second input .trk file.
    output_file : str
        Filename of output .trk file.

    Returns
    -------
    None.

    '''

    trk = load_tractogram(trk_file_1, 'same')
    trk.to_vox()
    trk.to_corner()
    streams_1 = trk.streamlines

    trk = load_tractogram(trk_file_2, 'same')
    trk.to_vox()
    trk.to_corner()
    streams_2 = trk.streamlines

    streams_1.extend(streams_2)

    trk_new = trk.from_sft(streams_1, trk)
    save_tractogram(trk_new, output_file)
