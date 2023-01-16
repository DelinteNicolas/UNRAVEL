# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:05:25 2022

@author: DELINTE  Nicolas

"""

import os
import warnings
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram


def deltas_to_D(dx: float, dy: float, dz: float, lamb=np.diag([1, 0, 0]),
                vec_len: float = 500):
    '''
    Function creating a diffusion tensor from three orthogonal components.

    Parameters
    ----------
    dx : float
        'x' component.
    dy : float
        'y' component.
    dz : float
        'z' component.
    lamb : 3x3 array, optional
        Diagonal matrix containing the diffusion eigenvalues. The default is
        np.diag([1, 0, 0]).
    vec_len : float, optional
        Value decreasing the diffusion. The default is 500.

    Raises
    ------
    np.linalg.LinAlgError
        DESCRIPTION.

    Returns
    -------
    D : 3x3 array
        Matrix containing the diffusion tensor.

    '''

    e = np.array([[dx, -dz-dy, dy*dx-dx*dz],
                  [dy, dx, -dx**2-(dz+dy)*dz],
                  [dz, dx, dx**2+(dy+dz)*dy]])

    try:
        e_1 = np.linalg.inv(e)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError

    D = (e.dot(lamb)).dot(e_1)/vec_len

    return D


def voxel_distance(position1: tuple, position2: tuple) -> tuple:
    '''
    Returns the distance between two voxels in multiple dimensions.

    Parameters
    ----------
    position1 : tuple
        First position. Ex: (1,1,1).
    position2 : tuple
        Second position. Ex: (1,2,3).

    Returns
    -------
    dis : tuple
        tuple containing the distance between the two positions in every
        direction. Ex: (0,1,2).

    '''

    dis = abs(np.floor(position1)-np.floor(position2))

    return dis


def voxels_from_segment(position1: tuple, position2: tuple,
                        subparts: int = 10) -> dict:
    '''
    Computes the voxels containing a segment (defined by the position 1 and 2)
    and the segment length that is contained within them.

    Parameters
    ----------
    position1 : tuple
        Position. Ex: (1,1,1).
    position2 : tuple
        Position. Ex: (1,2,3).
    subparts : int, optional
        Divide segment into multiple subsegments. Higher value is more precise
        but increases computation time. The default is 10.

    Returns
    -------
    voxList : dict
        Dictionary of the voxels containing a part of the segment .

    '''

    voxDis = sum(voxel_distance(position1, position2))

    voxList = {}

    if voxDis == 0:  # Trying to gain time
        voxList[tuple(np.floor(position1))] = 1

        return voxList

    subseg = np.linspace(position1, position2, subparts)

    for i in subseg:
        xyz = tuple(np.floor(i))
        if xyz not in voxList:
            voxList[xyz] = 1/subparts
        else:
            voxList[xyz] += (1/subparts)

    return voxList


def compute_subsegments(start, finish, vox_size=[1, 1, 1], offset=[0, 0, 0],
                        return_nodes: bool = False):
    '''
    Computes the voxels containing a segment (defined by the start and finish)
    and the segment length that is contained within them.

    Parameters
    ----------
    start : 1-D array of shape (d,)
        Starting point of segment.
    finish : 1-D array of shape (d,)
        End point of segment.
    vox_size : 1-D array of shape (d,)
        vox_size[i] is the (strictly positive) voxel size along the i-th axis,
        with i=1,...,d.
    offset : 1-D array of shape (d,)
        offset[i] is the start of the first voxel along the i-th axis, with
        i=1,...,d.
        In 3 dimensions (d=3), voxel (0, 0, 0) would be defined as
        [off_x, off_x + s_x[ x [off_y, off_y + s_y[ x [off_z, off_z + s_z[,
        where off_x = offset[0], s_x = vox_size[0], off_y = offset[1], etc.
    return_nodes : boolean, optional
        If True, the nodes making the sub-segment(s) are returned.
        The default is False.

    Returns
    -------
    tuple
        subseg_lengths: list of N-1 scalar(s) of type float, with N>=2.

        visited_voxels: list of N-1 1-D array(s) of shape (d,) and type int,
            with N>=2.

        nodes: only if return_nodes=True. List of N arrays of shape (d,) and
            type float, with N>=2.

    '''

    start = np.asarray(start)
    finish = np.asarray(finish)

    tol_lam = 1e-10

    step = finish - start
    Lres = np.sqrt(np.sum(step**2))  # residual step length
    cur = start  # current node

    # Output quantities
    subseg_lengths = []
    visited_voxels = []
    nodes = [start]

    while Lres > 0:
        vox_ID_cur = np.floor((cur-offset)/vox_size).astype(int)
        vox_start = offset + vox_ID_cur * vox_size

        # Case cur at voxel start but step is negative. Decrement voxel ID,
        # consider curr at the end of previous voxel
        decr_vox_ID = (cur == vox_start) & (step < 0)
        if np.any(decr_vox_ID):
            vox_ID_cur[decr_vox_ID] = vox_ID_cur[decr_vox_ID] - 1
            vox_start = offset + vox_ID_cur * vox_size

        # We now work in a centered voxel, i.e. a domain defined by
        # [0, vox_size[0]] x [0, vox_size[1]] x [0, vox_size[2]]
        # We search for smallest lambda in ]0, 1] such that one component i of
        # (cur-vox_start) + lambda * step
        # hits 0 or vox_size[i]
        step_nnz = step.copy()
        step_nnz[step == 0] = 1  # avoid division by 0
        # Lambda to reach start of voxel at 0
        lam_0 = (0.0 - (cur-vox_start))/step_nnz
        lam_0[step == 0] = 1
        # Lambda to reach end of voxel at vox_size[i] for all i
        lam_vs = (vox_size - (cur-vox_start))/step_nnz
        # note: if cur artificially considered at end of previous voxel (see
        # above), ignore small displacement to outer edge
        lam_vs[(step == 0) | decr_vox_ID] = 1
        # Find minimum lambda in ]0, 1]. If all > 1, it means the current step
        # can be completely performed => lambda = 1
        lam_cand = np.concatenate((lam_0, lam_vs))
        lam_cand[(lam_cand <= 0) | (lam_cand > 1)] = 1
        lam = np.min(lam_cand)

        old = cur
        # New location, either at finish or at voxel boundary
        if np.abs(lam - 1) < tol_lam:
            # avoid numerical round-off errors at last sub-step. Typically
            # happens when a step is along a voxel boundary.
            lam = 1
            cur = finish
        else:
            cur = cur + lam * step

        # Use mid-point between new and old node to get voxel in which
        # sub-segment lies:
        midpoint = (old + cur)/2
        vox_ID = np.floor((midpoint-offset)/vox_size).astype(int)

        if lam > tol_lam:
            # Update history (ignore if substep is too small and same voxel
            # visited twice, might happen at corners)
            subseg_lengths.append(lam * Lres)
            visited_voxels.append(vox_ID)
            nodes.append(cur)

        # Residual step
        step = (1-lam) * step  # equivalent to finish - cur
        Lres = (1-lam)*Lres  # or np.sqrt(np.sum(step**2))

    out = [subseg_lengths, visited_voxels]
    if return_nodes:
        out.append(nodes)

    voxList = {}
    for i, sub in enumerate(subseg_lengths):
        voxList[tuple(visited_voxels[i])] = sub

    return voxList


def angle_difference(v1, v2, direction: bool = False) -> float:
    '''
    Computes the angle difference between two vectors.

    Parameters
    ----------
    v1 : 1-D array
        Vector. Ex: [1,1,1]
    v2 : 1-D array
        Vector. Ex: [1,1,1]
    direction : bool, optional
        If False, the vectors are considered to be direction-agnostic -> maximum angle difference = 90.
        If True, the direction of the vectors is taken into account -> maximum angle difference = 180.
        The default is False.

    Returns
    -------
    ang : float
        Angle difference (in degrees).

    '''

    v1n = v1/np.linalg.norm(v1)
    v2n = v2/np.linalg.norm(v2)

    if (v1n == v2n).all():
        return 0

    if sum(v1n*v2n) > 1:
        return 0

    if sum(v1n*v2n) < -1:
        return 180

    ang = np.arccos(sum(v1n*v2n))*180/np.pi

    if ang > 90 and not direction:
        ang = 180-ang

    return ang


def angular_weighting(vs, vList: list, nList: list):
    '''
    Computes the relative contributions of the segments in vList to vs using
    angular weighting.

    Parameters
    ----------
    vs : 1-D array
        Segment vector
    vList : list
        List of the k vectors corresponding to each fiber population
    nList : list
        List of the null k vectors

    Returns
    -------
    ang_coef : list
        List of the k coefficients

    '''

    if len(vList) == 1:
        return [1]

    if len(vList)-sum(nList) <= 1:
        return [1-i for i in list(map(int, nList))]

    angle_diffList = []

    for i, v in enumerate(vList):
        if nList[i]:
            angle_diffList.append(0)
        else:
            angle_diffList.append(angle_difference(vs, v))

    sum_diff = np.sum(angle_diffList)

    ang_coef = []

    for i, angle_diff in enumerate(angle_diffList):
        if nList[i]:
            ang_coef.append(0)
        else:
            coef = 1-angle_diff/sum_diff
            coef = coef/(len(vList)-1-sum(nList))
            ang_coef.append(coef)

    return ang_coef


def closest_fixel_only(vs, vList: list, nList: list):
    '''
    Computes the relative contributions of the segments in vList to vs using
    closest fixel only approach.

    Parameters
    ----------
    vs : 1-D array
        Segment vector
    vList : list
        List of the k vectors corresponding to each fiber population
    nList : list
        List of the null k vectors

    Returns
    -------
    ang_coef : list
        List of the k coefficients

    '''

    if len(vList) == 1:
        return [1]

    if len(vList)-sum(nList) <= 1:
        return [1-i for i in list(map(int, nList))]

    angle_diffList = []

    for k, v in enumerate(vList):
        if nList[k]:
            angle_diffList.append(1000)
        else:
            angle_diffList.append(angle_difference(vs, v))

    ang_coef = []

    min_k = np.argmin(angle_diffList)

    for k, angle_diff in enumerate(angle_diffList):
        if nList[k]:
            ang_coef.append(0)
        else:
            if k == min_k:
                ang_coef.append(1)
            else:
                ang_coef.append(0)

    return ang_coef


def fraction_weighting(point: tuple, vList: list, nList: list, fList: list):
    '''


    Parameters
    ----------
    point : tuple
        x, y, z coordinates
    vList : list
        List of the k vectors corresponding to each fiber population
    nList : list
        List of the null k vectors

    Returns
    -------
    ang_coef : list
        List of the k coefficients

    '''

    if len(vList) == 1:
        return [1]

    if len(vList)-sum(nList) <= 1:
        return [1-i for i in list(map(int, nList))]

    ang_coef = []

    for k, v in enumerate(vList):
        if nList[k]:
            ang_coef.append(0)
        else:
            ang_coef.append(fList[k][point])

    return ang_coef


def t6ToMFpeak(t):
    '''
    (6,17,1,15) with info on 0,2,5 to (17,1,15,3)
    '''

    new_t = np.transpose(t, (1, 2, 3, 0))
    new_t[:, :, :, 0] = new_t[:, :, :, 0]
    new_t[:, :, :, 1] = new_t[:, :, :, 2]
    new_t[:, :, :, 2] = new_t[:, :, :, 5]
    new_t = new_t[:, :, :, :3]

    # t1 = nib.load(
    #     'C:/users/nicol/Documents/Doctorat/Data/Phantom/Diamond/LUCFRD_diamond_t0.nii.gz')

    # out = nib.Nifti1Image(new_t, np.eye(4))  # ,t1.header)
    # out.to_filename('C:/users/nicol/Desktop/LUCFRD_peak_f.nii.gz')

    return new_t


def peak_to_tensor(peaks, norm=None, pixdim=[2, 2, 2]):
    '''
    Takes peaks, such as the ones obtained with Microstructure Fingerprinting,
    and return the corresponding tensor, in the format used in DIAMOND.

    Parameters
    ----------
    peaks : 4-D array
        Array containing the peaks of shape (x,y,z,3)

    Returns
    -------
    t : 5-D array
        Tensor array of shape (x,y,z,1,6).

    '''

    t = np.zeros(peaks.shape[:3]+(1, 6))

    scaleFactor = 1000 / min(pixdim)

    for xyz in np.ndindex(peaks.shape[:3]):

        if peaks[xyz].all() == 0:
            continue

        dx, dy, dz = peaks[xyz]

        try:
            if norm is None:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor)
            else:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor*norm[xyz])
        except np.linalg.LinAlgError:
            continue

        t[xyz+(0, 0)] = D[0, 0]
        t[xyz+(0, 1)] = D[0, 1]
        t[xyz+(0, 2)] = D[1, 1]
        t[xyz+(0, 3)] = D[0, 2]
        t[xyz+(0, 4)] = D[1, 2]
        t[xyz+(0, 5)] = D[2, 2]

    return t


def tensor_to_peak(t):
    '''
    Takes peaks, such as the ones obtained with DIAMOND, and return the
    corresponding tensor, in the format used in Microstructure Fingerprinting.
    TODO : Speed up.

    Parameters
    ----------
    t : 5-D array
        Tensor array of shape (x,y,z,1,6).

    Returns
    -------
    peaks : 4-D array
        Array containing the peaks of shape (x,y,z,3)

    '''

    if len(t.shape) == 4:
        t = t[..., np.newaxis]
        t = np.transpose(t, (1, 2, 3, 4, 0))

        D_t = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 2]],
                        [t[:, :, :, 0, 1], t[:, :, :, 0, 3], t[:, :, :, 0, 4]],
                        [t[:, :, :, 0, 2], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]]
                       )

    else:

        D_t = np.array([[t[:, :, :, 0, 0], t[:, :, :, 0, 1], t[:, :, :, 0, 3]],
                        [t[:, :, :, 0, 1], t[:, :, :, 0, 2], t[:, :, :, 0, 4]],
                        [t[:, :, :, 0, 3], t[:, :, :, 0, 4], t[:, :, :, 0, 5]]]
                       )

    D_t = np.transpose(D_t, (2, 3, 4, 0, 1))

    val_t, vec_t = np.linalg.eig(D_t)

    vol_shape = t.shape[0]*t.shape[1]*t.shape[2]

    vec_t = vec_t.reshape((vol_shape, 3, 3))
    vec_t = np.transpose(vec_t, (0, 2, 1))
    idx = np.argmax(val_t.reshape((vol_shape, 3)), axis=1)

    peaks = vec_t[range(vol_shape), idx].reshape(t.shape[:3]+(3,)).real

    return peaks


def get_fixel_weight_MF(trk_file: str, MF_dir: str, Patient: str, K: int = 2,
                        method: str = 'angular_weight', streamList: list = []):
    '''
    Get the fixel weights from a tract specified in trk_file and the peaks
    obtained from Microsrcuture Fingerprinting.

    Parameters
    ----------
    trk_file : str
        Path to tractography file (.trk)
    MF_dir : str
        Path to folder containing MF peak files.
    Patient : str
        Patient name in MF_dir folder
    K : int, optional
        Maximum number of fixel in a voxel. The default is 2.
    cfo : bool, optional
        Uses 'closest fixel only' as a relative contribution. The default is
        False.
    streamList : list, optional
        List of int containing the number of the streamlines whose segment-wise
        contribution will be stored. The default is [].

    Returns
    -------
    fixelWeights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.
    number_of_streamlines: int
        Number of streamlines in the tractogram
    outputVoxelStream : list
        List of the relative contribution of each streamline segment for the
        streamlines specified in streamList.
    outputSegmentStream :list
        List of the relative contribution of each voxel for the streamlines
        specified in streamList.

    '''

    # Tract -----------------

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    # MF peaks --------------

    tList = []
    for k in range(K):
        # !!!
        img = nib.load(MF_dir+Patient+'_mf_peak_f'+str(k)+'.nii.gz')
        t = img.get_fdata()

        # If finger peaks organised as ...
        if not t.shape[3] == 3 and t.shape[0] == 6:
            t = t6ToMFpeak(t)

        tList.append(t)

    fList = []
    if method == 'vol':     # Relative volume fraction

        for k in range(K):
            # !!!
            img = nib.load(MF_dir+Patient+'_mf_frac_f'+str(k)+'.nii.gz')
            f = img.get_fdata()

            fList.append(f)

    return get_fixel_weight(trk, tList, method, streamList, fList)


def get_fixel_weight_DIAMOND(trk_file: str, DIAMOND_dir: str, Patient: str,
                             K: int = 2, method: str = 'angular_weight',
                             streamList: list = []):
    '''
    Get the fixel weights from a tract specified in trk_file and the tensors
    obtained from DIAMOND.

    Parameters
    ----------
    trk_file : str
        Path to tractography file (.trk)
    DIAMOND_dir : str
        Path to folder containing MF peak files.
    Patient : str
        Patient name in DIAMOND_dir folder
    K : int, optional
        Maximum number of fixel in a voxel. The default is 2.
    cfo : bool, optional
        Uses 'closest fixel only' as a relative contribution. The default is
        False.
    streamList : list, optional
        List of int containing the number of the streamlines whose segment-wise
        contribution will be stored. The default is [].

    Returns
    -------
    fixelWeights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.
    number_of_streamlines: int
        Number of streamlines in the tractogram
    outputVoxelStream : list
        List of the relative contribution of each streamline segment for the
        streamlines specified in streamList.
    outputSegmentStream :list
        List of the relative contribution of each voxel for the streamlines
        specified in streamList.

    '''

    # Tract -----------

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    # t0 & t1 ---------------

    if os.path.isfile(DIAMOND_dir+Patient+'_diamond_fractions.nii.gz'):

        f = nib.load(DIAMOND_dir+Patient +
                     '_diamond_fractions.nii.gz').get_fdata()

    tList = []
    for k in range(K):
        img = nib.load(DIAMOND_dir+Patient+'_diamond_t'+str(k)+'.nii.gz')
        t = img.get_fdata()

        # Removes tensor k where frac_k == 0
        if os.path.isfile(DIAMOND_dir+Patient+'_diamond_fractions.nii.gz'):

            ft = f[:, :, :, 0, k]

            t[ft == 0, :, :] = [[0, 0, 0, 0, 0, 0]]

        tList.append(tensor_to_peak(t))

    fList = []
    if method == 'vol':     # Relative volume fraction

        for k in range(K):

            fk = f[:, :, :, 0, k]

            fList.append(fk)

    return get_fixel_weight(trk, tList, method, streamList, fList)


def get_fixel_weight(trk, tList: list, method: str = 'angular_weight',
                     streamList: list = [], fList: list = []):
    '''
    Get the fixel weights from a tract specified in trk_file.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    tList : list
        List of 4-D arrays of shape (x,y,z,3) containing peak information.
    cfo : bool, optional
        Uses 'closest fixel only' as a relative contribution. The default is
        False.
    streamList : list, optional
        Plots every streamline corresponding to the number (int) in the list.
        The default is [].

    Returns
    -------
    fixelWeights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.
    number_of_streamlines: int
        Number of streamlines in the tractogram
    outputVoxelStream : list, optional
        List of the relative contribution of each streamline segment for the
        streamlines specified in streamList.
    outputSegmentStream :list, optional
        List of the relative contribution of each voxel for the streamlines
        specified in streamList.

    '''

    phi_maps = {}
    K = len(tList)
    # t10=np.zeros(tList[0].shape)
    fixelWeights = np.zeros(tList[0].shape[0:3]+(K,))

    sList = tract_to_streamlines(trk)

    outputVoxelStream = []
    outputSegmentStream = []

    for h, streamline in enumerate(sList):

        voxelStream = {}
        segmentStream = []

        previous_point = streamline[0, :]

        for i in range(1, streamline.shape[0]):

            point = streamline[i, :]

            # voxList=voxels_from_segment(point,previous_point)
            voxList = compute_subsegments(previous_point, point)

            vs = (point-previous_point)   # Tract deltas

            for x, y, z in voxList:

                x, y, z = (int(x), int(y), int(z))

                vList = []
                nList = []    # Null list, boolean

                for t in tList:

                    v = t[x, y, z, :]
                    vList.append(v)

                    # Fingerprint : null vector = [0,0,0]
                    # Diamond : null vector = [1,0,0]
                    nList.append(all(v == 0 for v in v[1:]))

                if all(nList):       # If no tensor in voxel
                    # t10[x,y,z,:]=np.zeros(3)
                    continue

                if (x, y, z) not in phi_maps:       # Never been to this voxel
                    phi_maps[(x, y, z)] = [[], []]

                # !!! Computed twice (also in angular_weighting/cfo)
                aList = []    # angle list
                for k, v in enumerate(vList):
                    if nList[k]:
                        aList.append(1000)
                    else:
                        aList.append(angle_difference(vs, v))

                min_k = np.argmin(aList)
                phi_maps[(x, y, z)][0].append(aList[min_k])

                if method == 'cfo':     # Closest-fixel-only
                    coefList = closest_fixel_only(vs, vList, nList)
                elif method == 'vol':   # Relative volume fraction
                    if len(vList) != len(fList):
                        warnings.warn("Warning : The number of fixels (" +
                                      str(len(vList)) +
                                      ") does not correspond to the number " +
                                      " of fractions given ("+str(len(fList))+").")
                    coefList = fraction_weighting(
                        (x, y, z), vList, nList, fList)
                else:   # Angular weighting
                    coefList = angular_weighting(vs, vList, nList)

                for k, coef in enumerate(coefList):
                    fixelWeights[x, y, z, k] += voxList[(x, y, z)]*coef
                phi_maps[(x, y, z)][1].append(
                    voxList[(x, y, z)]*coefList[min_k])

                if h in streamList:

                    s = []
                    for coef in coefList:
                        s.append(voxList[(x, y, z)]*coef)
                    segmentStream.append([(x, y, z)]+s)

                    if (x, y, z) not in voxelStream:
                        voxelStream[(x, y, z)] = []
                        for k, coef in enumerate(coefList):
                            voxelStream[(x, y, z)].append(
                                voxList[(x, y, z)]*coef)
                    else:
                        for k, coef in enumerate(coefList):
                            voxelStream[(x, y, z)
                                        ][k] += voxList[(x, y, z)]*coef

            previous_point = point

        if h in streamList:

            outputVoxelStream.append(voxelStream)
            outputSegmentStream.append(segmentStream)

    # img=nib.load(DIAMOND_dir+'_diamond_t0.nii.gz')
    # t=peak_to_tensor(t10)
    # save_nifti(output_dir+'afToTensor.nii.gz', t,img.affine,img.header)

    if streamList:
        return (fixelWeights, phi_maps, len(sList), outputVoxelStream,
                outputSegmentStream)
    else:
        return (fixelWeights, phi_maps, len(sList))


def main_fixel_map(fixelWeights):
    '''
    Ouputs a map representing the most aligned fixel.

    Parameters
    ----------
    fixelWeights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    mainFixelMap : 3-D array of shape (x,y,z).
        The array contains int values representing the most aligned fixel.

    '''

    mainFixelMap = np.zeros(fixelWeights.shape[0:3])
    mainFixelMap[fixelWeights[:, :, :, 0] > fixelWeights[:, :, :, 1]] = 1
    mainFixelMap[fixelWeights[:, :, :, 0] < fixelWeights[:, :, :, 1]] = 2

    return mainFixelMap


def tract_to_streamlines(trk) -> list:
    '''
    Return a list with the position of each step of each streamline from a
    tractogram.

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file

    Returns
    -------
    sList : list
        List of the streamlines contained in the tractogram.

    '''

    streams = trk.streamlines
    streams_data = trk.streamlines.get_data()
    b = np.float64(streams_data)

    sList = []

    for i, offset in enumerate(streams._offsets):

        sList.append(b[offset:offset+streams._lengths[i], :])

    return sList


def plot_streamline_metrics(streamList: list, metric_maps: list,
                            groundTruth_map=None):
    '''
    Plots the evolution of a metric along the course of a single streamline

    Parameters
    ----------
    streamList : list
        List of segment- or voxel-wise relative contributions.
    metric_maps : list
        List of K 3-D arrays of shape (x,y,z) containing metric estimations.
    groundTruth_map : array, optional
        3-D array of shape (x,y,z)containing the ground truth map.

    Returns
    -------
    None.

    '''

    import matplotlib.pyplot as plt

    K = len(metric_maps)

    for i, stream in enumerate(streamList):
        mList = []
        mcfoList = []
        mgtList = []
        vList = []
        qLists = []
        qTLists = []
        cfoLists = []

        for k in range(K):
            qLists.append([])
            qTLists.append([])
            cfoLists.append([])

        if type(stream) is dict:

            for voxel in stream:

                weight0 = stream[voxel][0]
                weight1 = stream[voxel][1]

                vList.append(str(voxel))
                _append_weights([weight0, weight1], qLists, qTLists)
                _append_CFO_weights_metrics(
                    [weight0, weight1], cfoLists, mcfoList, voxel, metric_maps)

                mList.append((weight0*metric_maps[0][voxel] +
                             weight1*metric_maps[1][voxel]) /
                             (weight0+weight1))

                if groundTruth_map is not None:
                    mgtList.append(groundTruth_map[voxel])

        else:

            for s, segment in enumerate(stream):

                voxel = segment[0]
                weight0 = segment[1]
                weight1 = segment[2]

                vList.append(str(s)+str((voxel[0], voxel[2])))
                _append_weights([weight0, weight1], qLists, qTLists)
                _append_CFO_weights_metrics(
                    [weight0, weight1], cfoLists, mcfoList, voxel, metric_maps)

                mList.append((weight0*metric_maps[0][voxel] +
                             weight1*metric_maps[1][voxel]) /
                             (weight0+weight1))

                if groundTruth_map is not None:
                    mgtList.append(groundTruth_map[voxel])

        fig, axs = plt.subplots(4, 1)
        bottom = [0]*len(cfoLists[0])
        for k, cfoList in enumerate(cfoLists):
            axs[0].bar(vList, cfoList, bottom=bottom, label='Pop '+str(k+1))
            bottom = [sum(x) for x in zip(bottom, cfoList)]
        axs[0].set_ylabel('Fixel weight \n (closest fixel only)')
        bottom = [0]*len(qLists[0])
        for k, qList in enumerate(qLists):
            axs[1].bar(vList, qList, bottom=bottom, label='Pop '+str(k+1))
            bottom = [sum(x) for x in zip(bottom, qList)]
        axs[1].legend()
        axs[1].set_ylabel('Fixel weight \n (angular weighting)')
        bottom = [0]*len(qTLists[0])
        for k, qTList in enumerate(qTLists):
            axs[2].bar(vList, qTList, bottom=bottom, label='Pop '+str(k+1))
            bottom = [sum(x) for x in zip(bottom, qTList)]
        axs[2].legend()
        axs[2].set_ylabel('Relative cont. \n (angular weighting)')
        axs[3].plot(vList, mList, label='Angular weighting')
        axs[3].plot(vList, mcfoList, label='Closest fixel only')
        if groundTruth_map is not None:
            axs[3].plot(vList, mgtList, label='Ground truth')
        axs[3].legend()
        fig.suptitle(i)


def _append_weights(weightList: list, qLists: list, qTLists: list):
    '''


    Parameters
    ----------
    weightList : list
        DESCRIPTION.
    qLists : list
        DESCRIPTION.
    qTLists : list
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    total_weight = 0
    for k in range(len(weightList)):
        qLists[k].append(weightList[k])
        total_weight += weightList[k]

    for k in range(len(weightList)):
        qTLists[k].append(weightList[k]/total_weight)


def _append_CFO_weights_metrics(weightList: list, cfoLists: list,
                                mcfoList: list, voxel: tuple,
                                metric_maps: list):
    '''


    Parameters
    ----------
    weightList : list
        DESCRIPTION.
    cfoLists : list
        DESCRIPTION.
    mcfoList : list
        DESCRIPTION.
    voxel : tuple
        DESCRIPTION.
    metric_maps : list
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    maxWeight = max(weightList)
    m = weightList.index(maxWeight)

    total_weight = 0
    for k in range(len(weightList)):
        total_weight += weightList[k]
        if k != m:
            cfoLists[k].append(0)

    cfoLists[m].append(total_weight)
    mcfoList.append(metric_maps[m][voxel])


def volumetric_agreement(fixel_weights):
    '''
    Parameters
    ----------
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    index : float
        Volumetric agreement index.
    index_map : 3-D array of shape (x,y,z)
        Voxel-wise volumetric agreement index.

    '''

    index_max = np.amax(fixel_weights, axis=3)

    index = np.sum(index_max)/np.sum(fixel_weights)

    index_map = index_max/np.sum(fixel_weights, axis=3)

    return index, index_map


def angular_agreement(phi_maps, volume_shape):
    '''
    Parameters
    ----------
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.

    Returns
    -------
    phi : float
        Angular agreement index.
    phi_map : 3-D array of shape (x,y,z)
        Voxel-wise angular agreement index.

    '''

    phi_sum = np.zeros(volume_shape[:3])
    phi_map = np.zeros(volume_shape[:3])
    w = np.zeros(volume_shape[:3])

    for (x, y, z) in phi_maps:
        lista = np.array(phi_maps[(x, y, z)][0]) * \
            np.array(phi_maps[(x, y, z)][1])
        phi_sum[x, y, z] = round(np.sum(lista), 3)
        w[x, y, z] = sum(phi_maps[(x, y, z)][1])

    phi = np.sum(phi_sum)/np.sum(w)

    phi_map = phi_sum/w

    return phi, phi_map


def total_segment_length(fixel_weights):
    '''
    Parameters
    ----------
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    sc : 3-D array of shape (x,y,z)
        Array contains the total segment length in each voxel.

    '''

    if len(fixel_weights.shape) <= 3:
        return fixel_weights

    sc = np.sum(fixel_weights, axis=3)

    return sc


def get_microstructure_map(fixelWeights, metricMapList: list):
    '''
    Returns a 3D volume representing the microstructure map

    Parameters
    ----------
    fixelWeights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    metricMapList : list
        List of K 3-D arrays of shape (x,y,z) containing metric estimations.

    Returns
    -------
    microMap : 3-D array of shape (x,y,z)
        Array containing the microstructure map

    '''

    microMap = np.zeros(metricMapList[0].shape)
    total_weight = np.sum(fixelWeights, axis=3)

    for k, metricMap in enumerate(metricMapList):
        microMap += metricMap*fixelWeights[:, :, :, k]

    microMap[total_weight != 0] /= total_weight[total_weight != 0]

    return microMap


def get_weighted_mean(microstructure_map, fixel_weights, weighting: str = 'tsl'):
    '''
    Returns the mean value of a microstructure map using either a voxel or
    total segment length (tsl) weighing method. Totals segment length
    attributes more weight to voxels containing mulitple streamline segments.

    Parameters
    ----------
    microstructure_map : 3-D array of shape (x,y,z)
        Array containing the microstructure map
    fixel_weights : 4-D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    weighting : str, optional
        Weighting used for the mean. The default is 'tsl'.

    Returns
    -------
    mean : float
        Weighted mean.
    dev : float
        Weighted sum.

    '''

    tsl = total_segment_length(fixel_weights)

    if weighting == 'roi':
        weighted_map = np.zeros(tsl.shape)
        weighted_map[tsl != 0] = 1
    else:
        weighted_map = tsl

    mean = np.sum(microstructure_map*weighted_map)/np.sum(weighted_map)

    M = np.count_nonzero(weighted_map)

    dev = np.sqrt(np.sum(weighted_map*np.square(microstructure_map-mean)) /
                  ((M-1)/M*np.sum(weighted_map)))

    return mean, dev


def get_weighted_sums(metricMapList: list, fixelWeightList: list):
    '''
    TODO: unify with get_microstructure_map and total_segment_length
    Outdated.

    Parameters
    ----------
    metricMapList : list
        List of K 3-D arrays of shape (x,y,z) containing metric estimations.
    fixelWeightList : list
        List containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    weightedMetricSum : 3-D array of shape (x,y,z)
        Microstructure map.
    fixelWeightSum : 3-D array of shape (x,y,z)
        Total length map.
    M : int
        Number of non-zero pixels in both metric maps.

    '''

    K = len(metricMapList)
    fixelWeightSum = np.zeros(fixelWeightList[0].shape)
    M = 0

    for fixelWeight in fixelWeightList:
        fixelWeightSum += fixelWeight
        M += np.count_nonzero(fixelWeight)

    weightedMetricSum = np.zeros(metricMapList[0].shape)

    for k in range(K):
        weightedMetricSum += metricMapList[k]*fixelWeightList[k]

    return weightedMetricSum, fixelWeightSum, M


def weighted_mean_dev(metricMapList: list, fixelWeightList: list,
                      retainAllValues: bool = False, weighting: str = 'tsl'):
    '''
    Return the weighted means and standard deviation from list of metric maps
    and corresponding fixel weights.
    Outdated.

    Parameters
    ----------
    metricMapList : list
        List of K 3-D arrays of shape (x,y,z) containing metric estimations.
    fixelWeightList : list
        List containing the relative weights of the K fixels in each voxel.
    retainAllValues : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    weightedMean : float
        Weighted mean.
    weightedDev : float
        Weighted sum.
    weightSum : float
        Total segment length.
    minMaxList : list
        [Min, Max]
    weightedList : list


    '''

    K = len(metricMapList)

    weightedMetricSum, fixelWeightSum, M = get_weighted_sums(
        metricMapList, fixelWeightList)
    weightSum = np.sum(fixelWeightSum)

    Max = np.max(weightedMetricSum[fixelWeightSum !=
                 0]/fixelWeightSum[fixelWeightSum != 0])
    Min = np.min(weightedMetricSum[fixelWeightSum !=
                 0]/fixelWeightSum[fixelWeightSum != 0])

    weightedMetrics = np.array(np.nansum((weightedMetricSum)/weightSum))
    weightedMean = float(np.mean(weightedMetrics[weightedMetrics > 0]).real)

    weightedVarSum = np.zeros(metricMapList[0].shape)

    # weightedDev=np.sqrt(np.nansum(weightedVarSum[weightedMetrics>0])/weightSum)

    for k in range(K):
        weightedVarSum += np.square(metricMapList[k] -
                                    weightedMean)*fixelWeightList[k]

    weightedDev = np.sqrt(np.nansum(weightedVarSum)/(weightSum*(M-1)/M))

    if retainAllValues:

        weightedList = []
        wMf = weightedMetricSum.flatten()
        mSf = fixelWeightSum.flatten()

        for i, value in enumerate(list(wMf)):
            if mSf[i] != 0:
                weightedList += [value/mSf[i]]*int(mSf[i])

        return weightedMean, weightedDev, weightSum, [Min, Max], weightedList

    else:

        return weightedMean, weightedDev, weightSum, [Min, Max]
        return weightedMean, weightedDev, weightSum, [Min, Max]
