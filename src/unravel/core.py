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


def voxel_distance(position1: tuple, position2: tuple):
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

    return np.abs(position2.astype(np.int32)-position1.astype(np.int32))


def voxels_from_segment(position1: tuple, position2: tuple,
                        subparts: int = 10) -> tuple:
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
        Divide segment into multiple subsegment. Higher value is more precise
        but increases computation time. The default is 10.

    Returns
    -------
    voxList : dict
        Dictionary of the voxels containing a part of the segment.

    '''

    voxDis = voxel_distance(position1, position2)

    if not np.any(voxDis):  # Trying to gain time
        voxels = position1.astype(np.int32)

        return voxels[np.newaxis, ...], np.ones(1)

    subseg = np.linspace(position1, position2, num=subparts, dtype=np.int32)

    v = np.unique(subseg, return_counts=True, axis=0)

    return v[0], v[1]/subparts


def angle_difference(vs, vf, direction: bool = False):
    '''
    Computes the angle difference between n vectors.

    Parameters
    ----------
    vs : 2D array of size (n,3)
        Segment vector
    vf : 3D array of size (n,3,k)
        List of the k vectors corresponding to each fiber population
    direction : bool, optional
        If False, the vectors are considered to be direction-agnostic : maximum
        angle difference = 90. If True, the direction of the vectors is taken
        into account : maximum angle difference = 180. The default is False.

    Returns
    -------
    ang : 2D array of size (n,k)
        Angle difference (in degrees).
    '''

    # Ensure vs is a 3D array for broadcasting
    if vs.ndim == 2:
        vs = vs[..., np.newaxis]
    if len(vf.shape) == 2:
        vf = vf[..., np.newaxis]

    # Normalize vectors, suppressing warnings for divisions by zero
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    n = np.linalg.norm(vs, axis=1, keepdims=True)
    vsn = np.divide(vs, n, where=(n != 0))
    n = np.linalg.norm(vf, axis=1, keepdims=True)
    vfn = np.divide(vf, n, where=(n != 0))

    # Compute dot product and clamp values to avoid invalid arccos inputs
    dot = np.einsum('ijk,ijk->ik', vsn, vfn)
    dot = np.clip(dot, -1.0, 1.0)

    # Compute the angle in degrees
    ang = np.arccos(dot) * 180 / np.pi

    # Adjust angles if direction is not considered
    if not direction:
        ang = np.where(ang > 90, 180 - ang, ang)

    return ang


def fraction_weighting(point, ff, nf=None):
    '''
    Computes the relative contribution of each fixel k based on its fraction ff
    in voxel (point).

    Parameters
    ----------
    point : 2D array of size (n,3)
        x, y, z coordinates.
    nf : 2D array of size (n,k)
        List of the null k vectors.
    ff : 4D array of size (x,y,z,k)
        Array containing the volume fractions of the k fixels.

    Returns
    -------
    ang_coef : 2D array of size (n,k)
        List of the k coefficients

    '''

    K = nf.shape[1]

    if nf is None:
        nf = np.zeros((point.shape[0],)+(K,))

    x, y, z = point.astype(np.int32).T

    ang_coef = ff[x, y, z, :]

    ang_coef *= (1-nf)
    s = np.sum(ang_coef, axis=1)
    ang_coef = ang_coef/np.stack((s,)*K, axis=1)

    return ang_coef


def closest_fixel_only(vs, vf, nf=None):
    '''
    Computes the relative contributions of the segments in vf to vs using the
    closest-fixel-only approach.

    Parameters
    ----------
    vs : 2D array of size (n,3)
        Segment vector
    vf : 3D array of size (n,3,k)
        List of the k vectors corresponding to each fiber population
    nf : array of size (n,k)
        List of the null k vectors

    Returns
    -------
    ang_coef : 2D array of size (n,k)
        List of the k coefficients

    '''

    if len(vf.shape) <= 2:
        vf = vf[..., np.newaxis]

    K = vf.shape[2]

    if nf is None:
        nf = np.zeros((vs.shape[0],)+(K,))

    angle_diff = angle_difference(vs, vf)
    ang_coef = (angle_diff == np.nanmin(angle_diff, axis=1)
                [:, None]).astype(np.int32)

    ang_coef *= (1-nf.astype(np.int32))
    s = np.sum(ang_coef, axis=1)
    ang_coef = ang_coef/np.stack((s,)*K, axis=1)

    return ang_coef


def angular_weighting(vs, vf, nf=None):
    '''
    Computes the relative contributions of the segments in vf to vs using
    angular weighting.

    Parameters
    ----------
    vs : 2D array of size (n,3)
        Segment vector
    vf : 3D array of size (n,3,k)
        List of the k vectors corresponding to each fiber population
    nf : array of size (n,k)
        List of the null k vectors

    Returns
    -------
    ang_coef : 2D array of size (n,k)
        List of the k coefficients

    '''

    if len(vf.shape) <= 2:
        vf = vf[..., np.newaxis]

    K = vf.shape[2]

    if nf is None:
        nf = np.zeros((vs.shape[0],)+(K,))

    angle_diff = angle_difference(vs, vf)
    # Angle product if not NaN
    prod = np.prod(angle_diff, axis=1, where=~np.isnan(angle_diff))

    # Catching warnings due to angle_fixel=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ang_coef = np.stack((prod,)*K, axis=1, dtype=np.float32)
        np.divide(ang_coef, angle_diff, out=ang_coef,
                  where=~np.isnan(angle_diff))
    ang_coef = np.nan_to_num(ang_coef, nan=1)

    ang_coef *= (1-nf)
    s = np.sum(ang_coef, axis=1, dtype=np.float32)
    s = np.stack((s,)*K, axis=1)
    np.divide(ang_coef, s, out=ang_coef, where=s != 0)

    return ang_coef


def relative_angular_weighting(vs, vf, nf=None):
    '''
    Computes the relative contributions of the segments in vList to vs using
    relative angular weighting, which attributes less weight to fixel
    perpendicular to the streamline segment.

    Parameters
    ----------
    vs : 2D array of size (n,3)
        Segment vector
    vf : 3D array of size (n,3,k)
        List of the k vectors corresponding to each fiber population
    nf : array of size (n,k)
        List of the null k vectors

    Returns
    -------
    ang_coef : 2D array of size (n,k)
        List of the k coefficients

    '''

    if len(vf.shape) <= 2:
        vf = vf[..., np.newaxis]

    K = vf.shape[2]

    if nf is None:
        nf = np.zeros((vs.shape[0],)+(K,))

    angle_diff = angle_difference(vs, vf)
    # Angle product if not NaN
    prod = np.prod(angle_diff, axis=1, where=~np.isnan(angle_diff))

    # Catching warnings due to angle_fixel=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ang_coef = np.stack((prod,)*K, axis=1, dtype=np.float32)
        np.divide(ang_coef, angle_diff, out=ang_coef,
                  where=~np.isnan(angle_diff))
    ang_coef *= (90-angle_diff)
    ang_coef = np.nan_to_num(ang_coef, nan=1)

    ang_coef *= (1-nf)
    s = np.sum(ang_coef, axis=1, dtype=np.float32)
    s = np.stack((s,)*K, axis=1)
    np.divide(ang_coef, s, out=ang_coef, where=s != 0)

    return ang_coef


def get_fixel_weight(trk, peaks, method: str = 'ang', ff=None,
                     return_phi: bool = False, subsegment: int = 10):
    '''
    Get the fixel weights from a tract specified in trk_file.
    TODO : re-implement return_phi

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    peaks : 4D array of shape (x,y,z,3,k)
        List of 4D arrays of shape (x,y,z,3) containing peak information.
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'ang'.
    ff : 4D array of size (x,y,z,k)
        Array containing the volume fractions of the k fixels.
        The default is None.
    subsegment : int
        Number of subsegments per segment (tractography step), increases
        precision at the cost of an increased computation time and RAM usage.
        The default is 10.
    return_phi : bool, optional
        If True, returns the phi_maps used for the angular agreement. Currently
        slows down the code. The default is False.

    Returns
    -------
    fixel_weight : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.

    '''

    assert method in ['ang', 'cfo', 'vol', 'raw'], ("Unknown method : "+method)

    K = peaks.shape[4]

    fixel_weight = np.zeros(peaks.shape[0:3]+(K,), dtype=np.float32)

    streams = trk.streamlines
    point = streams.get_data()

    # Creating subpoints
    subpoint = np.linspace(point, np.roll(point, -1, axis=0),
                           subsegment+1, axis=1)
    point = subpoint[:, :-1, :].reshape(point.shape[0]*subsegment, 3)

    # Computing streamline segment vectors
    vs = np.roll(point, -1, axis=0)-point

    # To allow for variable step size
    dist = np.linalg.norm(vs, axis=1)
    dist = np.stack((dist,)*K, axis=1)

    # Getting fixel vectors
    x, y, z = point.astype(np.int32).T
    vf = peaks[x, y, z, :, :].astype(np.float32)

    # Null fixel vectors
    nf = np.sum(peaks[x, y, z, 1:, :], axis=1) == 0

    # Computing relative contribution
    if method == 'vol':
        coef = fraction_weighting(point, ff, nf)*dist/subsegment
    elif method == 'cfo':
        coef = closest_fixel_only(vs, vf, nf)*dist/subsegment
    elif method == 'ang':
        coef = angular_weighting(vs, vf, nf)*dist/subsegment
    elif method == 'raw':
        coef = relative_angular_weighting(vs, vf, nf)*dist/subsegment
    del point, vs, nf, vf, dist

    # Removing streamline end points
    ends = (streams._offsets+streams._lengths-1)*subsegment
    idx = np.linspace(0, subsegment-1, subsegment, dtype=np.int32)
    ends = ends[:, np.newaxis] + idx
    ends = ends.flatten()
    coef[ends] = [0]*K

    np.add.at(fixel_weight, (x, y, z), coef)

    return fixel_weight


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
    lamb : 2D array of size (3,3), optional
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
    D : 2D array of size (3,3)
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


def t6ToMFpeak(t):
    '''
    (6,17,1,15) with info on 0,2,5 to (17,1,15,3)
    '''

    new_t = np.transpose(t, (1, 2, 3, 0))
    new_t[:, :, :, 0] = new_t[:, :, :, 0]
    new_t[:, :, :, 1] = new_t[:, :, :, 2]
    new_t[:, :, :, 2] = new_t[:, :, :, 5]
    new_t = new_t[:, :, :, :3]

    return new_t


def peak_to_tensor(peaks, norm=None, pixdim=[2, 2, 2]):
    '''
    Takes peaks, such as the ones obtained with Microstructure Fingerprinting,
    and return the corresponding tensor, in the format used in DIAMOND.

    Parameters
    ----------
    peaks : 4D array of size (x,y,z,3)
        Array containing the peaks of shape (x,y,z,3)
    norm : 3D array of size (x,y,z)
        Array containing the normalization factor of shape (x,y,z), usually
        between [0,1].

    Returns
    -------
    t : 5D array of size (x,y,z,1,6)
        Tensor array of shape (x,y,z,1,6).

    '''

    t = np.zeros(peaks.shape[:3]+(1, 6), dtype='float32')

    scaleFactor = 1000 / min(pixdim)

    for xyz in np.ndindex(peaks.shape[:3]):

        if peaks[xyz].all() == 0:
            continue

        dx, dy, dz = peaks[xyz]

        try:
            if norm is None:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor)
            else:
                D = deltas_to_D(dx, dy, dz, vec_len=scaleFactor/norm[xyz])
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
    Takes peaks, such as the ones obtained with DIAMOND, and returns the
    corresponding tensor, in the format used in Microstructure Fingerprinting.
    TODO : Speed up.

    Parameters
    ----------
    t : 5D array
        Tensor array of shape (x,y,z,1,6).

    Returns
    -------
    peaks : 4D array
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
                        method: str = 'ang'):
    '''
    Get the fixel weights from a tract specified in trk_file and the peaks
    obtained from Microsrcuture Fingerprinting.
    OUTDATED.
    TODO: adapt to new architecture.

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
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'ang'.

    Returns
    -------
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.

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
    peaks = np.stack(tList, axis=3)

    fList = []
    if method == 'vol':     # Relative volume fraction

        for k in range(K):
            # !!!
            img = nib.load(MF_dir+Patient+'_mf_frac_f'+str(k)+'.nii.gz')
            f = img.get_fdata()

            fList.append(f)
        ff = np.stack(fList, axis=3)
    else:
        ff = None

    return get_fixel_weight(trk, peaks, method, ff=ff)


def get_fixel_weight_DIAMOND(trk_file: str, DIAMOND_dir: str, Patient: str,
                             K: int = 2, method: str = 'ang'):
    '''
    Get the fixel weights from a tract specified in trk_file and the tensors
    obtained from DIAMOND.
    OUTDATED.
    TODO: adapt to new architecture.

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
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'ang'.

    Returns
    -------
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    phi_maps : dict
        Dictionnary containing the lists of the relative contribution and
        angle difference of the most aligned fixel in each voxel.

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
    peaks = np.stack(tList, axis=3)

    fList = []
    if method == 'vol':     # Relative volume fraction

        for k in range(K):

            fk = f[:, :, :, 0, k]

            fList.append(fk)
        ff = np.stack(fList, axis=3)
    else:
        ff = None

    return get_fixel_weight(trk, peaks, method, ff=ff)


def main_fixel_map(fixel_weights):
    '''
    Ouputs a map representing the most aligned fixel.

    Parameters
    ----------
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    mainFixelMap : 3D array of shape (x,y,z).
        The array contains int values representing the most aligned fixel.

    '''

    mainFixelMap = np.zeros(fixel_weights.shape[0:3], dtype='float32')
    mainFixelMap[fixel_weights[:, :, :, 0] > fixel_weights[:, :, :, 1]] = 1
    mainFixelMap[fixel_weights[:, :, :, 0] < fixel_weights[:, :, :, 1]] = 2

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


def get_streamline_weights(trk, peaks,
                           method_list: list = ['vol', 'cfo', 'ang', 'raw'],
                           streamline_number: int = 0, ff=None,
                           subsegment: int = 1):
    '''

    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    peaks : 4D array of shape (x,y,z,3,k)
        List of 4D arrays of shape (x,y,z,3) containing peak information.
    method_list : list, optional
        List of methods used for the relative contribution, either;
            'ang' : angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is ['vol', 'cfo', 'ang'].
    streamline_number : int, optional
        Number of the streamline to analyse. The default is 0.
    fList : list, optional
        List of 3D arrays (x,y,z) containing the fraction of each fiber
        population. Only used with 'vol' method. The default is [].

    Returns
    -------
    segmentStream :list, optional
        List of the relative contribution of each voxel for the streamline
        specified.

    '''

    sList = tract_to_streamlines(trk)
    point = sList[streamline_number]

    segmentStream = []

    for i in range(len(method_list)):
        segmentStream.append([])

    # Creating subpoints
    subpoint = np.linspace(point, np.roll(point, -1, axis=0),
                           subsegment+1, axis=1)
    point = subpoint[:, :-1, :].reshape(point.shape[0]*subsegment, 3)

    # Computing streamline segment vectors
    next_point = np.roll(point, -1, axis=0)
    vs = next_point-point

    # Getting fixel vectors
    x, y, z = point.astype(np.int32).T
    vf = peaks[x, y, z, :, :].astype(np.float32)

    # Null fixel vectors
    nf = np.sum(peaks[x, y, z, 1:, :], axis=1) == 0

    for j, method in enumerate(method_list):

        if method == 'vol':
            coef = fraction_weighting(point, ff, nf)/subsegment
        elif method == 'cfo':
            coef = closest_fixel_only(vs, vf, nf)/subsegment
        elif method == 'ang':
            coef = angular_weighting(vs, vf, nf)/subsegment
        elif method == 'raw':
            coef = relative_angular_weighting(vs, vf, nf)/subsegment

        segmentStream[j] = np.concatenate((x[..., np.newaxis],
                                           y[..., np.newaxis],
                                           z[..., np.newaxis], coef), axis=1)

    return segmentStream


def plot_streamline_metrics(trk, peaks, metric_maps,
                            method_list: list = ['vol', 'cfo', 'ang', 'raw'],
                            streamline_number: int = 0, ff=None,
                            segment_wise: bool = True, groundTruth_map=None,
                            barplot: bool = True):
    '''
    Plots the evolution of a metric along the course of a single streamline.
    TODO: re-implement barplot


    Parameters
    ----------
    trk : tractogram
        Content of a .trk file
    peaks : 4D array of shape (x,y,z,3,k)
        List of 4D arrays of shape (x,y,z,3) containing peak information.
    metric_maps : 4D array of shape (x,y,z,3,k)
        List of K 4D arrays of shape (x,y,z) containing metric estimations.
    method_list : list, optional
        List of methods used for the relative contribution, either;
            'ang' : angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is ['vol', 'cfo', 'ang'].
    streamline_number : int, optional
        Number of the streamline to analyse. The default is 0.
    fList : list, optional
        List of 3D arrays (x,y,z) containing the fraction of each fiber
        population. Only used with 'vol' method. The default is [].
    segment_wise : bool, optional
        If True then plots for each segment, else plots for each voxel.
        The default is True.
    groundTruth_map : array, optional
        3D array of shape (x,y,z) containing the ground truth map.
    barplot : bool, optional
        If False, does not plot the barplots of the relative contributions.
        The default is True.

    Returns
    -------
    None.

    '''

    streams = get_streamline_weights(trk, peaks, method_list=method_list,
                                     streamline_number=streamline_number,
                                     ff=ff)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1)

    for i, method in enumerate(method_list):

        mList = []
        mgtList = []
        vList = []

        for j in range(streams[i].shape[0]):

            x, y, z = streams[i][j, :3].astype(np.int32)
            coef = streams[i][j, 3:]

            m = np.sum(coef*metric_maps[x, y, z, :])/np.sum(coef)

            mList.append(m)
            vList.append(str(j)+str([x, y, z]))

            if groundTruth_map is not None:
                mgtList.append(groundTruth_map[x, y, z])

        axs.plot(vList, mList, label=method)

    if groundTruth_map is not None:
        axs.plot(vList, mgtList, label='Ground truth')
    axs.legend()
    axs.set_ylabel('Metric')
    axs.set_xlabel('Streamline segment position')
    axs.set_title('Microstructure along streamline')


def plot_streamline_metrics_old(streamList: list, metric_maps: list,
                                groundTruth_map=None, barplot: bool = True,
                                method_list: list = ['ang'], fList: list = []):
    '''
    Plots the evolution of a metric along the course of a single streamline.
    OUTDATED

    Parameters
    ----------
    streamList : list
        List of segment- or voxel-wise relative contributions.
    metric_maps : list
        List of K 3D arrays of shape (x,y,z) containing metric estimations.
    groundTruth_map : array, optional
        3D array of shape (x,y,z)containing the ground truth map.
    barplot : bool, optional
        If False, does not plot the barplots of the relative contributions.
        The default is True.
    method_list : list
        List of method used for the relative contribution, either;
            'ang' : angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'ang'.
    fList : list, optional
        List of 3D arrays (x,y,z) containing the fraction of each fiber
        population. Only used with 'vol' method. The default is [].

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

        if type(stream) is dict:    # Following voxels

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

        else:                   # Following streamline segments

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

        if barplot:
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
            a = 3
        else:
            fig, axs = plt.subplots(1, 1)
            a = 0
        axs[a].plot(vList, mList, label='Angular weighting')
        axs[a].plot(vList, mcfoList, label='Closest fixel only')
        if groundTruth_map is not None:
            axs[a].plot(vList, mgtList, label='Ground truth')
        axs[a].legend()
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
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    index : float
        Volumetric agreement index.
    index_map : 3D array of shape (x,y,z)
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
    phi_map : 3D array of shape (x,y,z)
        Voxel-wise angular agreement index.

    '''

    phi_sum = np.zeros(volume_shape[:3], dtype='float32')
    phi_map = np.zeros(volume_shape[:3], dtype='float32')
    w = np.zeros(volume_shape[:3], dtype='float32')

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
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    sc : 3D array of shape (x,y,z)
        Array contains the total segment length in each voxel.

    '''

    sc = np.sum(fixel_weights, axis=len(fixel_weights.shape)-1)

    return sc


def get_microstructure_map(fixel_weights, metric_maps):
    '''
    Returns a 3D volume representing the microstructure map

    Parameters
    ----------
    fixel_weights : 4D array of shape (x,y,z,K)
        Array containing the relative weights of the K fixels in each voxel.
    metric_maps : 4D array of shape (x,y,z,K)
        List of K 3D arrays of shape (x,y,z) containing metric estimations.

    Returns
    -------
    micro_map : 3D array of shape (x,y,z)
        Array containing the microstructure map

    '''

    total_weight = np.sum(fixel_weights, axis=-1)

    micro_map = np.sum(metric_maps*fixel_weights, axis=-1)

    micro_map[total_weight != 0] /= total_weight[total_weight != 0]

    return micro_map


def get_weighted_mean(micro_map, fixel_weights,
                      weighting: str = 'tsl'):
    '''
    Returns the mean value of a microstructure map using either a voxel or
    total segment length (tsl) weighing method. Totals segment length
    attributes more weight to voxels containing mulitple streamline segments.

    Parameters
    ----------
    micro_map : 3D array of shape (x,y,z)
        Array containing the microstructure map
    fixel_weights : 4D array of shape (x,y,z,K)
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
        weighted_map = np.where(tsl != 0, 1, 0)
    else:
        weighted_map = tsl

    if np.sum(weighted_map) == 0:
        return 0, 0

    mean = np.sum(micro_map*weighted_map)/np.sum(weighted_map)

    M = np.count_nonzero(weighted_map)

    if M == 1:
        return mean, 0

    den = (M-1)/M*np.sum(weighted_map)
    dev = np.sqrt(np.divide(np.sum(weighted_map*np.square(micro_map-mean)),
                            den))

    return mean, dev


def get_weighted_sums(metric_maps: list, fixelWeightList: list):
    '''
    TODO: unify with get_microstructure_map and total_segment_length
    Outdated.

    Parameters
    ----------
    metric_maps : list
        List of K 3D arrays of shape (x,y,z) containing metric estimations.
    fixelWeightList : list
        List containing the relative weights of the K fixels in each voxel.

    Returns
    -------
    weightedMetricSum : 3D array of shape (x,y,z)
        Microstructure map.
    fixel_weightsum : 3D array of shape (x,y,z)
        Total length map.
    M : int
        Number of non-zero pixels in both metric maps.

    '''

    K = len(metric_maps)
    fixel_weightsum = np.zeros(fixelWeightList[0].shape, dtype='float32')
    M = 0

    for fixelWeight in fixelWeightList:
        fixel_weightsum += fixelWeight
        M += np.count_nonzero(fixelWeight)

    weightedMetricSum = np.zeros(metric_maps[0].shape, dtype='float32')

    for k in range(K):
        weightedMetricSum += metric_maps[k]*fixelWeightList[k]

    return weightedMetricSum, fixel_weightsum, M


def weighted_mean_dev(metric_maps: list, fixelWeightList: list,
                      retainAllValues: bool = False, weighting: str = 'tsl'):
    '''
    Return the weighted means and standard deviation from list of metric maps
    and corresponding fixel weights.
    Outdated.

    Parameters
    ----------
    metric_maps : list
        List of K 3D arrays of shape (x,y,z) containing metric estimations.
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

    K = len(metric_maps)

    weightedMetricSum, fixel_weightsum, M = get_weighted_sums(
        metric_maps, fixelWeightList)
    weightSum = np.sum(fixel_weightsum)

    Max = np.max(weightedMetricSum[fixel_weightsum !=
                 0]/fixel_weightsum[fixel_weightsum != 0])
    Min = np.min(weightedMetricSum[fixel_weightsum !=
                 0]/fixel_weightsum[fixel_weightsum != 0])

    weightedMetrics = np.array(np.nansum((weightedMetricSum)/weightSum))
    weightedMean = float(np.mean(weightedMetrics[weightedMetrics > 0]).real)

    weightedVarSum = np.zeros(metric_maps[0].shape)

    # weightedDev=np.sqrt(np.nansum(weightedVarSum[weightedMetrics>0])/weightSum)

    for k in range(K):
        weightedVarSum += np.square(metric_maps[k] -
                                    weightedMean)*fixelWeightList[k]

    weightedDev = np.sqrt(np.nansum(weightedVarSum)/(weightSum*(M-1)/M))

    if retainAllValues:

        weightedList = []
        wMf = weightedMetricSum.flatten()
        mSf = fixel_weightsum.flatten()

        for i, value in enumerate(list(wMf)):
            if mSf[i] != 0:
                weightedList += [value/mSf[i]]*int(mSf[i])

        return weightedMean, weightedDev, weightSum, [Min, Max], weightedList

    else:

        return weightedMean, weightedDev, weightSum, [Min, Max]
