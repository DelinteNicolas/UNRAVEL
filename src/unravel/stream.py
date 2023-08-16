# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:25:30 2023

@author: DELINTE Nicolas
"""


import numpy as np
from sklearn.cluster import KMeans
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
from dipy.io.streamline import load_tractogram, save_tractogram


def extract_nodes(trk_file: str, level: int = 3):
    '''
    The start is assumed to be the lowest position along the last axis.

    Parameters
    ----------
    trk_file : str
        Path to tractogram file
    level : int, optional
        Number of steps in the mean streamline trajectory. The number of steps
        is equal to (2**level)+1. The default is 3.

    Returns
    -------
    point_array : 2D array of size (n, 3)
        Coordinates (x,y,z) of the n mean trajectory points.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    streams = trk.streamlines
    streams_data = trk.streamlines.get_data()

    # Clustering end nodes based on position
    end_0 = streams_data[streams._offsets, :]
    end_1 = np.roll(streams_data[streams._offsets-1, :], -1, axis=0)
    kmeans = KMeans(n_clusters=2, n_init="auto").fit(end_0)

    # Assigning start and end based on clustering
    start = end_0.copy()
    end = end_1.copy()
    start[kmeans.labels_ == 1, :] = end_1[kmeans.labels_ == 1, :]
    end[kmeans.labels_ == 1, :] = end_0[kmeans.labels_ == 1, :]

    # Only compute the mean end points of long fibers [Q3:Q3+1.5*IQR]
    q1, q3 = np.percentile(streams._lengths, [25, 75])
    long_streamlines = streams._lengths > q3
    outlier_streamlines = streams._lengths > q3+1.5*(q3-q1)
    selec_streamlines = long_streamlines*~outlier_streamlines
    m_start = np.mean(start[selec_streamlines], axis=0)
    m_end = np.mean(end[selec_streamlines], axis=0)

    # Re-orders start and end based on last axis position
    # !!! does not work on left-right starts if not in axial view
    if m_start[-1] > m_end[-1]:
        m_start, m_end = m_end, m_start

    # Iterating over specified level ---------------------------------

    # point_array = np.vstack((m_start, m_end))

    point_array = np.zeros((2**level+1, 3))
    point_array[0] = m_start
    point_array[-1] = m_end
    normal_array = np.zeros(point_array.shape)
    normal_array[0] = m_start-m_end
    normal_array[-1] = m_start-m_end

    for j in range(level):
        for i in range(2**j):

            m_start = point_array[2**(level-j-1)*2*i]
            m_end = point_array[2**(level-j-1)*(2*i+2)]

            # Computing normal of perpendicular surface at midpoint
            midpoint = (m_start+m_end)/2
            normal = m_start-m_end
            normal_array[2**(level-j-1)*(2*i+1)] = normal

            # Computing normal at start
            normal_previous = normal_array[2**(level-j-1)*(2*i+2)]
            ns_previous = streams_data-m_start
            sign_previous = np.where(np.sum(ns_previous*normal_previous,
                                            axis=1) > 0, 1, -1)

            # Computing normal at end
            normal_next = normal_array[2**(level-j-1)*(2*i+2)]
            ns_next = streams_data-m_end
            sign_next = np.where(np.sum(ns_next*normal_next, axis=1) > 0, 1, -1)

            # Creating filter based on previous and next surface
            mp_previous = midpoint-m_start
            mp_next = midpoint-m_end
            mp_sign_previous = np.where(np.sum(mp_previous*normal_previous) > 0,
                                        1, -1)
            mp_sign_next = np.where(np.sum(mp_next*normal_next) > 0, 1, -1)
            sign_previous = np.where(sign_previous == mp_sign_previous, 1, 0)
            sign_next = np.where(sign_next == mp_sign_next, 1, 0)
            idx_filter = np.argwhere(sign_next+sign_previous != 2)

            # Find indexes that cross the surface
            ns = streams_data-midpoint
            sign = np.where(np.sum(ns*normal, axis=1) > 0, 1, 0)
            idx = np.argwhere(abs(np.roll(sign, 1)-sign) == 1)
            idx = np.array(
                list(filter(lambda x: x not in streams._offsets, idx)))
            idx = np.array(list(filter(lambda x: x not in idx_filter, idx)))

            # Computing mean position on the surface
            points = streams_data[idx, :]
            point_array[2**(level-j-1)*(2*i+1)] = np.mean(points, axis=0)

    return point_array


def get_streamline_number_from_index(streams, index: int) -> int:
    '''


    Parameters
    ----------
    streams : streamlines.array_sequence.ArraySequence
        DESCRIPTION.
    index : int
        Number of the tractography point (x,y,z).

    Returns
    -------
    nb : int
        Streamline number.

    '''

    offsets = np.append(streams._offsets, streams.total_nb_rows)
    nb = int(np.argwhere(offsets-index > 0)[0, 0]-1)

    return nb


def remove_streamlines(streams, idx: int):
    '''


    Parameters
    ----------
    streams : streamlines.array_sequence.ArraySequence
        DESCRIPTION.
    idx : int
        Streamline number.

    Yields
    ------
    sl : streamline generator
        DESCRIPTION.

    '''

    for i, sl in enumerate(streams):
        if i not in idx:
            yield sl


def remove_outlier_streamlines(trk_file, point_array, out_file: str = None):
    '''
    Removes streamlines that are outliers for more than half of the bundle
    trajectory based on the distance to the mean trajectory.

    Parameters
    ----------
    trk_file : str
        Path to tractogram file.
    point_array : 2D array of size (n, 3)
        Coordinates (x,y,z) of the n mean trajectory points.
    out_file : str, optional
        Path to output file. The default is None.

    Returns
    -------
    None.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    streams = trk.streamlines
    streams_data = trk.streamlines.get_data()

    dist = np.zeros((point_array.shape[0], len(streams._offsets)))

    for i, point in enumerate(point_array):

        if i == 0:
            continue
        if i == point_array.shape[0]-1:
            break

        # Computing normal of perpendicular surface at midpoint
        midpoint = point_array[i]
        normal = point_array[i-1]-point_array[i+1]

        # Find indexes that cross the surface
        ns = streams_data-midpoint
        sign = np.where(np.sum(ns*normal, axis=1) > 0, 1, 0)
        idx = np.argwhere(abs(np.roll(sign, 1)-sign) == 1)
        idx = np.array(list(filter(lambda x: x not in streams._offsets, idx)))
        # Find position
        idx_pos = np.take_along_axis(streams_data, idx, axis=0)
        # Find distance
        idx_dist = np.linalg.norm(idx_pos-np.repeat(midpoint[np.newaxis, :],
                                                    idx_pos.shape[0], axis=0),
                                  axis=1)

        for j, i_dist in enumerate(idx_dist):

            n = get_streamline_number_from_index(streams, idx[j])
            dist[i, n] = i_dist

    # Compute outliers
    q1, q3 = np.percentile(dist, [25, 75], axis=1)
    iqr = q3+1.5*(q3-q1)
    outliers = dist > np.repeat(iqr[:, np.newaxis], dist.shape[1], axis=1)

    # Remove if more than half of pathway is outlier
    n_sign = np.sum(outliers, axis=0)
    n_sign = np.where(n_sign > (len(point_array)-2)/2, 1, 0)
    n_idx = np.argwhere(n_sign == 1)

    streams = remove_streamlines(streams, n_idx)

    print(str(len(n_idx))+' streamlines removed from tract')

    if out_file is None:
        out_file = trk_file

    trk_new = StatefulTractogram(
        streams, trk, Space.VOX, origin=Origin.TRACKVIS)

    save_tractogram(trk_new, out_file)
