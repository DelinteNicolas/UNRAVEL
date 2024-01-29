# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:41:36 2023

@author: DELINTE Nicolas
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
from scipy.interpolate import Akima1DInterpolator
import pyvista
import matplotlib.pyplot as plt
from matplotlib import cm
from unravel.core import (angular_weighting,  relative_angular_weighting,
                          closest_fixel_only)


def grayscale_to_rgb(array):
    '''
    Reapeats a 3D array three times to create a 3D rgb image.

    Parameters
    ----------
    array : 3-D array of shape (x,y,z)
        Grayscale array.

    Returns
    -------
    array : 3-D array of shape (x,y,z,3)
        grayscale (rgb) image.

    '''

    array = np.repeat(array[..., np.newaxis], 3, axis=3)

    return array


def overlap_volumes(vol_list: list, rgb: bool = True, order: int = 0):
    '''
    Overlaps multiple volumes, zero is taken as transparent.
    Order of list is important : [foreground,...,background]

    Parameters
    ----------
    vol_list : list
        List of 3-D array of shape (x,y,z) or (x,y,z,3).
    rgb : bool, optional
        Output rgb volume. The default is True.
    order : int, optional
        Increases quality when increasing resolution, also increases computation
        time. The default is 0.


    Returns
    -------
    back : 3-D array of shape (x,y,z) or (x,y,z,3).
        Overlaped volumes.

    '''

    max_size = [0, 0, 0]

    for vol in vol_list:
        x, y, z = vol.shape[:3]
        for axis, i in enumerate([x, y, z]):
            if i > max_size[axis]:
                max_size[axis] = i

    back = np.zeros(tuple(max_size)+(3,))

    while len(vol_list) > 0:

        layer = vol_list.pop()
        if layer.shape[-1] != 3:
            layer = grayscale_to_rgb(layer)

        size = list(layer.shape[:3])

        layer = zoom(layer, tuple([i / j for i, j in zip(max_size, size)])+(1,),
                     order=order)

        # Normalize
        layer /= np.max(layer)

        back[np.sum(layer, axis=3) != 0] = layer[np.sum(layer, axis=3) != 0]

    if rgb:

        return back

    else:

        return back[:, :, :, 0]


def convert_to_gif(array, output_folder: str, extension: str = 'webp',
                   axis: int = 2, transparency: bool = False,
                   keep_frames: bool = False):
    '''
    Creates a GIF from a 3D volume.

    Parameters
    ----------
    array : 3-D array of shape (x,y,z) or (x,y,z,3)
        DESCRIPTION.
    output_folder : str
        Output filename. Ex: 'output_path/filename'
    extension : str, optional
        File format. The default is 'webp'.
    axis : int, optional
        Axis number to iterate over. The default is 2.
    transparency : bool, optional
        If True, zero is converted to transparent. The default is False.
    keep_frames : bool, optional
        Only if transparent and gif. Overlaps new frames onto old frames.
        The default is False.

    Returns
    -------
    None.

    '''

    frames = []

    # Normalize
    array /= np.max(array)

    if len(array.shape) == 3:       # If not RGB
        array = grayscale_to_rgb(array)

    for i in tqdm(range(array.shape[axis])):

        slic = tuple([i if d == axis else slice(None)
                      for d in range(len(array.shape))])

        data = array[slic]

        if transparency:
            alpha = (np.sum(data, axis=2) != 0)*1
            data = np.dstack((data, alpha))

        data = (data*255).astype('uint8')

        image = Image.fromarray(data)
        frames.append(image)

    if keep_frames:
        disposal = 0
    else:
        disposal = 2

    frames[0].save(output_folder+'.'+extension,
                   lossless=True, save_all=True, append_images=frames,
                   disposal=disposal)


def plot_alpha_surface_matplotlib(vf: list, method: str = 'raw',
                                  weighting_function=None,
                                  show_v: bool = False):
    '''
    Computes and plots the mesh for the alpha coefficient surface based on the
    vectors of vf.

    Parameters
    ----------
    vf : list
        List of the k vectors corresponding to each fiber population
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'raw'.
    weighing_function : function, optional
        Overwrites the weighing function given in method to this method. Used
        for testing. The default is None.
    show_v : bool, optional
        Show vectors. The default is False.

    Returns
    -------
    None.

    '''

    x, y, z, coef = compute_alpha_surface(vf, method=method,
                                          weighting_function=weighting_function)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, facecolors=cm.plasma(
        coef), rstride=1, cstride=1)
    if show_v:
        for j in range(vf.shape[2]):
            v = vf[:, :, j]/np.linalg.norm(vf[:, :, j], axis=1)*2.5
            if j == 0:
                ax.plot([-v[0, 0], v[0, 0]], [-v[0, 1], v[0, 1]],
                        zs=[-v[0, 2], v[0, 2]], color='orange')
            else:
                ax.plot([-v[0, 0], v[0, 0]], [-v[0, 1], v[0, 1]],
                        zs=[-v[0, 2], v[0, 2]], color='white')
    ax.set_aspect('equal')

    plt.show()


def plot_alpha_surface_pyvista(vf, method: str = 'raw', weighting_function=None,
                               show_v: bool = False, v_color: str = 'white',
                               mesh_size: int = 200,
                               background_color: str = 'grey'):
    '''
    Computes and plots the mesh for the alpha coefficient surface based on the
    vectors of vf.

    Parameters
    ----------
    vf : list
        List of the k vectors corresponding to each fiber population
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'raw'.
    weighing_function : function, optional
        Overwrites the weighing function given in method to this method. Used
        for testing. The default is None.
    show_v : bool, optional
        Show vectors. The default is False.
    v_color : str, optional
        Vector color. The default is white.
    mesh_size : int, optional
        Resolution of the 3D surface. The default is 200.
    background_color: str, otpional
        Color of the background in the 3D render. The default is grey.

    Returns
    -------
    None.

    '''

    x, y, z, coef = compute_alpha_surface(vf, method=method,
                                          weighting_function=weighting_function,
                                          mesh_size=mesh_size)

    pc = pyvista.StructuredGrid(x, y, z)
    pl = pyvista.Plotter()
    _ = pl.add_mesh(pc, cmap='plasma', scalars=coef.T.flatten(),
                    smooth_shading=True, show_scalar_bar=False)
    if show_v:
        points = []
        for j in range(vf.shape[2]):
            v = vf[:, :, j]
            v = np.squeeze(v)
            v = v/np.linalg.norm(v)*2.5
            points.append(v*-1)
            points.append(v)
            if j == 0:
                _ = pl.add_lines(np.array(points),
                                 label=str(v), color='orange')
            else:
                _ = pl.add_lines(np.array(points),
                                 label=str(v), color=v_color)
            points = []
        pl.add_legend()
    pl.background_color = background_color
    pl.show()


def compute_alpha_surface(vf, method: str = 'raw',
                          weighting_function=None, mesh_size: int = 200):
    '''
    Computes the mesh for the alpha coefficient surface based on the vectors of
    vf.

    Parameters
    ----------
    vf : list
        List of the k vectors corresponding to each fiber population
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'raw'.
    weighing_function : function, optional
        Overwrites the weighing function given in method to this method. Used
        for testing. The default is None.

    Returns
    -------
    x : array of float64 of size ( mesh_size, mesh_size)
        Mesh X coordinates.
    y : array of float64 of size ( mesh_size, mesh_size)
        Mesh Y coordinates.
    z : array of float64 of size ( mesh_size, mesh_size)
        Mesh Z coordinates.
    coef : array of float64 of size ( mesh_size, mesh_size)
        Alpha coefficients.

    '''

    nf = np.zeros((1,)+(vf.shape[0],))

    u = np.linspace(0, 2 * np.pi,  mesh_size)
    v = np.linspace(0, np.pi,  mesh_size)

    x = np.outer(np.cos(u), np.sin(v)).flatten()
    y = np.outer(np.sin(u), np.sin(v)).flatten()
    z = np.outer(np.ones(np.size(u)), np.cos(v)).flatten()

    vs = np.stack((x, y, z), axis=1)

    if weighting_function is not None:
        a = weighting_function(vs, vf, nf)[:, 0]
    elif method == 'raw':
        a = relative_angular_weighting(vs, vf, nf)[:, 0]
    elif method == 'cfo':
        a = closest_fixel_only(vs, vf, nf)[:, 0]
    else:
        a = angular_weighting(vs, vf, nf)[:, 0]

    x *= (a+1)
    y *= (a+1)
    z *= (a+1)
    coef = a

    x = np.reshape(x, (mesh_size,  mesh_size))
    y = np.reshape(y, (mesh_size,  mesh_size))
    z = np.reshape(z, (mesh_size,  mesh_size))
    coef = np.reshape(coef, (mesh_size,  mesh_size))

    return x, y, z, coef


def export_alpha_surface(vf, output_path: str, method: str = 'raw',
                         show_v: bool = True, v_color: str = 'white',
                         weighting_function=None, mesh_size: int = 200):
    '''
    Computes and exports the mesh for the alpha coefficient surface based on the
    vectors of vList.

    Tutorial to powerpoint: save as .gltf, open with 3D viewer, save as .glb,
    open with 3D builder then repair then save as .3mf

    Parameters
    ----------
    vList : list
        List of the k vectors corresponding to each fiber population
    output_path : str
        Output filename.
    method : str, optional
        Method used for the relative contribution, either;
            'ang' : angular weighting
            'raw' : relative angular weighting
            'cfo' : closest-fixel-only
            'vol' : relative volume weighting.
        The default is 'raw'.
    weighing_function : function, optional
        Overwrites the weighing function given in method to this method. Used
        for testing. The default is None.
    show_v : bool, optional
        Show vectors. The default is True.
    v_color : str, optional
        Vector color. The default is white.
    mesh_size : int, optional
        Resolution of the 3D surface. The default is 200.

    Returns
    -------
    None.

    '''

    x, y, z, coef = compute_alpha_surface(vf, method=method,
                                          weighting_function=weighting_function,
                                          mesh_size=mesh_size)

    pc = pyvista.StructuredGrid(x, y, z)
    pl = pyvista.Plotter()
    _ = pl.add_mesh(pc, cmap='plasma', scalars=coef.T.flatten(),
                    smooth_shading=True, show_scalar_bar=False)
    if show_v:
        points = []
        for j in range(vf.shape[2]):
            v = vf[:, :, j]
            v = np.squeeze(v)
            v = v/np.linalg.norm(v)*2.5
            points.append(v*-1)
            points.append(v)
            if j == 0:
                _ = pl.add_lines(np.array(points),
                                 label=str(v), color='orange')
            else:
                _ = pl.add_lines(np.array(points),
                                 label=str(v), color=v_color)
            points = []
    pl.export_gltf(output_path)


def plot_nodes_and_surfaces(point_array, only_nodes: bool = False):
    '''
    Visualize output of stream.extract_nodes

    Parameters
    ----------
    point_array : 2D array of size (n, 3)
        Coordinates (x,y,z) of the n mean trajectory points.
    only_nodes : bool, optional
        Only plot the nodes and not the planes. The default is False.

    Returns
    -------
    None.

    '''

    surf_array = np.zeros((len(point_array)-1, 3, 2, 2))

    for i, midpoint in enumerate(point_array):

        if i == 0:
            continue
        if i == point_array.shape[0]-1:
            break

        m_start = point_array[i-1]
        m_end = point_array[i+1]

        # Computing perpendicular surface at midpoint
        normal = -m_start+m_end
        d = -np.sum(normal * midpoint)
        delta = 5
        xlim = midpoint[0] - delta, midpoint[0] + delta
        ylim = midpoint[1] - delta, midpoint[1] + delta
        xx, yy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
        zz = -(normal[0] * xx + normal[1] * yy + d) / normal[2]

        surf_array[i, 0, :, :] = xx
        surf_array[i, 1, :, :] = yy
        surf_array[i, 2, :, :] = zz

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, point in enumerate(point_array):
        x, y, z = point
        ax.scatter(x, y, z, marker='o', color='orange')
        if i != 0:
            x_o, y_o, z_o = point_array[i-1]
            ax.plot([x, x_o], [y, y_o], [z, z_o], color='orange')
    plt.axis('equal')
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()

    if only_nodes is False:
        for xx, yy, zz in surf_array:
            ax.plot_surface(xx, yy, zz, color='grey', alpha=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    plt.show()


def plot_roi_sections(roi, voxel: bool = False, background: str = 'grey'):
    '''


    Parameters
    ----------
    roi : 3D array of shape (x,y,z)
        Labeled volume of n sections of a tract.
    voxel : bool, optional
        If true, plots voxels. I False, plots a smoothed surface.
        The default is False.
    background : str, optional
        Color of the background. The default is 'grey'.

    Returns
    -------
    None.

    '''

    datapv = pyvista.wrap(roi)
    datapv.cell_data['labels'] = roi[:-1, :-1, :-1].flatten(order='F')
    vol = datapv.threshold(value=1, scalars='labels')
    mesh = vol.extract_surface()

    smooth = mesh.smooth_taubin(n_iter=12)

    if voxel:
        vol.plot(cmap='Set3', background=background, scalars='labels')
    else:
        smooth.plot(cmap='Set3', background=background, scalars='labels')


def plot_metric_along_trajectory(mean, dev, new_fig: bool = True,
                                 label: str = ''):
    '''
    Plots the output of unravel.analysis.get_metric_along_trajectory.

    Parameters
    ----------
    mean : 1D array of size (n)
        DESCRIPTION.
    dev : 1D array of size (n)
        DESCRIPTION.
    new_fig : bool, optional
        If false, print the plot on the previous 'plt' figure. Useful when
        plotting multiple lines in a single plot. The default is True.
    label : str, optional
        Line label. The default is ''.

    Returns
    -------
    None.

    '''

    spline = Akima1DInterpolator(range(len(mean)), mean)
    std_spline = Akima1DInterpolator(range(len(mean)), dev)
    xs = np.arange(1, len(mean)-1, 0.1)
    ys = spline(xs)
    stds = std_spline(xs)

    if new_fig:
        plt.figure()
    plt.plot(xs, ys, label=label)
    plt.fill_between(xs, np.array(ys)-np.array(stds),
                     np.array(ys) + np.array(stds), alpha=.15)
