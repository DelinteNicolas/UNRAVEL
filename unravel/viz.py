# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:41:36 2023

@author: DELINTE Nicolas
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom


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
