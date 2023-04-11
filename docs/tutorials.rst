Tutorials
============

This page contains a few tutorials explaining the usage of the main functions in UNRAVEL.

Fixel weight maps
-----------------

Fixel weights maps can be obtained using the :meth:`unravel.core.get_fixel_weight` function::

	import nibabel as nib

	t0 = nib.load("peak_file_0.nii.gz").get_fdata()
	t1 = nib.load("peak_file_1.nii.gz").get_fdata()

	trk = load_tractogram("trk_file.trk", 'same')
	trk.to_vox()
	trk.to_corner()

	fixel_weights,_,_ = get_fixel_weight(trk, [t0, t1])

Example
----------------

A complete example code of the main functions is available in the :class:`unravel.example` file.

.. literalinclude:: ../unravel/example.py
    :linenos:
    :language: python
    :lines: 13-

Adding color to streamline trajectory
-------------------------------------

The `color` parameter can be set to `True` to add directional colors::

	plot_streamline_trajectory(trk, resolution_increase=2,
						streamline_number=500, axis=1,
						color=True, norm_all_voxels=True)

.. image:: imgs/stream.png
	:width: 800
	:align: center

Create GIFs from 3D volumes
---------------------------

Short videos can be created using the :meth:`unravel.viz.convert_to_gif` function::

	import nibabel as nib

	rgb = get_streamline_density(trk, resolution_increase = 4, color = True)
	t1 = nib.load('path_to/registered_T1.nii.gz').get_fdata()

	# rgb = overlap_volumes([rgb, t1], order=0)

	convert_to_gif(rgb, output_folder='output/path', transparency=True, keep_frames=False,extension='gif', axis=0)

.. image:: imgs/rgb3.webp
	:width: 600
	:align: center

.. note::  More tutorials will be added in the near future.
